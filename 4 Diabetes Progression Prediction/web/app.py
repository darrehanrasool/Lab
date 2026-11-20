"""
========================================================================================
BACKEND WEB SERVER & REST API (web/app.py)
========================================================================================
Project: Diabetes Progression Prediction via Multiple Linear Regression
Course:  CAI504C Machine Learning Lab • Semester I • IUST Kashmir
Dataset: Efron et al. (2004) Authentic Clinical Diabetes Cohort (N = 442 Patients)

PURPOSE:
--------
Serves as the production deployment layer:
1. Web User Interface: An interactive clinical prognostic dashboard (GET/POST /).
2. RESTful JSON API: Programmatic endpoint (/predict) for integration with Hospital EHR systems.
3. In-Memory Asset Caching: Loads models and scalers into RAM at startup for zero-latency inference.
4. Comprehensive Clinical Engine: Translates regression outputs into risk stages, biomarker audits,
   cohort percentiles, mathematical linear decompositions, and personalized medical care plans.

KEY QUESTIONS SIR MIGHT ASK:
----------------------------
Q1: Why load the machine learning models at the top of app.py outside route functions?
    -> In production web services, loading models from disk (I/O) inside a route function on every
       HTTP request causes severe latency (50-100ms lag) and memory thrashing. Loading once at
       application startup into RAM allows instant, non-blocking inference (sub-millisecond latency).

Q2: What is the difference between the '/' route and the '/predict' route?
    -> '/' is the Server-Side Rendered (SSR) web dashboard route. It accepts HTML form submissions
       and renders the visual Jinja2 UI template ('index.html').
    -> '/predict' is a RESTful API endpoint. It accepts structured JSON payloads and returns
       standardized JSON responses, allowing integration with mobile apps or clinical software.

Q3: How is the 'Cohort Percentile' calculated?
    -> We calculate the empirical Cumulative Distribution Function (CDF):
           Percentile = (Number of historical patients with progression <= predicted_value) / Total_Patients * 100
       This gives the clinician an immediate benchmark of where this patient stands relative
       to the entire baseline diabetes cohort.
========================================================================================
"""

import os
from typing import Any, Dict, List, Tuple

import joblib  # Model and preprocessor deserialization
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, Response

# -------------------------------------------------------------------------
# 1. FLASK APPLICATION INITIALIZATION & CONFIGURATION
# -------------------------------------------------------------------------
app = Flask(__name__)

# Determine project root directory dynamically
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define absolute paths to persistent assets
MODEL_PATH: str = os.path.join(BASE_DIR, "models", "diabetes_model.joblib")
SCALER_PATH: str = os.path.join(BASE_DIR, "models", "scaler.joblib")
TRIAGE_PATH: str = os.path.join(BASE_DIR, "models", "triage_logistic_model.joblib")
RAW_DATA_PATH: str = os.path.join(BASE_DIR, "data", "raw", "diabetes.csv")

# Ordered list of 10 clinical features matching model schema
FEATURE_NAMES: List[str] = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

# -------------------------------------------------------------------------
# 2. IN-MEMORY MODEL & COHORT CACHING (0ms Cold-Start Architecture)
# -------------------------------------------------------------------------
# We load models into RAM upon application boot. All subsequent HTTP requests
# share these in-memory objects without performing disk reads.
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    triage_model = joblib.load(TRIAGE_PATH)
    df_raw = pd.read_csv(RAW_DATA_PATH)
    raw_targets = df_raw["progression"].values  # Cached for empirical CDF percentile calculations
    print("✓ Successfully loaded OLS regression model, StandardScaler, and Triage classifier into RAM.")
except (EOFError, ImportError, KeyError, OSError, ValueError) as err:
    print(f"Warning: Failed to load serialized assets from disk: {err}")
    model = None
    scaler = None
    triage_model = None
    raw_targets = np.array([25.0, 140.5, 152.1, 346.0])


# -------------------------------------------------------------------------
# 3. CLINICAL PRESET PATIENT PROFILES
# -------------------------------------------------------------------------
# Provides clinicians with validated reference archetypes across the disease spectrum
CLINICAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "mild": {
        "id": "mild",
        "name": "Case 1: Mild / Controlled",
        "badge": "Safe • Low Risk",
        "target_range": "< 100",
        "expected_score": 68.4,
        "description": "Patient 31: Young/middle-aged female with lean BMI (20.3), normotensive BP (71 mmHg), and high protective HDL (66 mg/dL).",
        "inputs": {
            "age": 42.0,
            "sex": 1.0,  # 1.0 = Female
            "bmi": 20.3,
            "bp": 71.0,
            "s1": 161.0,
            "s2": 81.2,
            "s3": 66.0,
            "s4": 2.0,
            "s5": 4.23,
            "s6": 81.0,
        },
    },
    "moderate": {
        "id": "moderate",
        "name": "Case 2: Moderate Progression",
        "badge": "Cohort Avg • Monitor",
        "target_range": "100 – 200",
        "expected_score": 141.8,
        "description": "Patient 66: Middle-aged male with normal BMI (24.7), moderate BP (85 mmHg), borderline elevated LDL, and average triglycerides.",
        "inputs": {
            "age": 46.0,
            "sex": 2.0,  # 2.0 = Male
            "bmi": 24.7,
            "bp": 85.0,
            "s1": 174.0,
            "s2": 123.2,
            "s3": 30.0,
            "s4": 6.0,
            "s5": 4.64,
            "s6": 96.0,
        },
    },
    "severe": {
        "id": "severe",
        "name": "Case 3: Severe Emergency",
        "badge": "Critical Triage Alert",
        "target_range": "≥ 200",
        "expected_score": 258.6,
        "description": "Patient 138: Older male with Class-II obesity (BMI 36.7), Stage-2 hypertension (BP 117 mmHg), hyperlipidemia, and hyperglycemia.",
        "inputs": {
            "age": 58.0,
            "sex": 2.0,  # 2.0 = Male
            "bmi": 36.7,
            "bp": 117.0,
            "s1": 166.0,
            "s2": 93.8,
            "s3": 44.0,
            "s4": 4.0,
            "s5": 4.95,
            "s6": 109.0,
        },
    },
}


def prepare_input_vector(raw_inputs: Dict[str, Any]) -> List[float]:
    """
    Parses and sanitizes 10 clinical input parameters from web form or JSON requests.
    Enforces correct demographic dummy encoding:
      - 1.0 (Female) -> 0.0 (reference baseline)
      - 2.0 (Male)   -> 1.0 (dummy indicator)

    Args:
        raw_inputs (Dict[str, Any]): Dictionary of parameter key-value pairs.

    Returns:
        List[float]: Ordered 10-element numerical feature vector.
    """
    age = float(raw_inputs.get("age", 59.0))
    raw_sex = float(raw_inputs.get("sex", 2.0))

    # Binary dummy variable mapping
    if raw_sex in [1.0, 2.0]:
        sex = 0.0 if raw_sex == 1.0 else 1.0
    else:
        sex = 1.0 if raw_sex == 1.0 else 0.0

    bmi = float(raw_inputs.get("bmi", 32.1))
    bp = float(raw_inputs.get("bp", 101.0))
    s1 = float(raw_inputs.get("s1", 157.0))
    s2 = float(raw_inputs.get("s2", 93.2))
    s3 = float(raw_inputs.get("s3", 38.0))
    s4 = float(raw_inputs.get("s4", 4.0))
    s5 = float(raw_inputs.get("s5", 4.86))
    s6 = float(raw_inputs.get("s6", 87.0))

    return [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]


def generate_clinical_analysis(
    pred_val: float, inputs: Dict[str, Any], scaled_features: pd.DataFrame
) -> Dict[str, Any]:
    """
    Translates raw quantitative regression predictions into actionable medical diagnostics.

    Computes:
      1. Empirical Cohort Percentile (where this patient ranks among 442 patients).
      2. UI Gauge percentage [0% to 100%].
      3. Categorical risk staging:
           - Mild: Progression < 100
           - Moderate: 100 <= Progression < 200
           - Severe: Progression >= 200
      4. Severe Triage probability from the cost-sensitive balanced Logistic Regression model.
      5. Biomarker risk audits (BMI, BP, HDL, Triglycerides, Glucose).
      6. Actionable clinical care recommendations (Nutrition, Exercise, Pharmacotherapy).
      7. White-box linear mathematical decomposition: Impact_j = Beta_j * Z_j.
      8. Clinically enriched metabolic proxies (Insulin resistance proxy, Atherogenic ratio).

    Args:
        pred_val (float): Predicted quantitative 1-year progression score.
        inputs (Dict[str, Any]): Raw physical biomarker values.
        scaled_features (pd.DataFrame): Standardized z-score DataFrame.

    Returns:
        Dict[str, Any]: Structured diagnostic analysis dictionary.
    """
    # 1. Empirical Cohort Percentile via Cumulative Distribution Function
    percentile = float(np.mean(raw_targets <= pred_val) * 100.0)

    # 2. UI Gauge percentage (clipped to bounds [25, 346])
    gauge_pct = float(np.clip((pred_val - 25.0) / (346.0 - 25.0) * 100.0, 0.0, 100.0))

    # 3. Clinical risk classification
    is_high_risk = bool(pred_val >= 150.0)
    is_severe = bool(pred_val >= 200.0)

    # 4. Severe Triage Probability via balanced Logistic Regression
    if triage_model is not None:
        severe_prob = float(triage_model.predict_proba(scaled_features)[0, 1] * 100.0)
    else:
        severe_prob = 80.0 if is_severe else (40.0 if is_high_risk else 15.0)

    # 5. Clinical status badges and alert styling
    if pred_val < 100.0:
        status_title = "STABLE / MILD PROGRESSION"
        status_sub = "Disease progression is well-controlled. Cellular metabolic load is minimal."
        theme = "green"
        sound_type = "stable"
    elif pred_val < 200.0:
        status_title = "MODERATE / MONITORING REQUIRED"
        status_sub = "Disease progression is currently within the moderate margin. Lifestyle intervention recommended."
        theme = "amber"
        sound_type = "stable"
    else:
        status_title = "SEVERE ACCELERATED PROGRESSION"
        status_sub = "Critical Triage Alert: Progression exceeds clinical severity threshold (200). Urgent escalation needed."
        theme = "red"
        sound_type = "alert"

    # 6. Detailed individual biomarker audits
    biomarkers = []

    # 6.1 Body Mass Index (BMI) Audit
    bmi = float(inputs["bmi"])
    if bmi >= 30.0:
        biomarkers.append({
            "name": "Body Mass Index (BMI)",
            "val": f"{bmi:.1f} kg/m²",
            "status": "Elevated (Obesity Class I+)",
            "type": "danger",
            "detail": "Major progression driver (weight: +23.72). Visceral adiposity drives systemic insulin resistance.",
        })
    elif bmi >= 25.0:
        biomarkers.append({
            "name": "Body Mass Index (BMI)",
            "val": f"{bmi:.1f} kg/m²",
            "status": "Borderline Overweight",
            "type": "warning",
            "detail": "Moderate excess mass; a 5-7% weight reduction substantially slows 1-year progression velocity.",
        })
    else:
        biomarkers.append({
            "name": "Body Mass Index (BMI)",
            "val": f"{bmi:.1f} kg/m²",
            "status": "Optimal / Normal",
            "type": "good",
            "detail": "Healthy baseline body composition minimizes metabolic tissue stress.",
        })

    # 6.2 Blood Pressure (BP) Audit
    bp = float(inputs["bp"])
    if bp >= 100.0:
        biomarkers.append({
            "name": "Mean Arterial BP",
            "val": f"{bp:.1f} mmHg",
            "status": "Hypertensive Range",
            "type": "danger",
            "detail": "Key vascular driver (weight: +17.76). Accelerates diabetic nephropathy and microvascular retinal shear stress.",
        })
    else:
        biomarkers.append({
            "name": "Mean Arterial BP",
            "val": f"{bp:.1f} mmHg",
            "status": "Normotensive / Controlled",
            "type": "good",
            "detail": "Arterial pressure within manageable clinical parameters.",
        })

    # 6.3 HDL Cholesterol (S3) Protective Audit
    s3 = float(inputs["s3"])
    if s3 < 40.0:
        biomarkers.append({
            "name": "HDL Cholesterol (S3)",
            "val": f"{s3:.1f} mg/dL",
            "status": "Low HDL (Depleted Protection)",
            "type": "danger",
            "detail": "Protective factor. Low HDL impairs reverse cholesterol transport and endothelial defense.",
        })
    else:
        biomarkers.append({
            "name": "HDL Cholesterol (S3)",
            "val": f"{s3:.1f} mg/dL",
            "status": "Protective HDL Concentration",
            "type": "good",
            "detail": "Healthy HDL particle density provides natural anti-inflammatory and cardioprotective buffering.",
        })

    # 6.4 Log Triglycerides (S5) Audit
    s5 = float(inputs["s5"])
    if s5 >= 4.5:
        biomarkers.append({
            "name": "Log Triglycerides (S5)",
            "val": f"{s5:.2f}",
            "status": "Elevated Serum Lipids",
            "type": "danger",
            "detail": "Dominant metabolic driver (weight: +34.89). Elevated triglycerides drive hypertriglyceridemia and vascular damage.",
        })
    else:
        biomarkers.append({
            "name": "Log Triglycerides (S5)",
            "val": f"{s5:.2f}",
            "status": "Controlled Serum Lipids",
            "type": "good",
            "detail": "Circulating lipid levels within metabolic stability threshold.",
        })

    # 6.5 Fasting Blood Glucose (S6) Audit
    s6 = float(inputs["s6"])
    if s6 >= 100.0:
        biomarkers.append({
            "name": "Fasting Blood Glucose (S6)",
            "val": f"{s6:.1f} mg/dL",
            "status": "Elevated Glycemia",
            "type": "warning",
            "detail": "Persistent circulating glucose increases advanced glycation end-products (AGEs).",
        })
    else:
        biomarkers.append({
            "name": "Fasting Blood Glucose (S6)",
            "val": f"{s6:.1f} mg/dL",
            "status": "Controlled Glycemia",
            "type": "good",
            "detail": "Basal pancreatic beta-cell insulin secretion is maintaining equilibrium.",
        })

    # 7. Actionable Clinical Care Recommendations
    if is_severe:
        care_plan = {
            "nutrition": "Strict low-glycemic Mediterranean dietary protocol (<45% total carbs from high-fiber, unrefined sources). Eliminate simple sugars and saturated fats to reduce hepatic lipid synthesis.",
            "exercise": "Prescribe 150 min/week moderate-intensity aerobic training combined with 2-3 sessions of resistance training to upregulate GLUT-4 receptor translocation.",
            "clinical": "Urgent Endocrinology Triage: Initiate or intensify metformin titration; consider GLP-1 receptor agonist or SGLT2 inhibitor; schedule microalbuminuria and dilated fundoscopic exams.",
        }
    elif is_high_risk:
        care_plan = {
            "nutrition": "Moderate carbohydrate restriction, increase dietary soluble fiber (>30g/day) to blunt postprandial glucose excursions and lower atherogenic lipids.",
            "exercise": "Structured physical activity: 30 minutes of daily brisk walking plus weekend resistance exercise.",
            "clinical": "Closer 3-month clinical monitoring: evaluate HbA1c and repeat lipid panel to prevent progression across the severity threshold.",
        }
    else:
        care_plan = {
            "nutrition": "Maintain anti-inflammatory, balanced dietary pattern rich in leafy vegetables, monounsaturated fats (olive oil, avocados), and lean proteins.",
            "exercise": "Continue routine daily movement (target 8,000–10,000 steps daily) to preserve peripheral insulin sensitivity.",
            "clinical": "Standard annual routine checkup. Continue current metabolic health maintenance without pharmaceutical escalation.",
        }

    # 8. White-Box Linear Decomposition: Impact_j = Beta_j * Z_j
    decomposition = []
    if model is not None:
        intercept = float(model.intercept_)
        for name, w, z in zip(FEATURE_NAMES, model.coef_, scaled_features.values[0]):
            decomposition.append({
                "feature": name,
                "weight": round(float(w), 2),
                "z_score": round(float(z), 2),
                "impact": round(float(w * z), 2),
            })
    else:
        intercept = 151.21

    return {
        "percentile": round(percentile, 1),
        "gauge_pct": round(gauge_pct, 1),
        "is_high_risk": is_high_risk,
        "is_severe": is_severe,
        "severe_prob": round(severe_prob, 1),
        "status_title": status_title,
        "status_sub": status_sub,
        "theme": theme,
        "sound_type": sound_type,
        "biomarkers": biomarkers,
        "care_plan": care_plan,
        "intercept": round(intercept, 2),
        "decomposition": decomposition,
        "enriched": {
            "insulin_proxy": round(float(inputs["bmi"]) * float(inputs["s6"]), 1),
            "athero_ratio": round(float(inputs["s2"]) / (float(inputs["s3"]) + 1e-5), 2),
            "metabolic_triad": round(
                (float(inputs["bmi"]) * float(inputs["bp"])) / (float(inputs["s3"]) + 1e-5), 1
            ),
        },
    }


# -------------------------------------------------------------------------
# 4. ROUTE DEFINITIONS: WEB DASHBOARD & REST API
# -------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home() -> str:
    """
    Main web application route.
    Handles GET requests (loading default or preset cases) and POST requests (user form submissions).
    Renders 'templates/index.html' with dynamic prognostic evaluation.
    """
    prediction = None
    analysis = None

    # Determine selected preset case from query params or form data
    selected_case = request.args.get("case") or request.form.get("selectedCase", "severe")

    # Load default inputs from clinical presets
    if selected_case in CLINICAL_PRESETS:
        input_data = CLINICAL_PRESETS[selected_case]["inputs"].copy()
    else:
        input_data = CLINICAL_PRESETS["severe"]["inputs"].copy()
        selected_case = "severe"

    # If the user submitted custom values via the web form (POST)
    if request.method == "POST":
        selected_case = request.form.get("selectedCase", selected_case)
        for k in list(input_data.keys()):
            if k in request.form:
                try:
                    input_data[k] = float(request.form[k])
                except (ValueError, TypeError):
                    pass

    # Execute model inference pipeline
    try:
        vec = prepare_input_vector(input_data)
        df_in = pd.DataFrame([vec], columns=FEATURE_NAMES)
        vec_scaled = scaler.transform(df_in)
        df_scaled = pd.DataFrame(vec_scaled, columns=FEATURE_NAMES)
        pred_val = float(model.predict(df_scaled)[0])

        prediction = f"{pred_val:.1f}"
        analysis = generate_clinical_analysis(pred_val, input_data, df_scaled)
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as err:
        prediction = f"Error: {err!s}"
        analysis = None

    # Render template using Jinja2
    return render_template(
        "index.html",
        prediction=prediction,
        analysis=analysis,
        input_data=input_data,
        selected_case=selected_case,
        presets=CLINICAL_PRESETS,
    )


@app.route("/predict", methods=["POST"])
def api_predict() -> Tuple[Response, int]:
    """
    REST API endpoint for programmatic clinical integration (EHR systems).
    Accepts JSON payload of patient vitals and returns structured diagnostic JSON.
    """
    try:
        data = request.get_json(force=True)
        vec = prepare_input_vector(data)
        df_in = pd.DataFrame([vec], columns=FEATURE_NAMES)
        vec_scaled = scaler.transform(df_in)
        df_scaled = pd.DataFrame(vec_scaled, columns=FEATURE_NAMES)
        pred_val = float(model.predict(df_scaled)[0])
        analysis = generate_clinical_analysis(pred_val, data, df_scaled)

        return (
            jsonify({
                "status": "success",
                "progression_score": round(pred_val, 2),
                "is_high_risk": analysis["is_high_risk"],
                "is_severe": analysis["is_severe"],
                "severe_triage_probability": analysis["severe_prob"],
                "status_title": analysis["status_title"],
                "cohort_percentile": analysis["percentile"],
                "linear_decomposition": analysis["decomposition"],
                "care_plan": analysis["care_plan"],
            }),
            200,
        )
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as err:
        return jsonify({"status": "error", "message": str(err)}), 400


if __name__ == "__main__":
    # Start web server on port 5003 for clinical dashboard testing
    app.run(host="0.0.0.0", port=5003, debug=False)
