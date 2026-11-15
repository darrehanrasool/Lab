import os

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
TRIAGE_PATH = os.path.join(BASE_DIR, "models", "triage_logistic_model.joblib")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "diabetes.csv")

FEATURE_NAMES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

# Load models and raw reference data into RAM on startup (0ms cold start)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    triage_model = joblib.load(TRIAGE_PATH)
    df_raw = pd.read_csv(RAW_DATA_PATH)
    raw_targets = df_raw["progression"].values
    print(
        "✓ Interpretable Multiple Linear Regression model, scaler, and triage model loaded into RAM."
    )
except (EOFError, ImportError, KeyError, OSError, ValueError) as e:
    model = None
    scaler = None
    triage_model = None
    raw_targets = np.array([25.0, 140.5, 152.1, 346.0])
    print(f"Warning: Failed to load assets: {e}")


def prepare_input_vector(raw_inputs):
    """
    Formats the 10 authentic clinical input features:
    Handles sex encoding (1.0 = Female -> 0.0, 2.0 = Male -> 1.0).
    """
    age = float(raw_inputs.get("age", 59.0))
    raw_sex = float(raw_inputs.get("sex", 2.0))
    # Map sex: if 1.0 or 2.0, map to 0.0 (female) or 1.0 (male)
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

    feature_row = [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]
    return feature_row


def generate_clinical_analysis(pred_val, inputs, scaled_features):
    """Generates detailed biomarker diagnostics, linear decomposition, and personalized recommendations."""
    percentile = float(np.mean(raw_targets <= pred_val) * 100.0)
    gauge_pct = float(np.clip((pred_val - 25.0) / (346.0 - 25.0) * 100.0, 0.0, 100.0))

    is_high_risk = bool(pred_val >= 150.0)
    is_severe = bool(pred_val >= 200.0)

    # Severe probability from balanced logistic triage model
    if triage_model is not None:
        severe_prob = float(triage_model.predict_proba(scaled_features)[0, 1] * 100.0)
    else:
        severe_prob = 80.0 if is_severe else (40.0 if is_high_risk else 15.0)

    # Status badges
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

    # Specific biomarker audits
    biomarkers = []

    # BMI audit
    bmi = inputs["bmi"]
    if bmi >= 30.0:
        biomarkers.append(
            {
                "name": "Body Mass Index (BMI)",
                "val": f"{bmi:.1f} kg/m²",
                "status": "Elevated (Obesity Class I+)",
                "type": "danger",
                "detail": "Major progression driver (weight: +23.46). Visceral adipose tissue drives systemic insulin resistance.",
            }
        )
    elif bmi >= 25.0:
        biomarkers.append(
            {
                "name": "Body Mass Index (BMI)",
                "val": f"{bmi:.1f} kg/m²",
                "status": "Borderline Overweight",
                "type": "warning",
                "detail": "Moderate excess mass; a 5-7% weight reduction substantially slows 1-year progression velocity.",
            }
        )
    else:
        biomarkers.append(
            {
                "name": "Body Mass Index (BMI)",
                "val": f"{bmi:.1f} kg/m²",
                "status": "Optimal / Normal",
                "type": "good",
                "detail": "Healthy baseline body composition minimizes metabolic tissue stress.",
            }
        )

    # Blood Pressure audit
    bp = inputs["bp"]
    if bp >= 100.0:
        biomarkers.append(
            {
                "name": "Mean Arterial BP",
                "val": f"{bp:.1f} mmHg",
                "status": "Hypertensive Range",
                "type": "danger",
                "detail": "Key vascular driver (weight: +17.37). Accelerates diabetic nephropathy and microvascular retinal shear stress.",
            }
        )
    else:
        biomarkers.append(
            {
                "name": "Mean Arterial BP",
                "val": f"{bp:.1f} mmHg",
                "status": "Normotensive / Controlled",
                "type": "good",
                "detail": "Arterial pressure within manageable clinical parameters.",
            }
        )

    # HDL (S3) Protective audit
    s3 = inputs["s3"]
    if s3 < 40.0:
        biomarkers.append(
            {
                "name": "HDL Cholesterol (S3)",
                "val": f"{s3:.1f} mg/dL",
                "status": "Low HDL (Depleted Protection)",
                "type": "danger",
                "detail": "Protective factor (weight: -6.26). Low HDL impairs reverse cholesterol transport and endothelial defense.",
            }
        )
    else:
        biomarkers.append(
            {
                "name": "HDL Cholesterol (S3)",
                "val": f"{s3:.1f} mg/dL",
                "status": "Protective HDL Concentration",
                "type": "good",
                "detail": "Healthy HDL particle density provides natural anti-inflammatory and cardioprotective buffering.",
            }
        )

    # Log Triglycerides (S5) audit
    s5 = inputs["s5"]
    if s5 >= 4.5:
        biomarkers.append(
            {
                "name": "Log Triglycerides (S5)",
                "val": f"{s5:.2f}",
                "status": "Elevated Serum Lipids",
                "type": "danger",
                "detail": "Dominant metabolic driver (weight: +24.27). Elevated triglycerides drive hypertriglyceridemia and vascular damage.",
            }
        )
    else:
        biomarkers.append(
            {
                "name": "Log Triglycerides (S5)",
                "val": f"{s5:.2f}",
                "status": "Controlled Serum Lipids",
                "type": "good",
                "detail": "Circulating lipid levels within metabolic stability threshold.",
            }
        )

    # Fasting Glucose (S6) audit
    s6 = inputs["s6"]
    if s6 >= 100.0:
        biomarkers.append(
            {
                "name": "Fasting Blood Glucose (S6)",
                "val": f"{s6:.1f} mg/dL",
                "status": "Elevated Glycemia",
                "type": "warning",
                "detail": "Persistent circulating glucose increases advanced glycation end-products (AGEs).",
            }
        )
    else:
        biomarkers.append(
            {
                "name": "Fasting Blood Glucose (S6)",
                "val": f"{s6:.1f} mg/dL",
                "status": "Controlled Glycemia",
                "type": "good",
                "detail": "Basal pancreatic beta-cell insulin secretion is maintaining equilibrium.",
            }
        )

    # Actionable Care Recommendations
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

    # Linear feature decomposition: impact = weight * z_score
    decomposition = []
    if model is not None:
        intercept = float(model.intercept_)
        for name, w, z in zip(FEATURE_NAMES, model.coef_, scaled_features.values[0]):
            decomposition.append(
                {
                    "feature": name,
                    "weight": round(float(w), 2),
                    "z_score": round(float(z), 2),
                    "impact": round(float(w * z), 2),
                }
            )
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
            "insulin_proxy": round(inputs["bmi"] * inputs["s6"], 1),
            "athero_ratio": round(inputs["s2"] / (inputs["s3"] + 1e-5), 2),
            "metabolic_triad": round(
                (inputs["bmi"] * inputs["bp"]) / (inputs["s3"] + 1e-5), 1
            ),
        },
    }


CLINICAL_PRESETS = {
    "mild": {
        "id": "mild",
        "name": "Case 1: Mild / Controlled",
        "badge": "Safe • Low Risk",
        "target_range": "< 100",
        "expected_score": 68.4,
        "description": "Patient 31: Young/middle-aged female with lean BMI (20.3), normotensive BP (71 mmHg), and high protective HDL (66 mg/dL).",
        "inputs": {
            "age": 42.0,
            "sex": 1.0,
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
            "sex": 2.0,
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
            "sex": 2.0,
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


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    analysis = None
    selected_case = request.args.get("case") or request.form.get("selectedCase", "severe")

    if selected_case in CLINICAL_PRESETS:
        input_data = CLINICAL_PRESETS[selected_case]["inputs"].copy()
    else:
        input_data = CLINICAL_PRESETS["severe"]["inputs"].copy()
        selected_case = "severe"

    if request.method == "POST":
        selected_case = request.form.get("selectedCase", selected_case)
        for k in list(input_data.keys()):
            if k in request.form:
                try:
                    input_data[k] = float(request.form[k])
                except (ValueError, TypeError):
                    pass

    # Always compute prognostic evaluation so dashboard is immediately active and informative
    try:
        vec = prepare_input_vector(input_data)
        df_in = pd.DataFrame([vec], columns=FEATURE_NAMES)
        vec_scaled = scaler.transform(df_in)
        df_scaled = pd.DataFrame(vec_scaled, columns=FEATURE_NAMES)
        pred_val = float(model.predict(df_scaled)[0])

        prediction = f"{pred_val:.1f}"
        analysis = generate_clinical_analysis(pred_val, input_data, df_scaled)
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
        prediction = f"Error: {e!s}"
        analysis = None

    return render_template(
        "index.html",
        prediction=prediction,
        analysis=analysis,
        input_data=input_data,
        selected_case=selected_case,
        presets=CLINICAL_PRESETS,
    )


@app.route("/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic clinical integration."""
    try:
        data = request.get_json(force=True)
        vec = prepare_input_vector(data)
        df_in = pd.DataFrame([vec], columns=FEATURE_NAMES)
        vec_scaled = scaler.transform(df_in)
        df_scaled = pd.DataFrame(vec_scaled, columns=FEATURE_NAMES)
        pred_val = float(model.predict(df_scaled)[0])
        analysis = generate_clinical_analysis(pred_val, data, df_scaled)

        return jsonify(
            {
                "status": "success",
                "progression_score": round(pred_val, 2),
                "is_high_risk": analysis["is_high_risk"],
                "is_severe": analysis["is_severe"],
                "severe_triage_probability": analysis["severe_prob"],
                "status_title": analysis["status_title"],
                "cohort_percentile": analysis["percentile"],
                "linear_decomposition": analysis["decomposition"],
                "care_plan": analysis["care_plan"],
            }
        )
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    # Listen on port 5003 for direct clinical dashboard testing
    app.run(host="0.0.0.0", port=5003, debug=False)
