from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.joblib')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'diabetes.csv')

# Load models and raw reference data into RAM on startup (0ms cold start)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df_raw = pd.read_csv(RAW_DATA_PATH)
    raw_targets = df_raw['progression'].values
    print("✓ Production model, scaler, and baseline cohort loaded into RAM.")
except Exception as e:
    model = None
    scaler = None
    raw_targets = np.array([25.0, 140.5, 152.1, 346.0])
    print(f"Warning: Failed to load assets: {e}")

def compute_enriched_vector(raw_inputs):
    """
    Computes the 13-dimensional feature vector from 10 raw clinical inputs:
    Adds:
      1. insulin_res_proxy = bmi * s6
      2. atherogenic_ratio = s2 / (s3 + 1e-5)
      3. metabolic_risk_triad = (bmi * bp) / (s3 + 1e-5)
    """
    age = float(raw_inputs.get('age', 59.0))
    sex = float(raw_inputs.get('sex', 2.0))
    bmi = float(raw_inputs.get('bmi', 32.1))
    bp  = float(raw_inputs.get('bp', 101.0))
    s1  = float(raw_inputs.get('s1', 157.0))
    s2  = float(raw_inputs.get('s2', 93.2))
    s3  = float(raw_inputs.get('s3', 38.0))
    s4  = float(raw_inputs.get('s4', 4.0))
    s5  = float(raw_inputs.get('s5', 4.86))
    s6  = float(raw_inputs.get('s6', 87.0))
    
    # Enriched interaction terms
    insulin_proxy = bmi * s6
    athero_ratio  = s2 / (s3 + 1e-5)
    metabolic_triad = (bmi * bp) / (s3 + 1e-5)
    
    feature_row = [
        age, sex, bmi, bp, s1, s2, s3, s4, s5, s6,
        insulin_proxy, athero_ratio, metabolic_triad
    ]
    return feature_row, {
        'insulin_proxy': insulin_proxy,
        'athero_ratio': athero_ratio,
        'metabolic_triad': metabolic_triad
    }

def generate_clinical_analysis(pred_val, inputs, enriched):
    """Generates detailed biomarker diagnostics and personalized recommendations."""
    # Cohort percentile rank
    percentile = float(np.mean(raw_targets <= pred_val) * 100.0)
    gauge_pct = float(np.clip((pred_val - 25.0) / (346.0 - 25.0) * 100.0, 0.0, 100.0))
    
    is_high_risk = bool(pred_val >= 150.0)
    
    # Status badges
    if pred_val < 100.0:
        status_title = "STABLE / LOW PROGRESSION"
        status_sub = "Disease progression is well-controlled. Cellular metabolic load is minimal."
        theme = "green"
        sound_type = "stable"
    elif pred_val < 150.0:
        status_title = "MODERATE / MONITORING REQUIRED"
        status_sub = "Disease progression is currently within the safe sub-150 margin, but shows emerging risk indicators."
        theme = "amber"
        sound_type = "stable"
    else:
        status_title = "HIGH-RISK ACCELERATED PROGRESSION"
        status_sub = "Critical Alert: Progression index exceeds clinical threshold (150). Accelerated organ & vascular risk."
        theme = "red"
        sound_type = "alert"

    # Specific biomarker audits
    biomarkers = []
    
    # BMI audit
    bmi = inputs['bmi']
    if bmi >= 30.0:
        biomarkers.append({
            'name': 'Body Mass Index (BMI)',
            'val': f"{bmi:.1f} kg/m²",
            'status': 'Elevated (Obesity Class I+)',
            'type': 'danger',
            'detail': 'Elevated visceral adipose tissue drives systemic insulin resistance and chronic cytokine release.'
        })
    elif bmi >= 25.0:
        biomarkers.append({
            'name': 'Body Mass Index (BMI)',
            'val': f"{bmi:.1f} kg/m²",
            'status': 'Borderline Overweight',
            'type': 'warning',
            'detail': 'Moderate excess mass; weight reduction of 5-7% will substantially lower 1-year disease progression velocity.'
        })
    else:
        biomarkers.append({
            'name': 'Body Mass Index (BMI)',
            'val': f"{bmi:.1f} kg/m²",
            'status': 'Optimal / Normal',
            'type': 'good',
            'detail': 'Healthy baseline body composition minimizes mechanical and metabolic tissue stress.'
        })

    # Blood Pressure audit
    bp = inputs['bp']
    if bp >= 100.0:
        biomarkers.append({
            'name': 'Mean Arterial BP',
            'val': f"{bp:.1f} mmHg",
            'status': 'Hypertensive Range',
            'type': 'danger',
            'detail': 'Accelerates diabetic nephropathy and microvascular retinal shear stress.'
        })
    else:
        biomarkers.append({
            'name': 'Mean Arterial BP',
            'val': f"{bp:.1f} mmHg",
            'status': 'Normotensive / Controlled',
            'type': 'good',
            'detail': 'Arterial pressure within manageable clinical parameters.'
        })

    # Atherogenic Ratio (LDL/HDL)
    ar = enriched['athero_ratio']
    if ar >= 3.0:
        biomarkers.append({
            'name': 'Atherogenic Ratio (LDL/HDL)',
            'val': f"{ar:.2f}",
            'status': 'Atherogenic Dyslipidemia',
            'type': 'danger',
            'detail': 'High LDL relative to HDL accelerates vascular plaque deposition and endothelial inflammation.'
        })
    else:
        biomarkers.append({
            'name': 'Atherogenic Ratio (LDL/HDL)',
            'val': f"{ar:.2f}",
            'status': 'Favorable Lipid Balance',
            'type': 'good',
            'detail': 'HDL particle density offers protective reverse cholesterol clearance.'
        })

    # Fasting Glucose (S6)
    s6 = inputs['s6']
    if s6 >= 100.0:
        biomarkers.append({
            'name': 'Fasting Blood Glucose (S6)',
            'val': f"{s6:.1f} mg/dL",
            'status': 'Elevated Glycemia',
            'type': 'danger',
            'detail': 'Persistent circulating glucose increases advanced glycation end-products (AGEs).'
        })
    else:
        biomarkers.append({
            'name': 'Fasting Blood Glucose (S6)',
            'val': f"{s6:.1f} mg/dL",
            'status': 'Controlled Glycemia',
            'type': 'good',
            'detail': 'Basal pancreatic beta-cell insulin secretion is maintaining equilibrium.'
        })

    # Actionable Care Recommendations
    if is_high_risk:
        care_plan = {
            'nutrition': 'Strict low-glycemic Mediterranean dietary protocol (<45% total carbs from high-fiber, unrefined sources). Eliminate all simple sugars and trans-fats to reduce hepatic lipid synthesis.',
            'exercise': 'Prescribe 150 minutes/week of moderate-intensity aerobic training (e.g. brisk walking, cycling) combined with 2-3 sessions of progressive resistance training to upregulate GLUT-4 receptor translocation.',
            'clinical': 'Urgent Endocrinology Consult: Initiate or intensify metformin titration; consider GLP-1 receptor agonist or SGLT2 inhibitor therapy; schedule comprehensive renal microalbuminuria and dilated fundoscopic exams.'
        }
    else:
        care_plan = {
            'nutrition': 'Maintain anti-inflammatory, balanced dietary pattern rich in leafy greens, healthy monounsaturated fats (olive oil, avocados), and lean proteins to sustain current metabolic stability.',
            'exercise': 'Continue routine daily movement (target 8,000–10,000 steps daily) plus light aerobic conditioning to preserve peripheral insulin sensitivity.',
            'clinical': 'Routine Clinical Follow-Up: 6-month checkup with repeated lipid panel and HbA1c screening. Continue current standard-of-care regimen without immediate aggressive drug escalation.'
        }

    return {
        'percentile': round(percentile, 1),
        'gauge_pct': round(gauge_pct, 1),
        'is_high_risk': is_high_risk,
        'status_title': status_title,
        'status_sub': status_sub,
        'theme': theme,
        'sound_type': sound_type,
        'biomarkers': biomarkers,
        'care_plan': care_plan,
        'enriched': {
            'insulin_proxy': round(enriched['insulin_proxy'], 1),
            'athero_ratio': round(enriched['athero_ratio'], 2),
            'metabolic_triad': round(enriched['metabolic_triad'], 1)
        }
    }

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    analysis = None
    input_data = {
        'age': 59.0, 'sex': 2.0, 'bmi': 32.1, 'bp': 101.0,
        's1': 157.0, 's2': 93.2, 's3': 38.0, 's4': 4.0,
        's5': 4.86,  's6': 87.0
    }
    
    if request.method == 'POST':
        try:
            for k in input_data.keys():
                if k in request.form:
                    input_data[k] = float(request.form[k])
                    
            vec, enriched = compute_enriched_vector(input_data)
            vec_scaled = scaler.transform([vec])
            pred_val = float(model.predict(vec_scaled)[0])
            
            prediction = f"{pred_val:.1f}"
            analysis = generate_clinical_analysis(pred_val, input_data, enriched)
        except Exception as e:
            prediction = f"Error: {str(e)}"
            analysis = None
            
    return render_template('index.html', prediction=prediction, analysis=analysis,
                           input_data=input_data)

@app.route('/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for programmatic clinical integration."""
    try:
        data = request.get_json(force=True)
        vec, enriched = compute_enriched_vector(data)
        vec_scaled = scaler.transform([vec])
        pred_val = float(model.predict(vec_scaled)[0])
        analysis = generate_clinical_analysis(pred_val, data, enriched)
        
        return jsonify({
            'status': 'success',
            'progression_score': round(pred_val, 2),
            'is_high_risk': analysis['is_high_risk'],
            'status_title': analysis['status_title'],
            'cohort_percentile': analysis['percentile'],
            'enriched_features': analysis['enriched'],
            'care_plan': analysis['care_plan']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Listen on port 5003 for direct clinical dashboard testing
    app.run(host='0.0.0.0', port=5003, debug=False)
