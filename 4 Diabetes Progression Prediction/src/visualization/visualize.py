import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import confusion_matrix
import scipy.stats as stats

# Set styling aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def main():
    print("==========================================================")
    print("STEP 4: GENERATING ADVANCED CLINICAL VISUALIZATIONS")
    print("==========================================================")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load unscaled raw data for clinical exploration
    df_raw = pd.read_csv('data/raw/diabetes.csv')
    
    # -------------------------------------------------------------
    # FIGURE 1: TARGET PROGRESSION DISTRIBUTION & CLINICAL RISK STRATIFICATION
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.histplot(df_raw['progression'], kde=True, color='#0284c7', bins=25, stat='density', alpha=0.6, ax=ax)
    
    mean_val = df_raw['progression'].mean()
    median_val = df_raw['progression'].median()
    ax.axvline(mean_val, color='#ef4444', linestyle='--', lw=2.5, label=f'Mean Progression: {mean_val:.1f}')
    ax.axvline(median_val, color='#10b981', linestyle='-', lw=2.5, label=f'Median Progression: {median_val:.1f}')
    
    # Clinical zones
    ax.axvspan(25, 100, color='#10b981', alpha=0.10, label='Mild Progression (<100)')
    ax.axvspan(100, 200, color='#f59e0b', alpha=0.10, label='Moderate Progression (100-200)')
    ax.axvspan(200, 350, color='#ef4444', alpha=0.10, label='Severe / Accelerated (>200)')
    
    ax.set_title('Diabetes Disease Progression (1-Year Post Baseline) Distribution', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Quantitative Disease Progression Metric (Scale: 25 - 346)', fontsize=12)
    ax.set_ylabel('Empirical Probability Density', fontsize=12)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('reports/figures/01_target_progression_distribution.png')
    plt.close()
    print("✓ Saved 01_target_progression_distribution.png")

    # -------------------------------------------------------------
    # FIGURE 2: MULTI-PANEL CLINICAL FEATURE DISTRIBUTIONS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    features_to_plot = [
        ('age', 'Patient Age (Years)', '#6366f1'),
        ('bmi', 'Body Mass Index (BMI in kg/m²)', '#0ea5e9'),
        ('bp', 'Mean Arterial Blood Pressure (mmHg)', '#f43f5e'),
        ('s1', 'Total Serum Cholesterol (mg/dL)', '#eab308'),
        ('s5', 'Log Serum Triglycerides (s5)', '#8b5cf6'),
        ('s6', 'Fasting Blood Glucose (mg/dL)', '#10b981')
    ]
    for i, (col, title, color) in enumerate(features_to_plot):
        r, c = i // 3, i % 3
        sns.histplot(df_raw[col], kde=True, color=color, ax=axes[r, c], bins=20)
        axes[r, c].set_title(title, fontsize=11, fontweight='bold')
        axes[r, c].set_ylabel('Patient Count')
    plt.suptitle('Physiological & Biochemical Feature Distributions (Authentic Clinical Units)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig('reports/figures/02_clinical_feature_distributions.png')
    plt.close()
    print("✓ Saved 02_clinical_feature_distributions.png")

    # -------------------------------------------------------------
    # FIGURE 3: CORRELATION HEATMAP & MULTICOLLINEARITY DETECTION
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 9), dpi=300)
    corr = df_raw.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation Coefficient (r)'})
    plt.title('Diabetes Biomarker Correlation Matrix: Identifying Severe Multicollinearity', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('reports/figures/03_correlation_heatmap_multicollinearity.png')
    plt.close()
    print("✓ Saved 03_correlation_heatmap_multicollinearity.png")

    # -------------------------------------------------------------
    # FIGURE 4: BMI VS PROGRESSION JOINT PLOT
    # -------------------------------------------------------------
    g = sns.jointplot(data=df_raw, x='bmi', y='progression', kind='reg',
                      color='#0284c7', height=8, marginal_kws=dict(bins=20, fill=True),
                      scatter_kws={'alpha': 0.6, 'edgecolor': 'k', 's': 45},
                      line_kws={'color': '#ef4444', 'lw': 2.5})
    g.fig.suptitle('Primary Physiological Driver: Body Mass Index (BMI) vs Progression (r = +0.59)', fontsize=13, fontweight='bold', y=1.02)
    g.ax_joint.set_xlabel('Body Mass Index (BMI in kg/m²)', fontsize=11)
    g.ax_joint.set_ylabel('1-Year Disease Progression Index', fontsize=11)
    plt.tight_layout()
    plt.savefig('reports/figures/04_bmi_vs_progression_jointplot.png')
    plt.close()
    print("✓ Saved 04_bmi_vs_progression_jointplot.png")

    # -------------------------------------------------------------
    # FIGURE 5: PAIRWISE BIOMARKER INTERACTIONS BY RISK GROUP
    # -------------------------------------------------------------
    df_plot = df_raw[['bmi', 'bp', 's5', 's6', 'progression']].copy()
    df_plot['Risk_Group'] = pd.cut(df_plot['progression'], bins=[0, 100, 200, 400], labels=['Mild (<100)', 'Moderate (100-200)', 'Severe (>200)'])
    pairplot = sns.pairplot(df_plot, hue='Risk_Group', palette={'Mild (<100)': '#10b981', 'Moderate (100-200)': '#f59e0b', 'Severe (>200)': '#ef4444'},
                            diag_kind='kde', plot_kws={'alpha': 0.7, 's': 40})
    pairplot.fig.suptitle('Pairwise Metabolic Biomarker Clusters Across Progression Severity', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('reports/figures/05_pairplot_key_biomarkers.png')
    plt.close()
    print("✓ Saved 05_pairplot_key_biomarkers.png")

    # -------------------------------------------------------------
    # LOAD TEST RESULTS FOR DIAGNOSTIC FIGURES
    # -------------------------------------------------------------
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test  = pd.read_csv('data/processed/X_test.csv')
    y_test  = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    model = joblib.load('models/diabetes_model.joblib')
    test_preds = model.predict(X_test)

    # -------------------------------------------------------------
    # FIGURE 6: ACTUAL VS PREDICTED REGRESSION SCATTER
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 7), dpi=300)
    plt.scatter(y_test, test_preds, color='#0284c7', edgecolors='k', alpha=0.75, s=70, label='Holdout Test Patients (N=45)')
    min_pt, max_pt = min(y_test.min(), test_preds.min()) - 10, max(y_test.max(), test_preds.max()) + 10
    plt.plot([min_pt, max_pt], [min_pt, max_pt], 'r--', lw=2.5, label='Perfect Prediction Line (y = x)')
    plt.xlabel('Actual Disease Progression Index', fontsize=12)
    plt.ylabel('Model Predicted Progression Index', fontsize=12)
    plt.title('Actual vs. Predicted Diabetes Progression (Holdout Test Set)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/06_actual_vs_predicted_regression.png')
    plt.close()
    print("✓ Saved 06_actual_vs_predicted_regression.png")

    # -------------------------------------------------------------
    # FIGURE 7: RESIDUAL DIAGNOSTICS & HOMOSCEDASTICITY CHECK
    # -------------------------------------------------------------
    residuals = y_test - test_preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # 1. Residuals vs Fitted
    ax1.scatter(test_preds, residuals, color='#8b5cf6', edgecolors='k', alpha=0.75, s=65)
    ax1.axhline(0, color='red', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Progression', fontsize=11)
    ax1.set_ylabel('Residual Error (y - ŷ)', fontsize=11)
    ax1.set_title('Residuals vs Fitted (Homoscedasticity Test)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Q-Q Normal Probability Plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('#0ea5e9')
    ax2.get_lines()[0].set_markeredgecolor('k')
    ax2.get_lines()[0].set_markersize(7.0)
    ax2.get_lines()[1].set_color('red')
    ax2.get_lines()[1].set_linewidth(2.0)
    ax2.set_title('Normal Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('reports/figures/07_residual_diagnostics_homoscedasticity.png')
    plt.close()
    print("✓ Saved 07_residual_diagnostics_homoscedasticity.png")

    # -------------------------------------------------------------
    # FIGURE 8: LEARNING CURVE (BIAS-VARIANCE DIAGNOSIS)
    # -------------------------------------------------------------
    train_sizes, train_scores, cv_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.15, 1.0, 6), scoring='r2', random_state=42
    )
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#ef4444', lw=2.5, label='Training R² Score')
    plt.plot(train_sizes, np.mean(cv_scores, axis=1), 's-', color='#10b981', lw=2.5, label='5-Fold Cross-Validation R² Score')
    plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.15, color='#ef4444')
    plt.fill_between(train_sizes, np.mean(cv_scores, axis=1) - np.std(cv_scores, axis=1),
                     np.mean(cv_scores, axis=1) + np.std(cv_scores, axis=1), alpha=0.15, color='#10b981')
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Learning Curve: Diagnosing Model Generalization (Bias vs. Variance)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/08_learning_curve_bias_variance.png')
    plt.close()
    print("✓ Saved 08_learning_curve_bias_variance.png")

    # -------------------------------------------------------------
    # FIGURE 9: 5-FOLD CROSS-VALIDATION SCORE STABILITY
    # -------------------------------------------------------------
    cv_kfold = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    plt.figure(figsize=(8, 5.5), dpi=300)
    bars = plt.bar([f'Fold {i}' for i in range(1, 6)], cv_kfold, color='#38bdf8', edgecolor='#0f172a', width=0.55)
    mean_cv = np.mean(cv_kfold)
    plt.axhline(mean_cv, color='#ef4444', linestyle='--', lw=2, label=f'Mean R²: {mean_cv:.3f}')
    plt.ylabel('R² Score', fontsize=12)
    plt.title('5-Fold Cross-Validation Stability Across Patient Partitions', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('reports/figures/09_cross_validation_5fold_stability.png')
    plt.close()
    print("✓ Saved 09_cross_validation_5fold_stability.png")

    # -------------------------------------------------------------
    # FIGURE 10: FEATURE IMPORTANCE RANKING
    # -------------------------------------------------------------
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X_train.columns
        sorted_idx = np.argsort(importances)
        plt.figure(figsize=(10, 6.5), dpi=300)
        plt.barh(range(len(sorted_idx)), importances[sorted_idx], color='#0284c7', edgecolor='k', align='center')
        plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx], fontsize=10)
        plt.xlabel('Relative Feature Importance (Mean Decrease in Impurity)', fontsize=12)
        plt.title('Random Forest Feature Importance Ranking for Disease Progression', fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig('reports/figures/10_feature_importances_ranking.png')
        plt.close()
        print("✓ Saved 10_feature_importances_ranking.png")

    # -------------------------------------------------------------
    # FIGURE 11: DUAL FRAMING - CLINICAL RISK CLASSIFICATION CONFUSION MATRIX
    # -------------------------------------------------------------
    threshold = 150.0
    y_test_binary = (y_test >= threshold).astype(int)
    preds_binary  = (test_preds >= threshold).astype(int)
    cm = confusion_matrix(y_test_binary, preds_binary)
    
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={'size': 18, 'weight': 'bold'},
                xticklabels=['Controlled (<150)', 'High Risk (≥150)'],
                yticklabels=['Controlled (<150)', 'High Risk (≥150)'])
    plt.xlabel('Predicted Clinical Progression Category', fontsize=11, fontweight='bold')
    plt.ylabel('Actual Patient True Category', fontsize=11, fontweight='bold')
    plt.title('Dual Diagnostic: Clinical Risk Stratification Matrix (Threshold: 150)', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('reports/figures/11_clinical_risk_confusion_matrix.png')
    plt.close()
    print("✓ Saved 11_clinical_risk_confusion_matrix.png")
    
    print("\n✓ Successfully generated all 11 advanced clinical visualizations in reports/figures/!")

if __name__ == '__main__':
    main()
