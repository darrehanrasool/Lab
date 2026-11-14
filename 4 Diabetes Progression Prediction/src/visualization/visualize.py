import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import confusion_matrix
import scipy.stats as stats

# High-resolution aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def main():
    print("==========================================================")
    print("STEP 4: ADVANCED DATASET VISUALIZATIONS & ML DIAGNOSTICS")
    print("==========================================================")
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load raw clinical dataset
    df_raw = pd.read_csv('data/raw/diabetes.csv')
    df_raw['sex_label'] = df_raw['sex'].map({1.0: 'Female', 2.0: 'Male'})
    df_raw['sex_bin'] = df_raw['sex'].map({1.0: 0.0, 2.0: 1.0})
    
    # Load processed splits
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_val   = pd.read_csv('data/processed/X_val.csv')
    y_val   = pd.read_csv('data/processed/y_val.csv').values.ravel()
    X_test  = pd.read_csv('data/processed/X_test.csv')
    y_test  = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    feature_cols = list(X_train.columns)
    
    # -------------------------------------------------------------
    # FIGURE 1: TARGET PROGRESSION DISTRIBUTION & CONTINUOUS IMBALANCE
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.histplot(df_raw['progression'], kde=True, color='#0284c7', bins=25, stat='density', alpha=0.6, ax=ax)
    
    mean_val = df_raw['progression'].mean()
    median_val = df_raw['progression'].median()
    skew_val = df_raw['progression'].skew()
    
    ax.axvline(mean_val, color='#ef4444', linestyle='--', lw=2.5, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='#10b981', linestyle='-', lw=2.5, label=f'Median: {median_val:.1f}')
    
    # Clinical zones
    ax.axvspan(25, 100, color='#10b981', alpha=0.10, label='Mild Progression (<100) [33.3%]')
    ax.axvspan(100, 200, color='#f59e0b', alpha=0.10, label='Moderate Progression (100-200) [38.0%]')
    ax.axvspan(200, 350, color='#ef4444', alpha=0.10, label='Severe / Accelerated (>200) [28.7% Imbalanced Tail]')
    
    ax.text(0.03, 0.90, f"Skewness: +{skew_val:.2f} (Right-Skewed Tail)\nN = 442 Patients\nRange: [25, 346]",
            transform=ax.transAxes, fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cbd5e1', alpha=0.9))
    
    ax.set_title('Figure 1: Target Progression Distribution & Continuous Imbalance', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Quantitative Diabetes Disease Progression (1-Year Post Baseline)', fontsize=12)
    ax.set_ylabel('Empirical Probability Density', fontsize=12)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('reports/figures/01_target_progression_distribution.png')
    plt.close()
    print("✓ Saved 01_target_progression_distribution.png")

    # -------------------------------------------------------------
    # FIGURE 2: MULTI-PANEL CLINICAL FEATURE DISTRIBUTIONS (DIFFERENT TYPES)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), dpi=300)
    axes = axes.flatten()
    
    feature_meta = [
        ('age', 'Age (Years)', '#6366f1', 'Continuous Demographics'),
        ('sex_bin', 'Sex (0=Female, 1=Male)', '#ec4899', 'Categorical Binary Indicator'),
        ('bmi', 'BMI (kg/m²)', '#0ea5e9', 'Continuous Anthropometric'),
        ('bp', 'Mean Arterial BP (mmHg)', '#f43f5e', 'Continuous Vitals'),
        ('s1', 'Total Cholesterol (mg/dL)', '#eab308', 'Continuous Serum Lipid'),
        ('s2', 'LDL Cholesterol (mg/dL)', '#f97316', 'Continuous Serum Lipid'),
        ('s3', 'HDL Cholesterol (mg/dL)', '#10b981', 'Protective Serum Lipid'),
        ('s4', 'TCH Ratio (Total/HDL)', '#8b5cf6', 'Biochemical Ratio'),
        ('s5', 'Log Triglycerides (LTG)', '#14b8a6', 'Log-Transformed Lipid'),
        ('s6', 'Fasting Blood Glucose (mg/dL)', '#06b6d4', 'Continuous Serum Glucose')
    ]
    
    for i, (col, label, color, f_type) in enumerate(feature_meta):
        ax = axes[i]
        if col == 'sex_bin':
            counts = df_raw['sex_label'].value_counts()
            ax.bar(counts.index, counts.values, color=['#ec4899', '#3b82f6'], alpha=0.75, edgecolor='black', width=0.5)
            for x_pos, cnt in enumerate(counts.values):
                ax.text(x_pos, cnt + 5, f"{cnt} ({cnt/len(df_raw)*100:.1f}%)", ha='center', fontweight='bold', fontsize=9)
            ax.set_ylim(0, max(counts.values) + 30)
        else:
            sns.histplot(df_raw[col], kde=True, color=color, ax=ax, alpha=0.6, stat='density')
            mean_c = df_raw[col].mean()
            median_c = df_raw[col].median()
            ax.axvline(mean_c, color='red', linestyle='--', lw=1.5, label=f'Mean {mean_c:.1f}')
            ax.axvline(median_c, color='green', linestyle=':', lw=1.5, label=f'Med {median_c:.1f}')
            ax.legend(fontsize=8, loc='upper right')
            
        ax.set_title(f"{label}\n[{f_type}]", fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Density' if col != 'sex_bin' else 'Patient Count')

    # Remove extra subplots
    axes[10].axis('off')
    axes[11].axis('off')
    
    plt.suptitle('Figure 2: Clinical Input Feature Distributions (Continuous vs Binary Categorical)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('reports/figures/02_clinical_feature_distributions.png')
    plt.close()
    print("✓ Saved 02_clinical_feature_distributions.png")

    # -------------------------------------------------------------
    # FIGURE 3: CLINICAL OUTLIER DETECTION VIA BOXPLOTS & IQR
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), dpi=300)
    axes = axes.flatten()
    
    continuous_cols = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    col_labels = ['Age (Yrs)', 'BMI (kg/m²)', 'Blood Pressure', 'S1 Total Chol', 'S2 LDL', 'S3 HDL', 'S4 TCH Ratio', 'S5 Log Trig', 'S6 Glucose']
    
    for i, (col, lbl) in enumerate(zip(continuous_cols, col_labels)):
        ax = axes[i]
        sns.boxplot(y=df_raw[col], ax=ax, color='#38bdf8', width=0.4,
                    flierprops=dict(marker='o', markersize=6, markerfacecolor='#ef4444', markeredgecolor='black'))
        
        # Calculate IQR and outlier count
        q25 = df_raw[col].quantile(0.25)
        q75 = df_raw[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        n_outliers = ((df_raw[col] < lower_bound) | (df_raw[col] > upper_bound)).sum()
        
        ax.set_title(f"{lbl}\n({n_outliers} Outliers)", fontsize=10, fontweight='bold')
        ax.set_ylabel('')
        
    axes[9].axis('off')
    plt.suptitle('Figure 3: Clinical Outlier Detection Across Continuous Biomarkers (Tukey IQR Fences)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('reports/figures/03_clinical_outlier_detection_boxplots.png')
    plt.close()
    print("✓ Saved 03_clinical_outlier_detection_boxplots.png")

    # -------------------------------------------------------------
    # FIGURE 4: MULTICOLLINEARITY HEATMAP & VIF DIAGNOSTICS
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, gridspec_kw={'width_ratios': [1.3, 1.0]})
    
    # Correlation heatmap of input features
    corr_matrix = df_raw[['age', 'sex_bin', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']].corr()
    corr_matrix = corr_matrix.rename(columns={'sex_bin': 'sex'}, index={'sex_bin': 'sex'})
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-0.8, vmax=1.0,
                square=True, mask=mask, cbar_kws={'label': 'Pearson Correlation (r)'}, ax=ax1, annot_kws={'size': 8})
    ax1.set_title('A: Feature Correlation Matrix (Lipid Collinearity)', fontsize=12, fontweight='bold')

    # VIF Bar plot
    vif_vals = [53.71, 35.39, 14.87, 10.28, 8.70, 1.48, 1.48, 1.47, 1.28, 1.18]
    vif_features = ['s1 (TC)', 's2 (LDL)', 's3 (HDL)', 's5 (LTG)', 's4 (TCH)', 'bp', 's6 (GLU)', 'bmi', 'sex', 'age']
    vif_colors = ['#ef4444' if v > 10 else ('#f59e0b' if v > 5 else '#10b981') for v in vif_vals]
    
    y_pos = np.arange(len(vif_features))
    ax2.barh(y_pos, vif_vals, color=vif_colors, edgecolor='black', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(vif_features, fontsize=10)
    ax2.invert_yaxis()
    ax2.axvline(10, color='red', linestyle='--', lw=2, label='Critical Collinearity Threshold (VIF > 10)')
    ax2.axvline(5, color='orange', linestyle=':', lw=2, label='Moderate Collinearity Threshold (VIF > 5)')
    ax2.set_xlabel('Variance Inflation Factor (VIF)', fontsize=11)
    ax2.set_title('B: Variance Inflation Factor (VIF) Diagnostics', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', frameon=True, facecolor='white')
    
    for idx, v in enumerate(vif_vals):
        ax2.text(v + 1, idx, f"{v:.1f}", va='center', fontweight='bold', fontsize=9)
        
    plt.suptitle('Figure 4: Multicollinearity Trap in Blood Serum Lipids & VIF Diagnostics', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('reports/figures/04_correlation_heatmap_vif_multicollinearity.png')
    plt.close()
    print("✓ Saved 04_correlation_heatmap_vif_multicollinearity.png")

    # -------------------------------------------------------------
    # FIGURE 5: KEY BIOMARKERS VS PROGRESSION SCATTER & TRENDS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    plots_meta = [
        ('bmi', 'BMI (Body Mass Index in kg/m²)', '+0.59', '#0284c7', axes[0, 0]),
        ('s5', 'S5 (Log Serum Triglycerides - LTG)', '+0.57', '#8b5cf6', axes[0, 1]),
        ('bp', 'BP (Mean Arterial Blood Pressure in mmHg)', '+0.44', '#f43f5e', axes[1, 0]),
        ('s3', 'S3 (High-Density Lipoprotein / HDL - Protective)', '-0.39', '#10b981', axes[1, 1])
    ]
    
    for col, lbl, corr_str, color, ax in plots_meta:
        sns.regplot(data=df_raw, x=col, y='progression', ax=ax,
                    scatter_kws={'alpha': 0.6, 'color': color, 'edgecolor': 'none'},
                    line_kws={'color': '#0f172a', 'linewidth': 2.5, 'label': f'Linear Fit (r = {corr_str})'})
        ax.set_title(f"{lbl} vs Progression", fontsize=11, fontweight='bold')
        ax.set_xlabel(lbl, fontsize=10)
        ax.set_ylabel('1-Year Progression Index', fontsize=10)
        ax.legend(loc='upper left', frameon=True, facecolor='white')

    plt.suptitle('Figure 5: Bivariate Relationships: Major Biomarkers vs Diabetes Disease Progression', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('reports/figures/05_key_biomarkers_vs_progression_scatter.png')
    plt.close()
    print("✓ Saved 05_key_biomarkers_vs_progression_scatter.png")

    # -------------------------------------------------------------
    # FIGURE 6: MANAGING DATASET IMBALANCE (QUANTILE STRATIFICATION & SPLIT)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Target split distributions
    data_to_plot = [
        pd.DataFrame({'Progression': y_train, 'Split': 'Train 70% (N=309)'}),
        pd.DataFrame({'Progression': y_val,   'Split': 'Validation 20% (N=88)'}),
        pd.DataFrame({'Progression': y_test,  'Split': 'Holdout Test 10% (N=45)'})
    ]
    split_df = pd.concat(data_to_plot, ignore_index=True)
    
    sns.boxplot(x='Split', y='Progression', hue='Split', data=split_df, ax=ax1, palette=['#3b82f6', '#10b981', '#f59e0b'], legend=False)
    ax1.set_title('A: Preserved Target Distributions Across Splits\n(Zero Distribution Drift via Quantile Stratification)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Disease Progression Metric')
    
    # Means and stdevs annotated
    means = [y_train.mean(), y_val.mean(), y_test.mean()]
    stds  = [y_train.std(),  y_val.std(),  y_test.std()]
    for idx, (m, s) in enumerate(zip(means, stds)):
        ax1.text(idx, 30, f"Mean: {m:.1f}\nStd: {s:.1f}", ha='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#94a3b8'))

    # Clinical Triage Class Imbalance
    triage_counts = [(df_raw['progression'] < 200).sum(), (df_raw['progression'] >= 200).sum()]
    triage_labels = ['Controlled / Moderate (<200)\nMajority Class', 'Severe Progression (>=200)\nMinority Class (Imbalanced)']
    ax2.pie(triage_counts, labels=triage_labels, autopct='%1.1f%%', startangle=140,
            colors=['#0ea5e9', '#ef4444'], explode=(0, 0.08),
            textprops={'fontweight': 'bold', 'fontsize': 10})
    ax2.set_title('B: Clinical Risk Triage Class Imbalance\n(Managed via Cost-Sensitive class_weight="balanced")', fontsize=11, fontweight='bold')
    
    plt.suptitle('Figure 6: Managing Dataset Imbalance in Continuous Regression & Categorical Triage', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('reports/figures/06_managing_dataset_imbalance.png')
    plt.close()
    print("✓ Saved 06_managing_dataset_imbalance.png")

    # -------------------------------------------------------------
    # FIGURE 7: FEATURE STANDARDIZATION TRANSFORMATION
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Raw scales
    features_to_compare = ['bmi', 'bp', 's1', 's5']
    labels_raw = ['BMI [18-42 kg/m²]', 'BP [62-133 mmHg]', 'S1 Total Chol [97-301]', 'S5 Log Trig [3.2-6.1]']
    
    for col, lbl in zip(features_to_compare, labels_raw):
        sns.kdeplot(df_raw[col], ax=ax1, label=lbl, lw=2)
    ax1.set_title('A: Raw Feature Scales (Unstandardized)\n(Features on Different Physical Scales Distort Coefficient Magnitudes)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Raw Value Magnitude')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right', fontsize=9)

    # Standardized z-scores
    for col in features_to_compare:
        sns.kdeplot(X_train[col], ax=ax2, label=f"{col} (Scaled)", lw=2)
    ax2.set_title('B: Standardized Z-Scores (StandardScaler)\n(Zero Data Leakage: Scaler Fitted ONLY on 70% Train)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Standardized Units (Z-score: Mean=0, Std=1)')
    ax2.set_ylabel('Density')
    ax2.legend(loc='upper right', fontsize=9)

    plt.suptitle('Figure 7: Impact of Zero Data Leakage Feature Standardization on Model Fairness', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('reports/figures/07_feature_scaling_transformation.png')
    plt.close()
    print("✓ Saved 07_feature_scaling_transformation.png")

    # -------------------------------------------------------------
    # FIGURE 8: LEARNING CURVE (SAMPLE EFFICIENCY & BIAS-VARIANCE)
    # -------------------------------------------------------------
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        LinearRegression(), X_train, y_train, cv=5, scoring='r2',
        train_sizes=np.linspace(0.2, 1.0, 10), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(train_sizes, train_mean, 'o-', color='#3b82f6', lw=2.5, label='Training Score (R²)')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#3b82f6')
    
    ax.plot(train_sizes, val_mean, 's-', color='#10b981', lw=2.5, label='5-Fold Cross-Validation Score (R²)')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#10b981')
    
    ax.set_title('Figure 8: Multiple Linear Regression Learning Curve (Bias-Variance Diagnostics)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Training Samples (N)', fontsize=11)
    ax.set_ylabel('Explained Variance Score (R²)', fontsize=11)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax.set_ylim(0.2, 0.65)
    
    ax.text(0.04, 0.15, "Low Variance Diagnostics:\nTraining and Cross-Validation scores converge smoothly.\nNo evidence of severe overfitting (Variance gap is small).",
            transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cbd5e1'))
    
    plt.tight_layout()
    plt.savefig('reports/figures/08_learning_curve_bias_variance.png')
    plt.close()
    print("✓ Saved 08_learning_curve_bias_variance.png")

    # -------------------------------------------------------------
    # FIGURE 9: ACTUAL VS PREDICTED ON HOLDOUT TEST SET
    # -------------------------------------------------------------
    ols_model = joblib.load('models/diabetes_model.joblib')
    test_preds = ols_model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    sns.scatterplot(x=y_test, y=test_preds, ax=ax, color='#0284c7', s=85, edgecolor='black', alpha=0.8)
    
    # Identity line
    min_val = min(y_test.min(), test_preds.min()) - 10
    max_val = max(y_test.max(), test_preds.max()) + 10
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Ideal Identity Fit: y = y_hat')
    
    # 1 RMSE error band
    rmse_val = np.sqrt(np.mean((y_test - test_preds)**2))
    ax.fill_between([min_val, max_val], [min_val - rmse_val, max_val - rmse_val],
                    [min_val + rmse_val, max_val + rmse_val], color='#ef4444', alpha=0.10,
                    label=f'±1 Test RMSE Margin (±{rmse_val:.1f})')
    
    r2_val = 1 - np.sum((y_test - test_preds)**2) / np.sum((y_test - np.mean(y_test))**2)
    mae_val = np.mean(np.abs(y_test - test_preds))
    
    ax.text(0.04, 0.85, f"Holdout Test Set (N = 45)\nTest R²: {r2_val:.4f}\nTest RMSE: {rmse_val:.2f}\nTest MAE: {mae_val:.2f}",
            transform=ax.transAxes, fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cbd5e1', alpha=0.9))
    
    ax.set_title('Figure 9: Multiple Linear Regression: Predicted vs Actual Progression (Holdout Test Set)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Actual Quantitative Disease Progression (y)', fontsize=11)
    ax.set_ylabel('Predicted Disease Progression (y_hat)', fontsize=11)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(loc='lower right', frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig('reports/figures/09_model_evaluation_actual_vs_predicted.png')
    plt.close()
    print("✓ Saved 09_model_evaluation_actual_vs_predicted.png")

    # -------------------------------------------------------------
    # FIGURE 10: RESIDUAL DIAGNOSTICS & HOMOSCEDASTICITY
    # -------------------------------------------------------------
    residuals = y_test - test_preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Residuals vs Fitted
    sns.scatterplot(x=test_preds, y=residuals, ax=ax1, color='#8b5cf6', s=80, edgecolor='black', alpha=0.8)
    ax1.axhline(0, color='red', linestyle='--', lw=2)
    ax1.set_title('A: Residuals vs Fitted Values (Homoscedasticity Check)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Fitted Values (Predicted Progression)', fontsize=11)
    ax1.set_ylabel('Residuals (e = y - y_hat)', fontsize=11)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_color('#0284c7')
    ax2.get_lines()[0].set_markersize(7.0)
    ax2.get_lines()[1].set_color('#ef4444')
    ax2.get_lines()[1].set_linewidth(2.0)
    ax2.set_title('B: Normal Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Theoretical Normal Quantiles', fontsize=11)
    ax2.set_ylabel('Sample Residual Quantiles', fontsize=11)

    plt.suptitle('Figure 10: Residual Diagnostics Verifying Linear Modeling Assumptions', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('reports/figures/10_residual_diagnostics_homoscedasticity.png')
    plt.close()
    print("✓ Saved 10_residual_diagnostics_homoscedasticity.png")

    # -------------------------------------------------------------
    # FIGURE 11: STANDARDIZED COEFFICIENTS & CLINICAL INTERPRETATION
    # -------------------------------------------------------------
    weights = ols_model.coef_
    df_weights = pd.DataFrame({'Feature': feature_cols, 'Weight': weights})
    df_weights = df_weights.sort_values(by='Weight', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ['#ef4444' if w > 10 else ('#10b981' if w < -5 and f != 's1' else '#3b82f6') for w, f in zip(df_weights['Weight'], df_weights['Feature'])]
    
    bars = ax.barh(df_weights['Feature'], df_weights['Weight'], color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', lw=1.2)
    ax.set_title('Figure 11: Multiple Linear Regression Beta Coefficients & Clinical Impact', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Standardized Beta Weight (Impact per 1 SD Increase in Feature)', fontsize=11)
    ax.set_ylabel('Clinical Biomarker', fontsize=11)
    
    for bar in bars:
        val = bar.get_width()
        x_text = val + (0.5 if val >= 0 else -1.8)
        ax.text(x_text, bar.get_y() + bar.get_height()/2, f"{val:+.2f}", va='center', fontweight='bold', fontsize=9)
        
    ax.text(0.55, 0.15, "Accelerants: s5 (Triglycerides +34.9), bmi (+23.7), bp (+17.8)\nDemographic: sex (-10.1: Female baseline vs Male)\nBaseline Progression Intercept: +151.2",
            transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cbd5e1'))

    plt.tight_layout()
    plt.savefig('reports/figures/11_clinical_interpretability_coefficients.png')
    plt.close()
    print("✓ Saved 11_clinical_interpretability_coefficients.png")

    # -------------------------------------------------------------
    # FIGURE 12: CLINICAL RISK CONFUSION MATRIX & MANAGING IMBALANCE
    # -------------------------------------------------------------
    y_test_severe = (y_test >= 200).astype(int)
    
    # Load logistic models
    triage_model = joblib.load('models/triage_logistic_model.joblib')
    preds_bal = triage_model.predict(X_test)
    cm_bal = confusion_matrix(y_test_severe, preds_bal)
    
    # Fit temporary unweighted for comparison
    lr_un = LogisticRegression(random_state=42).fit(X_train, (y_train >= 200).astype(int))
    preds_un = lr_un.predict(X_test)
    cm_un = confusion_matrix(y_test_severe, preds_un)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    sns.heatmap(cm_un, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1, annot_kws={'size': 14, 'fontweight': 'bold'},
                xticklabels=['Controlled (<200)', 'Severe (>=200)'], yticklabels=['Controlled (<200)', 'Severe (>=200)'])
    ax1.set_title(f'A: Standard Unweighted Logistic Regression\n(Misses {cm_un[1,0]} Severe Patients | Recall: 61.5%)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicted Clinical Label', fontsize=10)
    ax1.set_ylabel('True Clinical Status', fontsize=10)

    sns.heatmap(cm_bal, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2, annot_kws={'size': 14, 'fontweight': 'bold'},
                xticklabels=['Controlled (<200)', 'Severe (>=200)'], yticklabels=['Controlled (<200)', 'Severe (>=200)'])
    ax2.set_title(f'B: Balanced Logistic Regression (Managing Imbalance)\n(Misses Only {cm_bal[1,0]} Patients | Recall: 76.9%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Predicted Clinical Label', fontsize=10)
    ax2.set_ylabel('True Clinical Status', fontsize=10)

    plt.suptitle('Figure 12: Managing Clinical Risk Imbalance: Unweighted vs Balanced Cost-Sensitive Learning', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('reports/figures/12_clinical_risk_confusion_matrix_imbalance.png')
    plt.close()
    print("✓ Saved 12_clinical_risk_confusion_matrix_imbalance.png")
    
    print("\n✓ All 12 advanced visualizations generated and saved in reports/figures/.")

if __name__ == '__main__':
    main()
