import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Generating EDA visualizations...")
    df = pd.read_csv('data/raw/titanic.csv')
    
    # Preprocessing for visualization
    df_viz = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].copy()
    df_viz['age'] = df_viz['age'].fillna(df_viz['age'].median())
    df_viz['sex'] = df_viz['sex'].map({'male': 0, 'female': 1})
    
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Pairplot
    print("Generating Pairplot...")
    sns.pairplot(df_viz, hue='survived', palette='husl')
    plt.suptitle('Pairplot of Titanic Features', y=1.02)
    plt.savefig('reports/figures/pairplot.png')
    plt.close()
    
    # 2. Correlation Heatmap
    print("Generating Heatmap...")
    plt.figure(figsize=(10, 8))
    corr = df_viz.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('reports/figures/heatmap.png')
    plt.close()
    
    # 3. Feature Distributions
    print("Generating Feature Distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
    
    for i, feature in enumerate(features):
        row, col = i // 3, i % 3
        sns.histplot(data=df_viz, x=feature, hue='survived', kde=True, ax=axes[row, col], palette='husl')
        axes[row, col].set_title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig('reports/figures/feature_distributions.png')
    plt.close()
    
    print("All visualizations saved to reports/figures/")

if __name__ == '__main__':
    main()