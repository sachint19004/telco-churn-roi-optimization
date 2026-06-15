import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def generate_visualizations(df):
    """Generates the three core EDA plots for the business presentation."""
    print("Generating EDA visualizations...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 5))

    # Plot 1: Churn vs. Contract Type
    plt.subplot(1, 3, 1)
    sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2')
    plt.title('Churn by Contract Type')

    # Plot 2: Churn vs. Tenure
    plt.subplot(1, 3, 2)
    sns.histplot(data=df, x='tenure', hue='Churn', multiple="stack", palette='Set2')
    plt.title('Churn by Tenure (Months)')

    # Plot 3: Churn vs. Monthly Charges
    plt.subplot(1, 3, 3)
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple="stack", palette='Set2')
    plt.title('Churn by Monthly Charges')

    plt.tight_layout()
    plt.show()

def run_statistical_test(df):
    """Runs a Chi-Square test to prove statistical significance."""
    print("\n--- Running Statistical Proof ---")
    contingency_table = pd.crosstab(df['Contract'], df['Churn'])
    print(contingency_table)
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"\nP-Value: {p_value}")

    if p_value < 0.05:
        print("Conclusion: STATISTICALLY SIGNIFICANT! The CEO needs to know: Contract type definitively impacts Churn.")
    else:
        print("Conclusion: Not Significant. It might just be random noise.")