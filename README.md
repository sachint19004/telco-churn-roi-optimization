# 📊 Data-Driven Customer Churn & Business Campaign ROI Optimization Engine

A cross-disciplinary enterprise case study bridging Exploratory Data Analysis (EDA), Inferential Statistics, and Machine Learning Pipeline Optimization to maximize retention campaign profitability.

## 🚀 Executive Value Summary & Realized Impact
* **Financial Scaling:** Boosted net monthly campaign value from a baseline of **$190.00** to **$520.00** (a **173.6% net profit optimization**).
* **Budget Waste Mitigation:** Drastically reduced unnecessary marketing expenditures on False Positives (perfectly happy customers misclassified as flight risks) by **62.2%**, dropping false alarms from 106 down to 40.
* **Risk Isolation:** Engineered a custom probability threshold boundary to filter out high-probability structural churn targets before capital injection.

---

## 🛠️ Cross-Domain Technical Blueprint

### 1. Data Analytics Lens (Data Engineering & Feature Space)
* **Data Cleansing:** Resolved subtle datatype data corruption in the original IBM dataset by using forced numeric coercion to parse hidden white-space entry anomalies in the `TotalCharges` category.
* **Feature Engineering:** Developed a predictive behavioral metric named **Spending Velocity Index**:
  $$\text{Charges\_Per\_Month\_Of\_Loyalty} = \frac{\text{MonthlyCharges}}{\text{tenure} + 1}$$
  This feature penalizes high-bill, low-tenure customer profiles, allowing the model to quickly distinguish between high-revenue stable clients and erratic, brand-new accounts.

### 2. Data Science Lens (Statistical Rigor)
* **Hypothesis Testing:** Established operational baseline variable significance using an inferential **Chi-Square Test of Independence** via `scipy.stats`.
* **Mathematical Proof:** Evaluated the categorical distribution of *Contract Type* against *Churn Status*, mathematically rejecting the Null Hypothesis ($H_0$) with an extreme level of certainty:
  $$p \approx 7.32 \times 10^{-257}$$
  This definitively proved to executive leadership that non-contractual, month-to-month users represent a critical structural risk to retention.

### 3. Machine Learning Engineering Lens (Pipeline Architecture)
* **Ensemble Modeling:** Implemented a robust **Random Forest Classifier** composed of 100 deep decision tree estimators.
* **Imbalance Handling:** Programmatically mitigated a native 73/27 dataset target skew by configuring a cost-sensitive `class_weight='balanced'` vector space.
* **Boundary Tuning:** Extracted raw network floating-point probability distributions via `.predict_proba()` and overrode the standard default 50% decision split. Instantiated a strict **65% business confidence threshold** to fiercely insulate the marketing budget from low-probability predictions.

---

## 📈 Metric Comparative Analysis

| Performance Matrix | Baseline Model Architecture | Optimized Business Engine | Net Operational Variance |
| :--- | :---: | :---: | :---: |
| **False Positives (Wasted Budget)** | 106 Accounts | 40 Accounts | **-62.2% Error Minimization** |
| **Total Customers Reclaimed** | 89 Accounts | 56 Accounts | Robust High-Confidence Subset |
| **Gross Revenue Preserved** | $4,450.00 | $2,800.00 | High-Margin Retention Values |
| **Total Retention Cost** | $4,260.00 | $2,280.00 | **-46.4% Operational Savings** |
| **Net Monthly Value Generated** | **$190.00** | **$520.00** | **+173.6% Financial Yield** |

---

## 💻 Repository Structure & Core Implementation

The pipeline is written natively in Python 3 inside an asynchronous Jupyter execution space (`.ipynb`). The optimized runtime loop handles data extraction, feature processing, modeling, and automated financial tracking in a singular execution thread:

```python
# Strategic execution excerpt showing automated ROI calculations
def calculate_model_roi(y_true, y_pred, monthly_revenue=50, discount_cost=15, success_rate=0.5):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    revenue_saved = tp * success_rate * monthly_revenue
    discount_cost_total = (tp + fp) * discount_cost
    net_roi = revenue_saved - discount_cost_total
    
    print(f"False Positives: {fp}")
    print(f"Net Monthly Model Value: ${net_roi:.2f}")
