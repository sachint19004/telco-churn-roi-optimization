import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate(df):
    """Prepares data, trains a Random Forest, and evaluates its performance."""
    print("\n--- Preparing Data for ML ---")
    # Drop useless ID column & turn text categories into binary numbers (1s and 0s)
    X = pd.get_dummies(df.drop(['customerID', 'Churn'], axis=1), drop_first=True)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the AI Model... Please wait...")
    # Train model using balanced class weights to fix the data skew
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating the Model...")
    predictions = model.predict(X_test)

    # Generate Confusion Matrix Visual
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Did we catch the churners?')
    plt.ylabel('Actual Churn (0=No, 1=Yes)')
    plt.xlabel('Predicted Churn (0=No, 1=Yes)')
    plt.show()

    # Print Final Classification Report
    print("\n--- AI Final Exam Report ---")
    print(classification_report(y_test, predictions))