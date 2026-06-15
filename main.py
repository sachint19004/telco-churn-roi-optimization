# Import the functions we built inside the src folder
from src.data_preprocessing import load_and_clean_data
from src.eda_and_stats import generate_visualizations, run_statistical_test
from src.model_pipeline import train_and_evaluate

if __name__ == "__main__":
    # Point this to wherever you save your CSV locally inside your repository
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Step 1: Clean the data
    clean_df = load_and_clean_data(DATA_PATH)
    
    # Step 2: Analyze the data
    generate_visualizations(clean_df)
    run_statistical_test(clean_df)
    
    # Step 3: Train the model
    train_and_evaluate(clean_df)
    
    print("\n✅ End-to-End Pipeline Execution Complete!")