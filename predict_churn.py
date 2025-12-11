import joblib
import pandas as pd

# Load the saved model
model = joblib.load("churn_model.joblib")
print("Model loaded. Now you can enter customer details.")

def get_int(prompt):
    return int(input(prompt))

def get_float(prompt):
    return float(input(prompt))

def get_str(prompt):
    return input(prompt).strip()

while True:
    print("\nEnter customer details")

    # Simple quit option
    quit_choice = input("Press Enter to continue, or type 'q' to quit: ").strip().lower()
    if quit_choice == "q":
        break

    # Collect inputs from user
    CreditScore = get_int("CreditScore (e.g., 600): ")
    Geography    = get_str("Geography (France/Germany/Spain): ")
    Gender       = get_str("Gender (Male/Female): ")
    Age          = get_int("Age (e.g., 40): ")
    Tenure       = get_int("Tenure (years with bank, e.g., 5): ")
    Balance      = get_float("Balance (e.g., 50000.0): ")
    NumOfProducts = get_int("NumOfProducts (e.g., 1 or 2): ")
    HasCrCard   = get_int("HasCrCard (1 = yes, 0 = no): ")
    IsActiveMember    = get_int("IsActiveMember (1 = yes, 0 = no): ")
    EstimatedSalary   = get_float("EstimatedSalary (e.g., 60000.0): ")

    # Build a one-row DataFrame with the same column names as training
    new_data = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary
    }])

    # Predict using the loaded pipeline
    pred_class = model.predict(new_data)[0]                # 0 or 1
    pred_proba = model.predict_proba(new_data)[0, 1]       # probability of churn (class 1)

    if pred_class == 1:
        print(f"\nPrediction: This customer is LIKELY to churn.")
    else:
        print(f"\nPrediction: This customer is NOT likely to churn.")

    print(f"Estimated churn probability: {pred_proba:.3f}")
