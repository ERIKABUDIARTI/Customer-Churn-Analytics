#prediction.py
import joblib

model = joblib.load("models/gradient_boosting_model.joblib")
result_target = joblib.load("models/encoder_target.joblib")

def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Churn or Not Churn)
    """
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)
    final_result = "Churn" if final_result == 1 else "Not Churn"
    return final_result
