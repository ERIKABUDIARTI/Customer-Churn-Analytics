#data_preprocessing.py
import joblib
import numpy as np
import pandas as pd

encoder_area_code = joblib.load("models/encoder_area_code.joblib")
encoder_international_plan = joblib.load("models/encoder_international_plan.joblib")
encoder_state = joblib.load("models/encoder_state.joblib")
encoder_voice_mail_plan = joblib.load("models/encoder_voice_mail_plan.joblib")
scaler_account_length = joblib.load("models/scaler_account_length.joblib")
scaler_number_customer_service_calls= joblib.load("models/scaler_number_customer_service_calls.joblib")
scaler_number_vmail_messages = joblib.load("models/scaler_number_vmail_messages.joblib")
scaler_total_day_calls = joblib.load("models/scaler_total_day_calls.joblib")
scaler_total_day_charge = joblib.load("models/scaler_total_day_charge.joblib")
scaler_total_day_minutes = joblib.load("models/scaler_total_day_minutes.joblib")
scaler_total_eve_calls = joblib.load("models/scaler_total_eve_calls.joblib")
scaler_total_eve_charge = joblib.load("models/scaler_total_eve_charge.joblib")
scaler_total_eve_minutes = joblib.load("models/scaler_total_eve_minutes.joblib")
scaler_total_night_calls = joblib.load("models/scaler_total_night_calls.joblib")
scaler_total_night_charge = joblib.load("models/scaler_total_night_charge.joblib")
scaler_total_night_minutes = joblib.load("models/scaler_total_night_minutes.joblib")
scaler_total_intl_calls = joblib.load("models/scaler_total_intl_calls.joblib")
scaler_total_intl_charge = joblib.load("models/scaler_total_intl_charge.joblib")
scaler_total_intl_minutes = joblib.load("models/scaler_total_intl_minutes.joblib")


def preprocessing(data):
    """Preprocessing data
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
        
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    df = data.copy()  # Avoid modifying the original data

    # Perform transformations
    df["area_code"] = data["area_code"]
    df["international_plan"] = data["international_plan"]
    df["state"] = data["state"]
    df["voice_mail_plan"] = data["voice_mail_plan"]
    df["account_length"] = scaler_account_length.transform(np.asarray(data["account_length"]).reshape(-1, 1))
    df["number_customer_service_calls"] = scaler_number_customer_service_calls.transform(np.asarray(data["number_customer_service_calls"]).reshape(-1, 1))
    df["number_vmail_messages"] = scaler_number_vmail_messages.transform(np.asarray(data["number_vmail_messages"]).reshape(-1, 1))
    df["total_day_calls"] = scaler_total_day_calls.transform(np.asarray(data["total_day_calls"]).reshape(-1, 1))
    df["total_day_charge"] = scaler_total_day_charge.transform(np.asarray(data["total_day_charge"]).reshape(-1, 1))
    df["total_day_minutes"] = scaler_total_day_minutes.transform(np.asarray(data["total_day_minutes"]).reshape(-1, 1))
    df["total_eve_calls"] = scaler_total_eve_calls.transform(np.asarray(data["total_eve_calls"]).reshape(-1, 1))
    df["total_eve_charge"] = scaler_total_eve_charge.transform(np.asarray(data["total_eve_charge"]).reshape(-1, 1))
    df["total_eve_minutes"] = scaler_total_eve_minutes.transform(np.asarray(data["total_eve_minutes"]).reshape(-1, 1))
    df["total_night_calls"] = scaler_total_night_calls.transform(np.asarray(data["total_night_calls"]).reshape(-1, 1))
    df["total_night_charge"] = scaler_total_night_charge.transform(np.asarray(data["total_night_charge"]).reshape(-1, 1))
    df["total_night_minutes"] = scaler_total_night_minutes.transform(np.asarray(data["total_night_minutes"]).reshape(-1, 1))
    df["total_intl_calls"] = scaler_total_intl_calls.transform(np.asarray(data["total_intl_calls"]).reshape(-1, 1))
    df["total_intl_charge"] = scaler_total_intl_charge.transform(np.asarray(data["total_intl_charge"]).reshape(-1, 1))
    df["total_intl_minutes"] = scaler_total_intl_minutes.transform(np.asarray(data["total_intl_minutes"]).reshape(-1, 1))
    
    return df