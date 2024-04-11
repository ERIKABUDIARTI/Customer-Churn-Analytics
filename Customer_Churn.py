#Customer Churn.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from PIL import Image
import time
from preprocessing import preprocessing
from preprocessing import encoder_area_code, encoder_international_plan, encoder_state, encoder_voice_mail_plan
from preprocessing import scaler_account_length, scaler_number_customer_service_calls, scaler_number_vmail_messages, scaler_total_day_calls, scaler_total_day_charge, scaler_total_day_minutes, scaler_total_eve_calls, scaler_total_eve_charge, scaler_total_eve_minutes, scaler_total_night_calls, scaler_total_night_charge, scaler_total_night_minutes, scaler_total_intl_calls, scaler_total_intl_charge, scaler_total_intl_minutes
from prediction import prediction


#Setting page
st.set_page_config(page_title="Customer Churn", 
                   layout="wide")

#Introduction
#List of image file names
image_files = ['logo_JJT.png']

#Desired image size in pixels
desired_width = 160
desired_height = 160

col1, col2 = st.columns([2, 10])

with col1:
    for idx, image_file in enumerate(image_files):
        img = Image.open(image_file)
        resized_img = img.resize((desired_width, desired_height))
        st.image(resized_img)
with col2:
    st.header(':sparkles: JAYA JAYA TELECOM :sparkles:')
    st.subheader(":runner Prediction of Customer Churn :runner")

st.sidebar.write("""
    The Jaya Jaya Telecom is a telecommunications company that was established in 2000. 
    Until now, it has built a substantial customer base and maintains an excellent reputation. 
    However, there are also many customers who do not renew their subscriptions, commonly referred to as churn. 
    This high churn rate is undoubtedly a significant issue for the company. 
    Therefore, Jaya Jaya Telecom aims to detect customers who may be at risk of churn as early as possible so that they can be provided with special benefits.
""")

add_selectitem = st.sidebar.selectbox("Want to open about?", ("Customer Churn",))

st.sidebar.write(
    "Data obtained from [Telecom Churn Dataset](https://www.kaggle.com/datasets/arashnic/telecom-churn-dataset)"
    )

# Initialize an empty dictionary to store user input
data = {}

# Convert user input dictionary to DataFrame
user_input_df = pd.DataFrame(data, index=[0])

col1, col2, col3, col4 = st.columns(4)
with col1:
    encoder_area_code = LabelEncoder()
    encoder_area_code.fit(['408', '415', '510'])
    area_code = st.selectbox(label='Area Code', options=['408', '415', '510'], index=0)
    data['area_code'] = [encoder_area_code.transform([area_code])[0]]
with col2:
    encoder_state = LabelEncoder()
    encoder_state.fit(['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])
    state= st.selectbox(label='State', options=['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'], index=0)
    data["state"] = [encoder_state.transform([state])[0]]
with col3:
    encoder_international_plan = LabelEncoder()
    encoder_international_plan.fit(['With International Plan', 'Without International Plan'])
    international_plan= st.selectbox(label='International Plan', options=['With International Plan', 'Without International Plan'], index=0)
    data["international_plan"] = [encoder_international_plan.transform([international_plan])[0]]
with col4:
    encoder_voice_mail_plan = LabelEncoder()
    encoder_voice_mail_plan.fit(['With Voice Mail Plan', 'Without Voice Mail Plan'])
    voice_mail_plan= st.selectbox(label='Voice Mail Plan', options=['With Voice Mail Plan', 'Without Voice Mail Plan'], index=0)
    data["voice_mail_plan"] = [encoder_voice_mail_plan.transform([voice_mail_plan])[0]]

col5, col6, col7 = st.columns(3)
with col5:
    account_length = st.number_input(label='Account Length', value=100)
    data['account_length'] = [account_length]
with col6:
    number_customer_service_calls = st.number_input(label='Number of CS Calls', value=5)
    data['number_customer_service_calls'] = [number_customer_service_calls]
with col7:
    number_vmail_messages = st.number_input(label='Number of Vmail Msgs', value=26)
    data['number_vmail_messages'] = [number_customer_service_calls]

col8, col9, col10 = st.columns(3)
with col8:
    total_day_calls = st.number_input(label='Total Day Calls', value=100)
    data['total_day_calls'] = [total_day_calls]
with col9:
    total_day_charge = st.number_input(label='Total Day Charge', value=30)
    data['total_day_charge'] = [total_day_charge]
with col10:
    total_day_minutes = st.number_input(label='Total Day Minutes', value=180)
    data['total_day_minutes'] = [total_day_minutes]
    
col11, col12, col13 = st.columns(3)
with col11:
    total_eve_calls = st.number_input(label='Total Eve Calls', value=100)
    data['total_eve_calls'] = [total_eve_calls]
with col12:
    total_eve_charge = st.number_input(label='Total Eve Charge', value=17)
    data['total_eve_charge'] = [total_eve_charge]
with col13:
    total_eve_minutes = st.number_input(label='Total Eve Minutes', value=200)
    data['total_eve_minutes'] = [total_eve_minutes]

col14, col15, col16 = st.columns(3)
with col14:
    total_night_calls = st.number_input(label='Total Night Calls', value=100)
    data['total_night_calls'] = [total_night_calls]
with col15:
    total_night_charge = st.number_input(label='Total Night Charge', value=9)
    data['total_night_charge'] = [total_night_charge]
with col16:
    total_night_minutes = st.number_input(label='Total Night Minutes', value=200)
    data['total_night_minutes'] = [total_night_minutes]

col17, col18, col19 = st.columns(3)
with col17:
    total_intl_calls = st.number_input(label='Total International Calls', value=4)
    data['total_intl_calls'] = [total_intl_calls]
with col18:
    total_intl_charge = st.number_input(label='Total International Charge', value=3)
    data['total_intl_charge'] = [total_intl_charge]
with col19:
    total_intl_minutes = st.number_input(label='Total International Minutes', value=10)
    data['total_intl_minutes'] = [total_intl_minutes]

# Convert user input dictionary to DataFrame
user_input_df = pd.DataFrame(data, index=[0])

# Display user input
with st.expander("Raw Dataset"):
        st.dataframe(data=user_input_df, width=1200, height=20)
# Preprocess data and make prediction on button click
if st.button('Click Here to Predict'):
    new_data = preprocessing(data=user_input_df)
    with st.spinner('Predicting...'):
        time.sleep(2)  # Simulating prediction process
        output = prediction(new_data)
        st.success(f"Prediction: {output}")

st.snow()