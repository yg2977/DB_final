import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# Sample data loading and preprocessing (replace with actual dataset)
@st.cache
def load_data():

    combined_data = pd.read_csv('combined_data.csv')
    X = combined_data.drop(columns=['Risk1Yr'])
    y = combined_data['Risk1Yr']
    return X, y

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
#rfc
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt')
rf_clf.fit(X_train, y_train)
#XG Boost
num_classes = len(y.unique())
xgb_model = XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

#UI
st.title('Binary Prediction Dashboard')
st.write("These are prediction using classifier models, please do not use to make any predictions for any health conditions! If you have cancer, please consult with a medical professional!") 
# Input fields for features
st.write("## Input Features for Prediction:")
age_input = st.number_input('How old is the patient')
dgn_input = st.selectbox('What is the diagnosis level of all lung cancer tumors?:', ('1', '2', '3', '4', '5', '6', '7', '8'))
PRE4_input = st.number_input('What is the patients Forced vital capacity?:')
PRE5_input = st.number_input('What is the patients volume that has been exhaled at the end of the first second of forced expiration?:')
activity_level = st.selectbox('What is the patients independent activity level? (0 being fully independent, 2 being bedridden)',
                         ('0','1','2'))
og_tumor_size = st.selectbox("Size of original tumor: ",
                         ('11','12','13','14'))
MI = st.selectbox("Has the paitent expereince a heart attack(s) within the past 6 months?: ",
                         ('Yes','No'))
PAD = st.selectbox("Is the patient diagnosed with peripheral arterial diseases?: ",
                         ('Yes','No'))
Haemoptysis = st.selectbox("Does the patient cough up blood?: ",
                         ('Yes','No'))
Smoking = st.selectbox("Does the patient have or had regular smoking habits?: ",
                         ('Yes','No'))
Asthma = st.selectbox("Is the patient diagnosed with Asthma?: ",
                         ('Yes','No'))
Pain_Before_Surgery = st.selectbox('Does the patient experience physical pain anywhere in the body?', ('Yes','No'))
Coughing = st.selectbox('Does the patient regularly cough unvoluntarily?', ('Yes','No'))
Shortness_Breath = st.selectbox('Does the patient experience regularly experience shortness of breath?', ('Yes','No'))
Physical_Weakness = st.selectbox('Does the patient feel phycially weak most of the time??', ('Yes','No'))
Type_2_Diabetes = st.selectbox("Is the patient diagnosed with Type 2 Diabetes?: ",
                         ('Yes','No'))

# Convert categorical inputs to appropriate format
inputs = {
    'DGN': int(dgn_input),
    'PRE4': PRE4_input,
    'PRE5': PRE5_input,
    'Activity_Scale': int(activity_level),
    'Pain_Before_Surgery': 1 if Pain_Before_Surgery == 'Yes' else 0,
    'Coughint_Blood': 1 if Haemoptysis == 'Yes' else 0,
    'Shortness_Breath': 1 if Shortness_Breath == 'Yes' else 0,
    'Cough': 1 if Coughing == 'Yes' else 0,
    'Phy_Weakness': 1 if Physical_Weakness == 'Yes' else 0,
    'OG_tumor_size': int(og_tumor_size),
    'Type2Diabetes': 1 if Type_2_Diabetes == 'Yes' else 0,
    'Heart_attack_6months': 1 if MI == 'Yes' else 0,
    'PAD': 1 if PAD == 'Yes' else 0,
    'Smoking': 1 if Smoking == 'Yes' else 0,
    'Asthma': 1 if Asthma == 'Yes' else 0,
    'AGE': age_input
}
input_df = pd.DataFrame([inputs])

rf_pred = rf_clf.predict(input_df)[0]
xgb_pred = xgb_model.predict(input_df)[0]

# Predict with both models
if st.button('Run Prediction'):
    # Prediction with RandomForest
    rf_pred = rf_clf.predict(input_df)[0]

    # Prediction with XGBoost
    xgb_pred = xgb_model.predict(input_df)[0]

    # Determine final prediction
    final_pred = 'Does Not Live More than 1 Year Post-Surgery' if (rf_pred + xgb_pred) >= 1 else 'Lives at least 1 Year Post Surgery'

    # Display the result
    st.write("\n### Prediction:")
    st.write(f"Patient: {final_pred}")

def interpret_prediction(pred):
    return "Will not survive past 1 year post surgery" if pred == 1 else "will live longer than 1 year post surgery"

st.write(f"Random Forest Classifier Prediction: {interpret_prediction(rf_pred)}")
st.write(f"XG Boost Classifier Prediction: {interpret_prediction(xgb_pred)}")

st.write("These are prediction using classifier models, please do not use to make any predictions for any health conditions! If you have cancer, please consult with a medical professional!")
