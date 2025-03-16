import streamlit as st
import joblib
import pandas as pd
import gdown
import os

try:
    if not os.path.exists('random_forest_model.pkl'):
        gdown.download("https://drive.google.com/uc?id=1tfyV__6kbfgYjhtelc9dM3yJqAlgKD_d", 'random_forest_model.pkl', quiet=False)
    
    svm_model = joblib.load('svm_model.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder_dict = joblib.load('encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    valid_options = joblib.load('valid_options.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Income Prediction Comparison")

def decode_labels(col, encoded_values):
    encoder = encoder_dict.get(col)
    return [encoder.inverse_transform([val])[0] for val in encoded_values] if encoder else encoded_values

def collect_input(model_name=""):
    age = st.number_input(f'Age ({model_name})', min_value=18, max_value=100, value=30, key=f'age_{model_name}')
    workclass = st.selectbox(f'Workclass ({model_name})', options=decode_labels('workclass', valid_options['workclass']), key=f'workclass_{model_name}')
    education = st.selectbox(f'Education ({model_name})', options=decode_labels('education', valid_options['education']), key=f'education_{model_name}')
    marital_status = st.selectbox(f'Marital Status ({model_name})', options=decode_labels('marital.status', valid_options['marital.status']), key=f'marital_status_{model_name}')
    occupation = st.selectbox(f'Occupation ({model_name})', options=decode_labels('occupation', valid_options['occupation']), key=f'occupation_{model_name}')
    relationship = st.selectbox(f'Relationship ({model_name})', options=decode_labels('relationship', valid_options['relationship']), key=f'relationship_{model_name}')
    race = st.selectbox(f'Race ({model_name})', options=decode_labels('race', valid_options['race']), key=f'race_{model_name}')
    native_country = st.selectbox(f'Native Country ({model_name})', options=decode_labels('native.country', valid_options['native.country']), key=f'native_country_{model_name}')

    input_data = pd.DataFrame({
        'age': [age], 'workclass': [workclass], 'education': [education], 'marital.status': [marital_status],
        'occupation': [occupation], 'relationship': [relationship], 'race': [race], 'native.country': [native_country]
    })
    return input_data

def preprocess_input(input_df):
    categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country']
    for col in categorical_columns:
        if col in encoder_dict:
            try:
                input_df[col] = encoder_dict[col].transform([input_df[col].iloc[0]])[0]
            except ValueError:
                st.error(f"Invalid value for {col}: {input_df[col].iloc[0]}")
                return None
    input_df[['age']] = scaler.transform(input_df[['age']])
    return input_df

def make_predictions(input_data):
    processed_input = preprocess_input(input_data)
    if processed_input is not None:
        svm_prediction = svm_model.predict(processed_input.values)
        svm_result = target_encoder.inverse_transform(svm_prediction)
        svm_result_text = "Income > 50K" if svm_result == 1 else "Income <= 50K"

        rf_prediction = rf_model.predict(processed_input.values)
        rf_result = target_encoder.inverse_transform(rf_prediction)
        rf_result_text = "Income > 50K" if rf_result == 1 else "Income <= 50K"

        return svm_result_text, rf_result_text

st.header("SVM Model Input")
svm_input_data = collect_input("SVM")
if st.button('Predict with SVM'):
    svm_result_text, _ = make_predictions(svm_input_data)
    st.success(f"SVM Prediction: {svm_result_text}")

st.header("Random Forest Model Input")
rf_input_data = collect_input("Random Forest")
if st.button('Predict with Random Forest'):
    _, rf_result_text = make_predictions(rf_input_data)
    st.success(f"Random Forest Prediction: {rf_result_text}")
