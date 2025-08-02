import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import streamlit as st


def preprocess_parkinsons(df):
    df = df.dropna()
    df.to_csv("parkinsons_cleaned.csv", index=False)
    X = df.drop(["name", "status"], axis=1)
    y = df["status"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist()

def preprocess_kidney(df):
    df.replace("?", np.nan, inplace=True)
    df.drop("id", axis=1, inplace=True)
    df = df.dropna()
    df.to_csv("kidney_cleaned.csv", index=False)
    le_dict = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
    X = df.drop("classification", axis=1)
    y = df["classification"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist(), le_dict

def preprocess_liver(df):
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df = df.dropna()
    df.to_csv("liver_cleaned.csv", index=False)
    X = df.drop("Dataset", axis=1)
    y = df["Dataset"].replace({2: 0})
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist()


parkinsons = pd.read_csv(r"D:\Disease Prediction\env\Scripts\parkinsons - parkinsons.csv")
kidney = pd.read_csv(r"D:\Disease Prediction\env\Scripts\kidney_disease - kidney_disease.csv")
liver = pd.read_csv(r"D:\Disease Prediction\env\Scripts\indian_liver_patient - indian_liver_patient.csv")


parkinsons_model, parkinsons_scaler, parkinsons_features = preprocess_parkinsons(parkinsons)
kidney_model, kidney_scaler, kidney_features, kidney_le_dict = preprocess_kidney(kidney)
liver_model, liver_scaler, liver_features = preprocess_liver(liver)


st.set_page_config(page_title="Multiple Disease Prediction System", layout="wide")
st.sidebar.title("Multiple Disease Prediction System")
disease = st.sidebar.radio("Select Prediction Type", ["Parkinsons Prediction", "Kidney Prediction", "Liver Prediction"])
st.markdown("<h1 style='text-align: center; color: black;'>Multiple Disease Prediction using ML</h1>", unsafe_allow_html=True)


if disease == "Parkinsons Prediction":
    st.header("Enter the following features:")

    parkinsons_defaults = {
        "MDVP:Fo(Hz)": 120.0,
        "MDVP:Fhi(Hz)": 140.0,
        "MDVP:Flo(Hz)": 100.0,
        "MDVP:Jitter(%)": 0.005,
        "MDVP:Jitter(Abs)": 0.00005,
        "MDVP:RAP": 0.0025,
        "MDVP:PPQ": 0.0028,
        "Jitter:DDP": 0.0075,
        "MDVP:Shimmer": 0.02,
        "MDVP:Shimmer(dB)": 0.3,
        "Shimmer:APQ3": 0.01,
        "Shimmer:APQ5": 0.013,
        "MDVP:APQ": 0.014,
        "Shimmer:DDA": 0.03,
        "NHR": 0.02,
        "HNR": 20.5,
        "RPDE": 0.55,
        "DFA": 0.72,
        "spread1": -5.5,
        "spread2": 0.2,
        "D2": 2.3,
        "PPE": 0.7
    }

    input_data = []
    for col in parkinsons_features:
        value = st.number_input(f"{col}:", value=parkinsons_defaults.get(col, 0.0), format="%f")
        input_data.append(value)

    if st.button("Predict Parkinson's"):
        scaled_input = parkinsons_scaler.transform([input_data])
        pred = parkinsons_model.predict(scaled_input)
        st.success("✅ Parkinson's Detected" if pred[0] == 1 else "✅ Healthy")


elif disease == "Kidney Prediction":
    st.header("Enter the following features:")

    categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    categorical_options = {
        'rbc': ['normal', 'abnormal'],
        'pc': ['normal', 'abnormal'],
        'pcc': ['present', 'notpresent'],
        'ba': ['present', 'notpresent'],
        'htn': ['yes', 'no'],
        'dm': ['yes', 'no'],
        'cad': ['yes', 'no'],
        'appet': ['good', 'poor'],
        'pe': ['yes', 'no'],
        'ane': ['yes', 'no']
    }

    input_data = []
    for col in kidney_features:
        if col in categorical_cols:
            value = st.selectbox(f"{col}:", categorical_options[col])
            encoded_value = kidney_le_dict[col].transform([value])[0]
            input_data.append(encoded_value)
        else:
            value = st.number_input(f"{col}:", value=0.0)
            input_data.append(value)

    if st.button("Predict Kidney Disease"):
        scaled_input = kidney_scaler.transform([input_data])
        pred = kidney_model.predict(scaled_input)
        st.success("✅ CKD Detected" if pred[0] == 1 else "✅ No CKD")


elif disease == "Liver Prediction":
    st.header("Enter the following features:")

    liver_defaults = {
        "Age": 45.0,
        "Total_Bilirubin": 1.0,
        "Direct_Bilirubin": 0.3,
        "Alkaline_Phosphotase": 200.0,
        "Alamine_Aminotransferase": 30.0,
        "Aspartate_Aminotransferase": 40.0,
        "Total_Protiens": 6.5,
        "Albumin": 3.5,
        "Albumin_and_Globulin_Ratio": 1.0
    }

    gender = st.selectbox("Gender", ["Male", "Female"])

    other_inputs = []
    for col in liver_features:
        if col == "Gender":
            continue
        value = st.number_input(
            f"{col}:", 
            value=liver_defaults.get(col, 0.0),
            min_value=0.0,
            format="%f"
        )
        other_inputs.append(value)

    input_data = [1 if gender == "Male" else 0] + other_inputs

    if st.button("Predict Liver Disease"):
        scaled_input = liver_scaler.transform([input_data])
        pred = liver_model.predict(scaled_input)
        st.success("✅ Liver Disease Detected" if pred[0] == 1 else "✅ Healthy")
