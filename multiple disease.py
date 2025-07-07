import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import streamlit as st

# --------------------- Parkinson's Model ---------------------
def preprocess_parkinsons(df):
    df = df.dropna()  # Drop rows with null values
    df.to_csv("parkinsons_cleaned.csv", index=False)  # Save cleaned data
    X = df.drop(["name", "status"], axis=1)
    y = df["status"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist()

# --------------------- Kidney Model ---------------------
def preprocess_kidney(df):
    df.replace("?", np.nan, inplace=True)
    df.drop("id", axis=1, inplace=True)
    df = df.dropna()  # Drop rows with null values
    df.to_csv("kidney_cleaned.csv", index=False)  # Save cleaned data
    le = LabelEncoder()
    df = df.apply(lambda col: le.fit_transform(col) if col.dtypes == 'object' else col)
    X = df.drop("classification", axis=1)
    y = df["classification"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist()

# --------------------- Liver Model ---------------------
def preprocess_liver(df):
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df = df.dropna()  # Drop rows with null values
    df.to_csv("liver_cleaned.csv", index=False)  # Save cleaned data
    X = df.drop("Dataset", axis=1)
    y = df["Dataset"].replace({2: 0})
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist()

# --------------------- Load Data ---------------------
parkinsons = pd.read_csv("D:\Disease Prediction\env\Scripts\parkinsons - parkinsons.csv")
kidney = pd.read_csv("D:\Disease Prediction\env\Scripts\kidney_disease - kidney_disease.csv")
liver = pd.read_csv("D:\Disease Prediction\env\Scripts\indian_liver_patient - indian_liver_patient.csv")

# --------------------- Train Models ---------------------
parkinsons_model, parkinsons_scaler, parkinsons_features = preprocess_parkinsons(parkinsons)
kidney_model, kidney_scaler, kidney_features = preprocess_kidney(kidney)
liver_model, liver_scaler, liver_features = preprocess_liver(liver)

# --------------------- Streamlit App ---------------------
st.set_page_config(page_title="Multiple Disease Prediction System", layout="wide")
st.sidebar.title("Multiple Disease Prediction System")
disease = st.sidebar.radio("Select Prediction Type", ["Parkinsons Prediction", "Kidney Prediction", "Liver Prediction"])
st.markdown("<h1 style='text-align: center; color: black;'>Multiple Disease Prediction using ML</h1>", unsafe_allow_html=True)

# --------------------- Parkinson's UI ---------------------
if disease == "Parkinsons Prediction":
    st.header("Enter the following features:")
    input_data = [st.number_input(col, value=0.0, format="%f") for col in parkinsons_features]
    if st.button("Predict Parkinson's"):
        scaled_input = parkinsons_scaler.transform([input_data])
        pred = parkinsons_model.predict(scaled_input)
        st.success("✅ Parkinson's Detected" if pred[0] == 1 else "✅ Healthy")

# --------------------- Kidney UI ---------------------
elif disease == "Kidney Prediction":
    st.header("Enter the following features:")
    input_data = []
    for col in kidney_features:
        if kidney[col].dtype == 'object':
            input_data.append(st.text_input(col, value=""))
        else:
            input_data.append(st.number_input(col, value=0.0))
    if st.button("Predict Kidney Disease"):
        temp_df = pd.DataFrame([input_data], columns=kidney_features)
        temp_df = temp_df.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtypes == 'object' else col)
        scaled_input = kidney_scaler.transform(temp_df)
        pred = kidney_model.predict(scaled_input)
        st.success("✅ CKD Detected" if pred[0] == 1 else "✅ No CKD")

# --------------------- Liver UI ---------------------
elif disease == "Liver Prediction":
    st.header("Enter the following features:")
    gender = st.selectbox("Gender", ["Male", "Female"])
    other_inputs = [st.number_input(col, value=0.0) for col in liver_features if col != "Gender"]
    input_data = [1 if gender == "Male" else 0] + other_inputs
    if st.button("Predict Liver Disease"):
        scaled_input = liver_scaler.transform([input_data])
        pred = liver_model.predict(scaled_input)
        st.success("✅ Liver Disease Detected" if pred[0] == 1 else "✅ Healthy")
