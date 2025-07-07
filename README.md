Multiple Disease Prediction Using Machine Learning

Project Overview

The Multiple Disease Prediction System is a machine learning-based web application developed to assist in the early detection of critical health conditions including Parkinson’s disease, Chronic Kidney Disease (CKD), and Liver Disease. With the increasing importance of preventive healthcare, this system offers an efficient way to identify potential risks based on patient data, enabling timely medical intervention and improved decision-making by healthcare professionals.

Objectives

- To create a reliable, scalable, and user-friendly web-based tool that predicts the presence of diseases.
- To minimize diagnostic delays by quickly analyzing test data and providing predictions.
- To support healthcare providers with data-driven decision support.
- To demonstrate how artificial intelligence can be applied meaningfully in healthcare.
  
Key Features

- Multi-disease prediction: Parkinson’s, Kidney, and Liver diseases.
- Clean, interactive interface using Streamlit.
- Real-time predictions using trained ML models.
- Built-in data preprocessing including null handling, encoding, and scaling.
- Cleaned datasets saved as .csv files.
- Easily extensible to other diseases.
  
Datasets Used

1. Parkinson’s Disease Dataset
2. Chronic Kidney Disease Dataset
3. Liver Disease Dataset
   
Each dataset contains medical parameters and a target classification used for supervised learning.

Technical Implementation

Each dataset undergoes preprocessing such as null removal, encoding, and scaling. Models used include Random Forest and XGBoost. The trained models are embedded into a Streamlit frontend that accepts user inputs and displays real-time predictions. Cleaned datasets are saved as: parkinsons_cleaned.csv, kidney_cleaned.csv, and liver_cleaned.csv.

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
  
Tools & Technologies

Programming Language: Python
Frontend: Streamlit
Machine Learning: scikit-learn, XGBoost
Data Handling: pandas, numpy
Preprocessing: LabelEncoder, StandardScaler
Sample Inputs & Outputs

Users enter test values based on selected disease. Models predict:

→ Parkinson’s Detected or Healthy
→ CKD Detected or No CKD
→ Liver Disease Detected or Healthy

Conclusion

This project highlights how machine learning can be practically applied in healthcare for early disease detection. It provides accurate predictions, user-friendly design, and can be scaled to include more conditions and insights.

