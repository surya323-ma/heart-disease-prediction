"# heart-disease-prediction" 
# ❤️ Heart Disease Prediction App

A Streamlit web application to predict heart disease risk using a trained Logistic Regression model. The app allows users to input personal and medical information, and provides a prediction, risk level, and probability visualization.

🔗 Live Demo

You can view the app live on Streamlit Cloud:
Open Heart Disease Prediction App
https://heartdisease0.streamlit.app/

(Replace YOUR_USERNAME and YOUR_REPO with your GitHub details.)

🧰 Features

Predicts heart disease risk: Healthy or Disease.

Shows prediction confidence and risk level (Low / Medium / High).

Interactive sliders and selectboxes for input features:

Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, ST Depression, Slope, Major Vessels, Thalassemia.

Probability visualization with Plotly bar chart.

Input summary expandable section.

Sidebar with model info and key risk factors.

🏗️ Installation

Clone the repository:

git clone https://github.com/surya323-ma/heart-disease-prediction.git
cd heart-disease-prediction


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run heart_app.py


Open the URL in your browser: http://localhost:8501

🧠 Model

Algorithm: Logistic Regression

Features: 13 medical attributes

Accuracy: ~80–85% on the sample dataset

Files:

heart_disease_model.pkl – trained model

heart_scaler.pkl – standard scaler

⚠️ Note: This app is for educational purposes only. It is not a substitute for professional medical advice.

📊 Screenshots


Input sliders and selections for personal and medical information.


Prediction result with confidence and risk level.

📂 Project Structure
heart-disease-prediction/
│
├── heart_app.py     # Streamlit app
|___heart_disease_prediction.ipynb    
├── heart_disease_model.pkl # Trained model
├── heart_scaler.pkl       # Scaler for features
├── requirements.txt       # Python dependencies
├── README.md              # Project description
└── screenshots/           # Optional: Screenshots for README

🔧 Future Improvements

Integrate full UCI Heart Disease dataset for better model accuracy.

Add user authentication for personalized tracking.

Deploy permanently on Streamlit Cloud with public access.
