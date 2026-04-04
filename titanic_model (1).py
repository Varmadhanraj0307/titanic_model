import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Ensure 'tested.csv' is in your GitHub repo!
    data = pd.read_csv("tested.csv")
    
    # Preprocessing
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Encoding for the model
    # We create a copy for training so the original remains readable for plots
    train_df = data.copy()
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return data, train_df

try:
    df, df_encoded = load_data()
except FileNotFoundError:
    st.error("Error: 'tested.csv' not found. Please upload it to your GitHub repository.")
    st.stop()

# --- MODEL TRAINING ---
@st.cache_resource
def train_models(data):
    X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return dt, rf

dt_model, rf_model = train_models(df_encoded)

# --- SIDEBAR INPUTS ---
st.sidebar.header("🚢 Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex_raw = st.sidebar.selectbox("Gender", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 25)
fare = st.sidebar.slider("Fare Price", 0, 500, 50)
embarked_raw = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Map inputs for prediction
sex = 0 if sex_raw == "male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked_raw]
input_data = np.array([[pclass, sex, age, fare, embarked]])

# --- MAIN UI ---
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Predict if a passenger would survive the Titanic disaster based on their details.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Make a Prediction")
    model_choice = st.selectbox("Choose AI Model", ["Decision Tree", "Random Forest"])
    
    if st.button("Predict Survival Status"):
        prediction = dt_model.predict(input_data) if model_choice == "Decision Tree" else rf_model.predict(input_data)
        
        if prediction[0] == 1:
            st.success("🎉 Results: The passenger likely Survived!")
        else:
            st.error("💀 Results: The passenger likely did not survive.")

with col2:
    st.subheader("📊 Data Visualizations")
    plot_option = st.selectbox("Select Insight", ["Survival Count", "Gender Distribution", "Class Distribution"])
    
    fig, ax = plt.subplots()
    if plot_option == "Survival Count":
        sns.countplot(x='Survived', data=df, ax=ax, palette="viridis")
    elif plot_option == "Gender Distribution":
        sns.countplot(x='Sex', data=df, ax=ax, palette="magma")
    elif plot_option == "Class Distribution":
        sns.countplot(x='Pclass', data=df, ax=ax, palette="crest")
    
    st.pyplot(fig)

st.divider()
st.write("Current Dataset Preview (First 5 rows):")
st.dataframe(df.head())
