import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🚢 Titanic Survival Prediction App")

# Load data
df = pd.read_csv("tested.csv")

# Data preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features & target
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 25)
fare = st.sidebar.slider("Fare", 0, 500, 50)
embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

# Convert input
sex = 0 if sex == "male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

input_data = np.array([[pclass, sex, age, fare, embarked]])

# Model selection
model_choice = st.selectbox("Choose Model", ["Decision Tree", "Random Forest"])

# Prediction
if st.button("Predict"):
    if model_choice == "Decision Tree":
        prediction = dt_model.predict(input_data)
    else:
        prediction = rf_model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("💀 Passenger Did Not Survive")

# Visualization Section
st.subheader("📊 Data Visualization")

plot_option = st.selectbox("Select Plot", ["Survival Count", "Gender Count", "Class Count"])

if plot_option == "Survival Count":
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=df, ax=ax)
    st.pyplot(fig)

elif plot_option == "Gender Count":
    fig, ax = plt.subplots()
    sns.countplot(x='Sex', data=df, ax=ax)
    st.pyplot(fig)

elif plot_option == "Class Count":
    fig, ax = plt.subplots()
    sns.countplot(x='Pclass', data=df, ax=ax)
    st.pyplot(fig)
