
# In[1]:
# Step 1: Import libraries
import pandas as pd
import numpy as np
import pickle

# In[2]:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# In[3]:
# Step 2: Load dataset
df = pd.read_csv(r'C:\Users\Pradip\Downloads\archive (1)\tested.csv')

# In[4]:
# Step 3: Data preprocessing
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# In[5]:
# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)

# In[6]:
# Convert categorical to numeric
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# In[7]:
# Step 4: Split data
X = df.drop('Survived', axis=1)
y = df['Survived']

# In[8]:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# In[9]:
# Step 5: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# In[10]:
# Step 6: Save model
pickle.dump(model, open('model.pkl', 'wb'))


# In[11]:
# Step 6: Save model
pickle.dump(model, open('model.pkl', 'wb'))

# In[12]:
print("Model trained and saved successfully!")

# In[13]:
import streamlit as st
import pickle
import numpy as np


# In[14]:
# Load model
model = pickle.load(open('model.pkl', 'rb'))

# In[15]:
st.title("🚢 Titanic Survival Prediction App")


# In[16]:
st.write("Enter passenger details:")


# In[17]:
# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80)
fare = st.number_input("Fare", value=50.0)

# In[18]:
sex = 1 if sex == "Male" else 0


# In[19]:
# Prediction
if st.button("Predict"):
    input_data = np.array([[pclass, sex, age, fare]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")

# In[20]:
