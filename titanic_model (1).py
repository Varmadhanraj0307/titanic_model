
# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report, accuracy_score

# In[2]:
df = pd.read_csv("tested.csv")

# In[3]:
df.head()

# In[4]:
df.tail()

# In[5]:
df.info()

# In[6]:
df.isnull().sum()

# In[7]:
df['Age'] = df['Age'].fillna(df['Age'].mode()[0])

# In[8]:
df.isnull().sum()

# In[9]:
df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0])

# In[10]:
df.isnull().sum()

# In[11]:
df = df.dropna()

# In[12]:
df.isnull().sum()

# In[13]:
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# In[14]:
sns.countplot(x='Sex', data=df)
plt.title("Gender Count")
plt.show()

# In[15]:
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Count")
plt.show()

# In[16]:
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# In[17]:
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Class")
plt.show()

# In[18]:
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title("Survival by Embarked")
plt.show()

# In[19]:
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

# In[20]:
sns.histplot(df['Fare'], kde=True)
plt.title("Fare Distribution")
plt.show()

# In[21]:
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Age vs Passenger Class")
plt.show()

# In[22]:
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare vs Passenger Class")
plt.show()

# In[23]:
plt.scatter(df['Age'], df['Fare'])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare")
plt.show()

# In[24]:
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# In[25]:
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']])
plt.show()

# In[26]:
sns.barplot(x='Pclass', y='Fare', data=df)
plt.title("Average Fare by Class")
plt.show()

# In[27]:
df['Survived'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Survival Percentage")
plt.ylabel('')
plt.show()

# In[28]:
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=df, split=True)
plt.title("Violin Plot: Age vs Class")
plt.show()

# In[29]:
df['FamilySize'] = df['SibSp'] + df['Parch']

sns.countplot(x='FamilySize', hue='Survived', data=df)
plt.title("Family Size vs Survival")
plt.show()

# In[30]:
pip install streamlit scikit-learn pandas seaborn matplotlib

# In[31]:
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

# In[32]:
