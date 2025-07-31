import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain',
                'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast',
                'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High',
                 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
             'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
                   'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Label encoding
df_encoded = df.copy()
label_encoders = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Split features and target
X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']

# Train Decision Tree (ID3)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Streamlit App Title
st.title("PlayTennis Prediction with ID3 Decision Tree")

# Sidebar inputs
st.sidebar.header("Input Weather Conditions")


def user_input():
    outlook = st.sidebar.selectbox("Outlook", df['Outlook'].unique())
    temp = st.sidebar.selectbox("Temperature", df['Temperature'].unique())
    humidity = st.sidebar.selectbox("Humidity", df['Humidity'].unique())
    wind = st.sidebar.selectbox("Wind", df['Wind'].unique())

    return pd.DataFrame([[outlook, temp, humidity, wind]],
                        columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])


input_df = user_input()

# Encode input for prediction
input_encoded = input_df.copy()
for col in input_encoded.columns:
    input_encoded[col] = label_encoders[col].transform(input_encoded[col])

# Make prediction
prediction = model.predict(input_encoded)[0]
prediction_label = label_encoders['PlayTennis'].inverse_transform([prediction])[0]

# Display prediction
st.subheader("Prediction:")
st.success(f"The model predicts: **{prediction_label}**")

# Show user input
st.subheader("Input Values:")
st.write(input_df)

# Show training data
st.subheader("Training Data:")
st.dataframe(df)

# Show decision tree
st.subheader("Decision Tree Visualization:")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=label_encoders['PlayTennis'].classes_, filled=True)
st.pyplot(fig)
