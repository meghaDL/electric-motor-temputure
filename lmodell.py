#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#import pickle
import streamlit as st
#!pip install streamlit
from sklearn.preprocessing import StandardScaler
import joblib
import os
from sklearn.linear_model import LinearRegression


# In[2]:


#pickle_in=open("XG.pkl",'rb')
#XG=pickle.load(pickle_in)

lm = joblib.load(r'C:\Users\Dell123\lm.pkl')
sd = StandardScaler()


# In[3]:


training_data = pd.read_csv(r'C:\Users\Dell123\Desktop\data science project\temperature_data.csv')
sd.fit(training_data.drop(columns=['motor_speed']))

import base64



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
def main():
   
    add_bg_from_local('images1.jpg') 

  # Increase font size and change text color in the title and description
    st.markdown(
        f"""
        <h2 style="font-size: 40px; color: #FFFFFF;">Motor Speed Prediction App</h1>
        <p style="font-size: 18px; color: #FFFFFF;">This app uses an LinearRegression model to predict motor speed based on the given input features.</p>
        """,
        unsafe_allow_html=True,
    )    
#st.title("Motor Speed Prediction App")
  #  st.write("This app uses an XGBoost model to predict motor speed based on the given input features.")


    # User Input Section
    st.sidebar.title("Input Features")
    
    # Create input widgets for each feature (you can add more if needed)
    ambient = st.sidebar.slider("Ambient Temperature", min_value=0.0, max_value=100.0, step=0.1)
    coolant = st.sidebar.slider("Coolant Temperature", min_value=0.0, max_value=100.0, step=0.1)
    u_d = st.sidebar.slider("D_Current Component ", min_value=0.0, max_value=100.0, step=0.1)
    u_q = st.sidebar.slider("Q_Current Component", min_value=0.0, max_value=100.0, step=0.1)
    torque = st.sidebar.slider("Torque", min_value=0.0, max_value=1000.0, step=1.0)
    i_d = st.sidebar.slider("D_Voltage Component ", min_value=0.0, max_value=100.0, step=0.1)
    i_q = st.sidebar.slider("Q_Voltage Component", min_value=0.0, max_value=100.0, step=0.1)
    pm =  st.sidebar.slider("Peramant Magnitude", min_value=0.0, max_value=100.0, step=0.1)
    stator_tooth =  st.sidebar.slider("Statr Tooth", min_value=0.0, max_value=100.0, step=0.1)
    stator_yoke =  st.sidebar.slider("Stator_Yoke", min_value=0.0, max_value=100.0, step=0.1) 
    

    # Prepare the user input data for prediction
    user_input = pd.DataFrame({
        'ambient': [ambient],
        'coolent': [coolant],
        'u_d':[u_d],
        'u_q':[u_q],
        'torque':[torque],
        'i_d':[i_d],
        'i_q':[i_q],
        'pm':[pm],
        'stator_tooth':[stator_tooth],
        'stator_yoke':[stator_yoke]
    })

    # Standardize the user input data using the same scaler used for the training data
    user_input_scaled = sd.fit_transform(user_input)
    
   # Print user input and scaled input for debugging
    st.write("User Input:")
    st.write(user_input)
   


    if st.button('Predict'):
        prediction = lm.predict(user_input)
        #st.write('Prediction:', prediction)
      # Display the prediction
        st.write("Predicted Motor Speed:", prediction[0])

      # Add the file upload button
    uploaded_file = st.file_uploader("Upload a file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded file (assuming it's a CSV) and display its contents
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded File:")
        st.write(df)
       
        df_scaled = sd.transform(df)
        df['Predicted_Speed'] = lm.predict(df_scaled)

        # Save the results to a new CSV file
        results_file = os.path.join('D:\ExcelR Data Science\Project', 'predicted_results.csv')
        df.to_csv(results_file, index=False)
        st.success(f"Predicted results saved to: {results_file}")


if __name__ == "__main__":
    sd = StandardScaler()
    main()


# In[ ]:





# In[ ]:




