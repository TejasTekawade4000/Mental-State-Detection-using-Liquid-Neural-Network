import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

# The code snippet `from keras.utils import to_categorical` is importing the `to_categorical` function
# from the Keras library. This function is commonly used to convert class vectors (integers) to binary
# class matrix representation (one-hot encoding) for neural network models.
# from keras.utils import to_categorical 
# # from keras.models import load_model
# from util import load_model
# def run():
#     st.title('EEG based Emotion Detection using Liquid Neural Networks') 
#     uploaded_file = st.file_uploader("Upload CSV for Testing", type="csv")
#     eeg_data = pd.read_csv(uploaded_file)

#     # Extract FFT data columns
#     fft_data = eeg_data.iloc[:, 6:-1]

#     # Streamlit app
#     st.title('EEG Data FFT Visualization')

#     progress_bar = st.sidebar.progress(0)
#     status_text = st.sidebar.empty()
#     last_rows = np.zeros((1, fft_data.shape[1]))
#     chart = st.line_chart(last_rows)

#     # Display the FFT data in a real-time updating manner
#     for i in range(1, 101):
#         new_data_index = int(i * len(fft_data) / 100)
#         new_rows = fft_data.iloc[:new_data_index, :].values
        
#         status_text.text(f"{i}% Complete")
#         chart.add_rows(new_rows)
#         progress_bar.progress(i)
#         last_rows = new_rows
#         time.sleep(0.05)

#     progress_bar.empty()

#     # Button to rerun the script
#     st.button("Re-run")
# # # This code snippet is performing the following tasks:
# #     input_file = r'emotions.csv'
# #     data = pd.read_csv(input_file)
# #     scaler = StandardScaler()
# #     label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
# #     data['label'] = data['label'].replace(label_mapping)
# #     X = data.drop('label', axis=1)
# #     y = data['label'].copy()
# #     scaler.fit(X)
# #     X = scaler.transform(X) 
# #     y = to_categorical(y)

# #     # Reshaping the independent variables
# #     X_resized = np.reshape(X, (X.shape[0],1,X.shape[1]))


# # Load the CSV file

import streamlit as st
import pandas as pd
import os

def run():
    st.title("LNN Test")

    # Section to view processed data
    st.header("View Processed CSV Data")
    output_folder = st.text_input("Output Folder Path", "results")

    if st.button("Load Data"):
        if os.path.exists(output_folder):
            all_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]
            if all_files:
                selected_file = st.selectbox("Select a file", all_files)
                if selected_file:
                    df = pd.read_csv(os.path.join(output_folder, selected_file))
                    st.write("### Patient Information")
                    patient_info = df[['Name', 'PatientID', 'Age', 'Gender', 'Date', 'Time']].drop_duplicates()
                    st.write(patient_info)
                    
                    st.write("### FFT Data")
                    fft_columns = [col for col in df.columns if col.startswith('fft_')]
                    fft_data = df[fft_columns]
                    st.dataframe(fft_data)
            else:
                st.error("No CSV files found in the specified folder.")
        else:
            st.error("Output folder does not exist.")

    # Add the LNN testing code here
    st.header("LNN Testing Section")
    # Implement the LNN test logic here
    # For example, you might have code to load a model, preprocess the data, and make predictions
    # Note: Ensure you have the relevant code and imports needed for the LNN test

if __name__ == "__main__":
    run()


