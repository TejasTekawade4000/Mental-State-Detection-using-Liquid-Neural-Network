import streamlit as st
import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
# from util import load_model
from tensorflow.keras.layers import LSTM, Dense, Flatten
import keras

def calm_negative_state():
    st.write("Ways to calm down:")
    st.write("- Take deep breaths")
    st.write("- Practice mindfulness or meditation")
    st.write("- Engage in physical activity or exercise")
    st.write("- Talk to a friend or loved one")

def preprocess_data(fft_data):
    scaler = StandardScaler()
    scaler.fit(fft_data)
    fft_data = scaler.transform(fft_data)
    return fft_data

def run():
    st.title("LNN Test and View Processed Data")

    # Section to view processed data
    st.header("View Processed CSV Data")

    # Input for selecting the folder
    folder_path = st.text_input("Select Folder Path", "results")

    # Button to list CSV files in the selected folder
    if st.button("List CSV Files"):
        if os.path.exists(folder_path):
            all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if all_files:
                selected_file = st.selectbox("Select a file", all_files)
                if selected_file:
                    df = pd.read_csv(os.path.join(folder_path, selected_file))
                    st.write("### Patient Information")
                    patient_info = df[['Name', 'PatientID', 'Age', 'Gender', 'Date', 'Time']].drop_duplicates()
                    st.write(patient_info)
                    
                    st.write("### FFT Data")
                    fft_columns = [col for col in df.columns if col.startswith('fft_')]
                    fft_data = df[fft_columns]
                    st.dataframe(fft_data)

                    # Store the FFT data in Streamlit session state
                    st.session_state['fft_data'] = fft_data.to_json()

            else:
                st.error("No CSV files found in the specified folder.")
        else:
            st.error("Selected folder does not exist.")

    # Preprocess the data outside the button block
    if 'fft_data' in st.session_state:
        preprocessed_data = preprocess_data(pd.read_json(st.session_state['fft_data']))
        num_classes = 3
        timesteps = 1
        input_shape = (timesteps, 2548)

        model = load_model('modell.keras')

        if st.button("Analyze"):
            pred = model.predict(preprocessed_data)
            mind_state = np.argmax(pred,axis=1)
            if mind_state == 0:
                calm_negative_state()
            elif mind_state == 1:
                st.write("Neutral state.")
            elif mind_state == 2:
                st.write("Positive state.")
            else:
                st.write("Invalid mind state detected.")

            st.text(mind_state)

    # Button to trigger processing
    if st.button("Process Data"):
        if 'fft_data' in st.session_state:
            st.session_state['fft_data'] = fft_data.to_json()
        else:
            st.session_state['fft_data'] = None

if __name__ == "__main__":
    run()
