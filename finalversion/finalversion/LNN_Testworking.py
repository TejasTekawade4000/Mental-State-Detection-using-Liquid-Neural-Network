import streamlit as st
import pandas as pd
import os
import time
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical 
from utils import load_model



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
    # Perform any required preprocessing of the FFT data
    # This may include scaling, reshaping, or other preprocessing steps
    preprocessed_data = fft_data  # Placeholder, replace with actual preprocessing logic
    return preprocessed_data


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

                    # Progress bar and line chart for visualization
                    st.subheader("Processing Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    chart = st.line_chart(fft_data.iloc[0])

                    # Simulate processing and update the chart
                for i in range(1, 101):
                    # Simulate data generation
                    noise = np.random.randn(len(fft_data))
                    new_rows = fft_data.iloc[-1, :] + noise.cumsum()
                    
                    # Update status text and progress bar
                    status_text.text("%i%% Complete" % i)
                    progress_bar.progress(i)
                    
                    # Update line chart with new data
                    chart.add_rows(new_rows)
                    
                    # Sleep to mimic processing time (optional)
                    time.sleep(0.05)

                    # Clear the progress bar after completion
                    progress_bar.empty()

            else:
                st.error("No CSV files found in the specified folder.")
        else:
            st.error("Selected folder does not exist.")

    # Add the LNN testing code here
    st.header("LNN Testing Section")
    model = load_model()
    model.load_weights("test.h5")
    if st.button("Analyze"):
        pred = model.predict(preprocess_data(fft_data))
        mind_state = np.argmax(pred,axis=1)
        if mind_state == 0:
            calm_negative_state()
        elif mind_state == 1:
            st.write("Neutral state.")
        elif mind_state == 2:
            st.write("Positive state.")
        else:
            st.write("Invalid mind state detected.")

    # st.text(mind_state)

    # For example, you might have code to load a model, preprocess the data, and make predictions
    # Note: Ensure you have the relevant code and imports needed for the LNN test

if __name__ == "__main__":
    run()
