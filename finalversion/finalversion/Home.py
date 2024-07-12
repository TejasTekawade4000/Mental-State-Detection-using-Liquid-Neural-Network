import streamlit as st
import pandas as pd
from datetime import datetime
import os

def run():
    st.title("CSV Processing App")

    # Define the default output folder inside the 'appmain' folder
    default_output_folder = os.path.join(os.getcwd(), 'appmain', 'results')
    if not os.path.exists(default_output_folder):
        os.makedirs(default_output_folder)

    # Inputs
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    patient_name = st.text_input("Name")
    patient_id = st.text_input("Patient ID")
    date_time = st.date_input("Date")
    time = st.time_input("Time")
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    output_folder = st.text_input("Output Folder Path", default_output_folder)
    row_number = st.slider("Row Number", min_value=0, max_value=1000, step=1)

    if st.button("Process"):
        if uploaded_file is not None:
            try:
                # Read the CSV
                df = pd.read_csv(uploaded_file)
                
                # Separate labels (Y) and features (X)
                Y = df.iloc[:, -1]
                # Keep all columns from 'fft_0_a' to 'fft_749_b'
                start_col = df.columns.get_loc("fft_0_a")
                end_col = df.columns.get_loc("fft_749_b") + 1
                X = df.iloc[:, start_col:end_col]

                # Perform label mapping
                label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
                Y = Y.map(label_mapping)

                # Merge X and Y
                merged_df = pd.concat([X, Y], axis=1)

                # Ensure row_number is within range
                row_number = min(row_number, len(merged_df) - 1)

                # Extract the desired row
                selected_row = merged_df.iloc[[row_number]]

                # Add patient details
                selected_row['Name'] = patient_name
                selected_row['PatientID'] = patient_id
                selected_row['Age'] = age
                selected_row['Gender'] = gender
                selected_row['Date'] = date_time
                selected_row['Time'] = time

                # Reorder columns to place patient details at the beginning
                cols = ['Name', 'PatientID', 'Age', 'Gender', 'Date', 'Time'] + selected_row.columns.tolist()[:-6]
                selected_row = selected_row[cols]

                # Create the output folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Save the new dataframe to a CSV file
                timestamp = datetime.combine(date_time, time).strftime("%Y%m%d_%H%M%S")
                output_filename = f"{patient_name}_{timestamp}.csv"
                output_path = os.path.join(output_folder, output_filename)

                selected_row.to_csv(output_path, index=False)
                
                st.success(f"File saved as {output_path}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload a CSV file")
