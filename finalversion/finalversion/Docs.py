import streamlit as st

def run():
    st.title("EEG Emotion Detection with Liquid Neural Network")

    st.header("Project Overview")
    st.markdown("""
    This project aims to detect human emotions using EEG (electroencephalogram) signals and a Liquid Neural Network. 
    EEG signals are captured using specialized hardware, and these signals are then processed and analyzed to identify 
    various emotional states such as happiness, sadness, anger, and relaxation.
    """)

    st.header("How It Works")
    st.markdown("""
    1. **Data Collection**: 
       - EEG signals are collected from participants using an EEG headset.
       - The collected data is preprocessed to remove noise and artifacts.

    2. **Feature Extraction**: 
       - Relevant features are extracted from the EEG signals. 
       - These features include time-domain, frequency-domain, and time-frequency domain features.

    3. **Model Training**: 
       - A Liquid Neural Network (LNN) is used to train the model on the extracted features.
       - LNNs are a type of recurrent neural network known for their adaptability and efficiency in learning temporal dependencies.

    4. **Emotion Detection**: 
       - The trained model is used to predict the emotional state from new EEG data.
       - The output is the probability of different emotional states.

    """)

    st.header("Liquid Neural Network")
    st.markdown("""
    Liquid Neural Networks are a special class of recurrent neural networks (RNNs) that have a dynamic nature, allowing 
    them to better adapt and respond to time-varying inputs. They are particularly effective in applications involving 
    time-series data such as EEG signals.
    """)

    st.header("Streamlit App Functionality")
    st.markdown("""
    The Streamlit app provides a user-friendly interface for interacting with the EEG emotion detection model. 
    The key functionalities include:
    - **Upload EEG Data**: Users can upload their EEG data for analysis.
    - **Emotion Prediction**: The app processes the uploaded data and predicts the emotional state.
    - **Visualization**: The app provides visualizations of the EEG signals and the predicted emotions.

    """)
    
    st.header("Usage Instructions")
    st.markdown("""
    1. **Upload Data**: Click on the 'Upload EEG Data' button to upload your EEG data file.
    2. **Run Prediction**: Once the data is uploaded, click on the 'Predict Emotion' button to get the emotion predictions.
    3. **View Results**: The predicted emotions will be displayed along with visualizations of the EEG signals.
    """)

    st.header("Dependencies and Installation")
    st.markdown("""
    - **Python**: Ensure you have Python installed (version 3.7 or higher).
    - **Streamlit**: Install Streamlit using `pip install streamlit`.
    - **Required Libraries**: Install the required libraries using the following command:
      ```sh
      pip install numpy pandas scikit-learn keras tensorflow
      ```

    - **EEG Data Processing Libraries**: Install MNE and other necessary libraries using:
      ```sh
      pip install mne
      ```

    - **Run the App**: Navigate to the project directory and run the Streamlit app using:
      ```sh
      streamlit run main.py
      ```

    """)

    st.header("Authors and Acknowledgements")
    st.markdown("""
    **Author**: Pratham Solanki, Sanket Bartakke,Chaitanya Sarovar

    **Acknowledgements**: 
    - Thanks to the JORDAN J. BIRD the for EEG data .
    - Thanks to the Streamlit team for providing an excellent framework for building web apps.
    """)

    st.header("License")
    st.markdown("""
    This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.
    """)

if __name__ == "__main__":
    run()
