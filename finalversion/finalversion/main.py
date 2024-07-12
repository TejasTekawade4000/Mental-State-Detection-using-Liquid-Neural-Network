import streamlit as st
from multiapp import MultiApp
import Home
import LNN_Test as LNN_Test
import Docs

st.set_page_config(
    page_title="LNN Project",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': "https://www.linkedin.com/in/pratham-solanki01/",
        'About': "here is a paper link for the project"
    }
)

app = MultiApp()

with st.sidebar:
    st.image('Loading_Neural_Brains.gif')
    st.title('Emotion detection')
    st.info('This Application demonstrates an efficient DL model for EEG devices')

# Add all your application here
app.add_app("Documentation", Docs.run, is_default=True)
app.add_app("CSV generator", Home.run)
app.add_app("LNN Test and View Data", LNN_Test.run)

# The main app
app.run()
