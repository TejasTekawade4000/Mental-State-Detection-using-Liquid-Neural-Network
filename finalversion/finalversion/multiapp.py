import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []
        self.default_app = None

    def add_app(self, title, func, is_default=False):
        self.apps.append({
            "title": title,
            "function": func
        })
        if is_default:
            self.default_app = title

    def run(self):
        if 'current_app' not in st.session_state:
            st.session_state.current_app = self.default_app

        st.sidebar.title('Navigation')
        for a in self.apps:
            if st.sidebar.button(a['title']):
                st.session_state.current_app = a['title']

        if st.session_state.current_app:
            app = next((a for a in self.apps if a['title'] == st.session_state.current_app), None)
            if app:
                app['function']()
