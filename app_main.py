import streamlit as st

st.set_page_config(
    page_title="AI Data Science Agent",
    layout="wide",
    initial_sidebar_state="collapsed"  # ✅ this collapses sidebar by default
)

st.title("🧠 AI Data Science Agent")

st.markdown("""
Welcome to your all-in-one AI-powered data science assistant!  
Use the arrow at the top left to open the sidebar and switch between:
- 📊 EDA Assistant
- 📁 Dataset Finder
- 🤖 ML Model Training
- 💬 Chatbot Assistant
""")
