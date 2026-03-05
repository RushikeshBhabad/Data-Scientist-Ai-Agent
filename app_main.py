import streamlit as st

st.set_page_config(
    page_title="Data Scientist AI Agent",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Data Scientist Agent")

st.markdown("""
Welcome to your all-in-one AI-powered data science assistant!  
Use the arrow at the top left to open the sidebar and switch between:

- 📊 **EDA Assistant** — Automated exploratory data analysis with AI insights
- 📁 **Dataset Finder** — Search Kaggle, GitHub, Data.gov for datasets
- 🤖 **ML Model Training** — Train models with AutoML and hyperparameter tuning
- 💬 **Chatbot Assistant** — Chat with your data using AI
- 🔥 **Data Analysis Agent** — Enterprise-grade autonomous analysis with self-correcting AI
""")
