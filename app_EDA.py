# Import required libraries
import os 
from apikey import groq_api_key

import streamlit as st
import pandas as pd
import plotly.express as px
import sweetviz as sv
import io
import base64

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv,find_dotenv

# --- LLM Setup ---
@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=groq_api_key
    )

# --- Cached: EDA steps explanation ---
@st.cache_data
def steps_eda(_llm):
    steps = _llm([HumanMessage(content="What are the steps of EDA")])
    return steps.content

# --- Main Analysis Function ---
@st.cache_data
def function_agent(df, _agent):
    st.write("### 🧾 Data Overview")
    st.write("#### Preview of the Dataset")
    st.dataframe(df.head())

    st.write("### 🧹 Data Cleaning")
    st.write("🔸 Column Descriptions:")
    st.write(safe_agent_run(_agent, "What are the meaning of the columns?"))

    st.write("🔸 Missing Values:")
    st.write(safe_agent_run(_agent, "How many missing values does this dataframe have? Start the answer with 'There are'"))

    st.write("🔸 Duplicate Records:")
    st.write(safe_agent_run(_agent, "Are there any duplicate values and if so where?"))

    st.write("### 📊 Data Summarisation")
    st.write("#### Summary Statistics (Numerical Columns)")
    st.dataframe(df.describe())

    st.write("🔸 Correlation Analysis:")
    st.write(safe_agent_run(_agent, "Calculate correlations between numerical variables to identify potential relationships."))

    st.write("🔸 Outlier Detection:")
    st.write(safe_agent_run(_agent, "Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis."))

    st.write("🔸 New Feature Suggestions:")
    st.write(safe_agent_run(_agent, "What new features would be interesting to create?"))

    st.write("### 🧠 Additional Insights")

    st.write("🔸 Data Types and Conversion Suggestions:")
    st.write(safe_agent_run(_agent, "What are the datatypes of each column? Are there any columns that should be converted to another type for better analysis?"))

    st.write("🔸 Class Imbalance (for Classification Tasks):")
    st.write(safe_agent_run(_agent, "Is there any class imbalance in the target column or categorical columns?"))

    st.write("🔸 Distribution Check for Numerical Columns:")
    st.write(safe_agent_run(_agent, "How are the numerical columns distributed? Which ones are skewed?"))

    st.write("🔸 High Cardinality Columns (categorical):")
    st.write(safe_agent_run(_agent, "Which categorical columns have high cardinality that may need preprocessing?"))

    st.write("🔸 Suggestions for Data Cleaning:")
    st.write(safe_agent_run(_agent, "Based on the dataset, what cleaning steps do you recommend before modeling?"))

# Utility to safely run agent prompts
def safe_agent_run(agent, prompt: str):
    try:
        response = agent.run(prompt)
        print(f"[✅ Prompt]: {prompt}\n[🧠 Response]: {response}\n")
        return response
    except ValueError as e:
        if "Could not parse LLM output" in str(e):
            # Extract the actual response from the error message
            error_msg = str(e)
            if "Could not parse LLM output: `" in error_msg:
                start_idx = error_msg.find("Could not parse LLM output: `") + 30
                end_idx = error_msg.find("`", start_idx)
                if end_idx > start_idx:
                    extracted_response = error_msg[start_idx:end_idx]
                    print(f"[✅ Prompt]: {prompt}\n[🧠 Response]: {extracted_response}\n")
                    return extracted_response
            
            # Fallback: try a simpler prompt
            try:
                simplified_prompt = prompt.split("?")[0] + "?"  # Take first question only
                response = agent.run(simplified_prompt)
                print(f"[✅ Simplified Prompt]: {simplified_prompt}\n[🧠 Response]: {response}\n")
                return response
            except:
                pass
        
        print(f"[❌ Prompt]: {prompt}\n[⚠️ Error]: {str(e)}\n")
        return f"❌ Analysis unavailable for this query."
    except Exception as e:
        print(f"[❌ Prompt]: {prompt}\n[⚠️ Error]: {str(e)}\n")
        return f"❌ Agent error: Unable to process request."

@st.cache_data
def function_question_variable(df, _agent, column):
    st.write(f"## 🔍 Analysis of `{column}`")

    # Visual Overview
    st.write("### 📈 Visualisations")
    st.line_chart(df[[column]])
    st.bar_chart(df[[column]])
    
    if pd.api.types.is_numeric_dtype(df[column]):
        unique_vals = df[column].nunique()

        if unique_vals <= 10:
            st.write(f"#### ⚠️ `{column}` has only {unique_vals} unique values. Treating as categorical.")
            fig_bar = px.bar(df[column].value_counts().reset_index(), 
                            x='index', y=column, 
                            labels={'index': column, column: 'Count'},
                            title=f"Bar Chart of {column}")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.write("#### Histogram")
            fig_hist = px.histogram(df, x=column, nbins=30, title=f"Histogram of {column}")
            st.plotly_chart(fig_hist, use_container_width=True)

            st.write("#### Boxplot")
            fig_box = px.box(df, y=column, title=f"Boxplot of {column}")
            st.plotly_chart(fig_box, use_container_width=True)

    # Agent-driven Insights
    st.write("### 📊 LLM-based Summary")
    
    with st.expander("🔹 Summary Statistics"):
        st.write(safe_agent_run(_agent, f"Give me a summary of the statistics of {column}"))

    with st.expander("🔹 Distribution Check"):
        st.write(safe_agent_run(_agent, f"Check for normality or specific distribution shapes of {column}"))

    with st.expander("🔹 Outlier Analysis"):
        st.write(safe_agent_run(_agent, f"Assess the presence of outliers of {column}"))

    with st.expander("🔹 Trend and Seasonality"):
        st.write(safe_agent_run(_agent, f"Analyse trends, seasonality, and cyclic patterns of {column}"))

    with st.expander("🔹 Missing Value Inspection"):
        st.write(safe_agent_run(_agent, f"Determine the extent of missing values of {column}"))

    with st.expander("🔹 Column Type Recommendation"):
        st.write(safe_agent_run(_agent, f"What is the datatype of {column}, and should it be converted to another type?"))

@st.cache_data
def function_question_dataframe(_agent, question):
    st.write(_agent.run(question))

def run():
    #GROQApiKey
    os.environ['GROQ_API_KEY'] = groq_api_key
    load_dotenv(find_dotenv())

    #Title
    st.title('AI Agent for Data Science 🤖')

    #Welcoming message
    st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

    #Explanation sidebar
    with st.sidebar:
        st.write('*Your Data Science Adventure Begins with an CSV File.*')
        st.caption('''**You may already know that every exciting data science journey starts with a dataset.
        That's why I'd love for you to upload a CSV file.
        Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
        Then, we'll work together to shape your business challenge into a data science framework.
        I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
        ''')

        st.divider()

        st.caption("<p style ='text-align:center'> made by Rushikesh </p>",unsafe_allow_html=True )

    # -==========================================  Session state init ==================================================================
    if 'clicked' not in st.session_state:
        st.session_state.clicked = {1: False}

    if 'df' not in st.session_state:
        st.session_state.df = None

    # --- Click handler ---
    def clicked(button):
        st.session_state.clicked[button] = True
        st.session_state.df = None  # Reset dataset when clicking button again

    # --- Button with UNIQUE key ---
    st.button("Let's get started with EDA", key="start_eda_btn", on_click=clicked, args=[1])

    # --- MAIN LOGIC ---
    if st.session_state.clicked[1]:
        user_csv = st.file_uploader("Upload your file here", type="csv", key="csv_uploader")
        
        if user_csv is not None:
            user_csv.seek(0)
            st.session_state.df = pd.read_csv(user_csv, low_memory=False)
            st.success("✅ CSV loaded successfully!")

        if st.session_state.df is not None:
            df = st.session_state.df
            llm = get_llm()

            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )

            st.header('Exploratory Data Analysis')
            st.subheader('General Information About the Dataset')

            with st.sidebar:
                with st.expander('📋 What are the steps of EDA?'):
                    st.write(steps_eda(llm))

            function_agent(df, agent)

            st.subheader('📊 Variable of Study')
            user_question_variable = st.text_input('What variable are you interested in?', key="var_input")

            if user_question_variable:
                if user_question_variable not in df.columns:
                    st.error(f"⚠️ Variable '{user_question_variable}' not found in dataset columns.")
                else:
                    function_question_variable(df, agent, user_question_variable)

            st.subheader('🔎 Further Study')
            user_question_dataframe = st.text_input("Is there anything else you would like to know about your dataframe?", key="df_q_input")
            if user_question_dataframe and user_question_dataframe.lower() not in ("no", ""):
                function_question_dataframe(agent, user_question_dataframe)


    # =========================================== Generate EDA CODE =================================================================================================

    if 'generate_code_clicked' not in st.session_state:
        st.session_state.generate_code_clicked = False

    # === Button to trigger EDA Code Generator section ===
    if st.button("Let's Generate EDA Code"):
        st.session_state.generate_code_clicked = True

    # === Conditional rendering based on button click ===
    if st.session_state.generate_code_clicked:

        st.header("🧪 Auto EDA Code Generator")
        st.subheader("Code Generator")

        st.write("Upload a CSV file to generate complete, clean Python code for exploratory data analysis (EDA).")

        #LLM 
        llm = ChatGroq(
                temperature=0,
                model_name="llama3-70b-8192",  # Or another available model like "mixtral-8x7b-32768"
                groq_api_key=groq_api_key
        )

        # File uploader
        eda_csv = st.file_uploader("Upload CSV for EDA Code", type="csv", key="eda_code_upload")

        if eda_csv is not None:
            eda_csv.seek(0)
            eda_df = pd.read_csv(eda_csv)

            if st.button("📄 Generate Full EDA Code for This CSV"):
                with st.spinner("Generating EDA code..."):
                    # Use only first few rows for prompt, formatted as CSV text
                    csv_sample = eda_df.head(5).to_csv(index=False)

                    eda_code_prompt = f"""
                        You are an expert data scientist and Python developer.

                        Below is a **sample** from a CSV file (the user will run the full code on `data.csv`):

                        {csv_sample}

                        Based on the **structure and meaning of the data**, generate **clean, production-ready Python code** for Exploratory Data Analysis (EDA). Tailor the analysis to the dataset. Specifically:

                        1. Import required libraries: pandas, numpy, matplotlib, seaborn.
                        2. Load the CSV as `data = pd.read_csv("data.csv")`.
                        3. Show general info (shape, column names, types, head).
                        4. Identify and handle invalid or placeholder values (e.g., zeroes in biomedical columns).
                        5. Show missing value percentages and handle them properly (mean/median for numerics).
                        6. Print summary statistics for numerical and categorical features.
                        7. Show unique value counts for categorical variables.
                        8. Perform correlation analysis for numerical variables.
                        9. Create meaningful plots (histograms, boxplots, countplots, pairplots).
                        10. Detect and comment on outliers, skewness, and distribution issues.
                        11. Identify potential feature engineering or transformation steps.
                        12. Use clear comments and maintain clean structure throughout.

                        💡 Be **intelligent** and context-aware: e.g., if `Outcome` is binary, treat it as a label and analyze accordingly.

                        Return only valid, well-commented Python code that can run in a Jupyter Notebook or Python script.
                        """
                    # Generate response
                    code_response = llm([HumanMessage(content=eda_code_prompt)])
                    eda_code = code_response.content.strip()

                    # Display and download
                    st.code(eda_code, language="python")
                    st.download_button(
                        label="📥 Download Python EDA Script",
                        data=eda_code,
                        file_name="eda_script.py",
                        mime="text/x-python"
                    )


    # ============================================= Report Generator ===================================================

    if "generate_report_clicked" not in st.session_state:
        st.session_state.generate_report_clicked = False

    # === Button to reveal the profiler UI ===
    if st.button("Let's Generate EDA Report"):
        st.session_state.generate_report_clicked = True

    # === Conditional rendering ===
    if st.session_state.generate_report_clicked:
        st.header("🔍 One-Click EDA Profiler with Sweetviz")
        st.subheader("Generate a detailed interactive EDA report using `sweetviz`")

        # File Upload section
        uploaded_file = st.file_uploader("📎 Upload a CSV file", type=["csv"], key="eda_profiler_upload")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV file loaded successfully!")

            if st.button("📊 Generate EDA Report for This CSV"):
                with st.spinner("⏳ Generating HTML EDA report using Sweetviz..."):
                    report = sv.analyze(df)
                    html_path = "sweetviz_report.html"
                    report.show_html(html_path, open_browser=False)

                    # Load and display HTML as downloadable file
                    with open(html_path, "rb") as f:
                        html_bytes = f.read()

                    st.download_button(
                        label="💾 Download HTML Report",
                        data=html_bytes,
                        file_name="eda_report_sweetviz.html",
                        mime="text/html"
                    )

                    st.success("✅ EDA Report generated successfully!")

        else:
            st.info("📂 Please upload a CSV file to continue.")


    # ======================================= Feature Engineering ===========================================================
    llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)

    # === Session state to toggle FE assistant ===
    if "feature_engineering_clicked" not in st.session_state:
        st.session_state.feature_engineering_clicked = False

    # === Button to Show Feature Engineering Assistant ===
    if st.button("Launch Feature Engineering Assistant"):
        st.session_state.feature_engineering_clicked = True

    # === Feature Engineering UI ===
    if st.session_state.feature_engineering_clicked:
        st.header("🔧 Intelligent Feature Engineering Assistant")
        st.subheader("Get smart, LLM-powered suggestions to enhance your dataset")

        fe_file = st.file_uploader("📂 Upload a CSV File", type=["csv"], key="fe_upload")

        if fe_file:
            df = pd.read_csv(fe_file)
            st.success("✅ CSV loaded successfully!")
            st.write("🔍 **Preview of your data:**", df.head())

            if st.button("⚙️ Suggest Feature Engineering Steps"):
                with st.spinner("🧠 Thinking like a senior data scientist..."):
                    csv_sample = df.head(5).to_csv(index=False)

                    # === LLM Prompt ===
                    fe_prompt = f"""
                        You are an expert ML engineer and Python developer.

                        Below is a sample from a CSV file:

                        {csv_sample}

                        Please analyze the structure of the data and suggest:
                        1. New feature engineering ideas (e.g., BMI from weight/height, binning age).
                        2. Appropriate transformations (e.g., log, min-max, z-score).
                        3. Scaling advice (e.g., use StandardScaler or MinMaxScaler).
                        4. Whether dimensionality reduction (PCA or dropping features) is needed — based on correlation or low variance.
                        5. Return valid, runnable Python code blocks with clear comments.
                        6. Assume full dataset is already loaded as: `df = pd.read_csv("data.csv")`

                        FIRST PROVIDE THE SUMMARY IN THE PYTHON COMMENTS  AND AFTER THAT THEN PROVIDE THE PYTHON CODE ONLY 
                    """

                    fe_response = llm([HumanMessage(content=fe_prompt)])
                    fe_code = fe_response.content.strip()

                    st.subheader("💡 Suggested Feature Engineering Code")
                    st.code(fe_code, language="python")

                    st.download_button(
                        label="📥 Download Feature Engineering Code",
                        data=fe_code,
                        file_name="feature_engineering.py",
                        mime="text/x-python"
                    )
        else:
            st.info("📌 Please upload a CSV file to continue.")