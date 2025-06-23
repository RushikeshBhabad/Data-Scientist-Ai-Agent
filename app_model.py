import os
from apikey import groq_api_key

import streamlit as st
import pandas as pd
import plotly.express as px
import sweetviz as sv
import io
import base64
import re
import sys
import matplotlib.pyplot as plt
import numpy as np


from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

from pycaret.classification import (
    setup as cls_setup,
    compare_models as cls_compare,
    pull as cls_pull,
    plot_model as cls_plot_model,
    save_model as cls_save_model,
    create_model as cls_create_model,
    tune_model as cls_tune_model,
    evaluate_model as cls_evaluate_model,
)

from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    pull as reg_pull,
    plot_model as reg_plot_model,
    save_model as reg_save_model,
    create_model as reg_create_model,
    tune_model as reg_tune_model,
    evaluate_model as reg_evaluate_model
)

from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split

# --- Patch <think> blocks ---
def patch_think_blocks(code: str) -> str:
    # Patch <think>...</think> to valid Python docstring
    code = re.sub(r'<think>(.*?)</think>', r'"""\n<think>\1</think>\n"""', code, flags=re.DOTALL)
    # Remove any Markdown-style fenced code blocks
    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```", "", code)
    return code.strip()

# Safe execution (optional use)
def safe_execute(code: str) -> str:
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        return str(local_vars)
    except Exception as e:
        return f"Error: {e}"

def run():
    # Load env
    load_dotenv(find_dotenv())

    # Initialize LLM
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=groq_api_key,
        model_name="deepseek-r1-distill-llama-70b"
    )

    # Initialize session state
    if 'model_training_df' not in st.session_state:
        st.session_state.model_training_df = None
    if 'automl_df' not in st.session_state:
        st.session_state.automl_df = None
    if 'model_training_file_uploaded' not in st.session_state:
        st.session_state.model_training_file_uploaded = False
    if 'automl_file_uploaded' not in st.session_state:
        st.session_state.automl_file_uploaded = False


    # ================================== MODEL SELECTION & TRAINING =============================================
    st.title("🤖 ML Model Training & AutoML Pipeline")

    st.header("🚀 Model Selection & Training")

    if st.button("🚀 Start Model Selection & Training", key="model_training_btn"):
        st.session_state.model_training_file_uploaded = True
        st.session_state.model_training_df = None  # Reset dataframe

    # Show file uploader for Model Training if button clicked
    if st.session_state.model_training_file_uploaded:
        st.subheader("📁 Upload CSV for Model Training")
        uploaded_file_training = st.file_uploader(
            "Upload your CSV file for Model Training", 
            type=["csv"], 
            key="model_training_uploader"
        )
        
        if uploaded_file_training is not None:
            try:
                # Load and store dataframe in session state
                st.session_state.model_training_df = pd.read_csv(uploaded_file_training)
                st.success("✅ File uploaded successfully for Model Training!")
                
                # Show preview
                st.subheader("📊 Preview of your dataset")
                st.dataframe(st.session_state.model_training_df.head())
                
                # Execute Model Training
                st.subheader("⚙️ Executing Model Selection & Training")
                
                with st.spinner("Analyzing and selecting model..."):
                    df = st.session_state.model_training_df
                    
                    # Step 1: Let LLM decide model type and generate training code
                    try:
                        from langchain.schema import HumanMessage  # Adjust import based on your setup
                        # Assuming 'llm' is defined elsewhere in your code
                        
                        prompt = HumanMessage(
                            content=f"""You are a professional data scientist.
                                Given this dataset:

                                {df.head(100).to_csv(index=False)}

                                1. Detect whether this is a classification or regression task.
                                2. Choose the best sklearn model.
                                3. Write Python code to:
                                - Preprocess the data (handle encoding (label and categorical both) and all stuff and define all variables properly)
                                - Split into train/test
                                - Train the model (random forest tress like model prefered)
                                - Evaluates using these metrics:
                                        Classification: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
                                        Regression: R², MAE, MSE, RMSE
                                - Print final evaluation metrics and confusion matrix
                                Ensure the code runs standalone assuming 'df' is already loaded as a Pandas DataFrame.
                                Just output Python code, no explanation.
                                """
                            )

                        response = llm([prompt])
                        generated_code = response.content

                        # 🩹 Patch <think> tags before execution
                        patched_code = patch_think_blocks(generated_code)

                        st.subheader("🧠 Generated Model Code")
                        st.code(patched_code, language="python")
                        st.download_button("💾 Download Code", patched_code, file_name="generated_model.py")

                        # Step 2: Run the generated code
                        st.subheader("⚙️ Training Output")
                        with st.spinner("Executing training code..."):
                            exec_env = {'df': df}
                            try:
                                # Capture stdout
                                old_stdout = sys.stdout
                                sys.stdout = mystdout = io.StringIO()

                                # Execute the generated code
                                exec(patched_code, {}, exec_env)

                                # Reset stdout and get output
                                sys.stdout = old_stdout
                                output = mystdout.getvalue()

                                st.success("✅ Model training completed.")
                                st.subheader("🖨️ Model Output:")
                                st.code(output)

                                # Display scalar values
                                output_vars = {k: v for k, v in exec_env.items() if not k.startswith("__")}
                                for key, val in output_vars.items():
                                    if isinstance(val, (float, int, str)):
                                        st.write(f"**{key}**: {val}")

                            except Exception as e:
                                sys.stdout = old_stdout  # Restore stdout
                                st.error(f"❌ Error during execution: {e}")
                                
                    except ImportError:
                        st.error("❌ LLM dependencies not available. Please install required packages.")
                    except Exception as e:
                        st.error(f"❌ Error in model training setup: {e}")
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    # ================================== AUTO ML =====================================================================
    st.header("🧪 AutoML Pipeline")

    # AutoML Button
    if st.button("🧪 Start AutoML", key="automl_btn"):
        st.session_state.automl_file_uploaded = True
        st.session_state.automl_df = None

    # Upload CSV
    if st.session_state.get("automl_file_uploaded", False):
        st.subheader("📁 Upload CSV for AutoML")
        uploaded_file_automl = st.file_uploader("Upload CSV", type=["csv"], key="automl_uploader")

        if uploaded_file_automl is not None:
            try:
                df = pd.read_csv(uploaded_file_automl)
                st.session_state.automl_df = df
                st.success("✅ File uploaded successfully!")

                st.subheader("📊 Dataset Preview")
                st.dataframe(df.head())

                target_column = st.selectbox("🎯 Select the target column", df.columns)

                if st.button("▶️ Execute AutoML Pipeline", key="execute_automl"):
                    with st.spinner("⏳ Running AutoML pipeline..."):
                        try:
                            y_type = type_of_target(df[target_column])
                            task_type = "classification" if y_type in ["binary", "multiclass"] else "regression"
                            st.write(f"📌 Detected task type: **{task_type.capitalize()}**")

                            model_fallback_used = False
                            best_model, result_df = None, None

                            if task_type == "classification":
                                from pycaret.classification import (
                                    setup, compare_models, pull,
                                    plot_model, save_model, get_config
                                )

                                setup(data=df, target=target_column, session_id=123, html=False, verbose=False)
                                X_train = get_config('X_train')
                                X_test = get_config('X_test')

                                st.markdown(f"🔢 **Training Examples:** {len(X_train)}")
                                st.markdown(f"🔬 **Testing Examples:** {len(X_test)}")

                                try:
                                    best_model = compare_models(sort="F1")
                                    result_df = pull()
                                except Exception as e:
                                    st.warning(f"⚠️ compare_models failed: {e}")
                                    st.info("➡️ Using fallback: LogisticRegression")
                                    from sklearn.linear_model import LogisticRegression
                                    X = df.drop(columns=[target_column])
                                    y = df[target_column]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
                                    best_model = LogisticRegression()
                                    best_model.fit(X_train, y_train)
                                    model_fallback_used = True

                            else:
                                from pycaret.regression import (
                                    setup, compare_models, pull,
                                    plot_model, save_model, get_config
                                )

                                setup(data=df, target=target_column, session_id=123, html=False, verbose=False)
                                X_train = get_config('X_train')
                                X_test = get_config('X_test')

                                st.markdown(f"🔢 **Training Examples:** {len(X_train)}")
                                st.markdown(f"🔬 **Testing Examples:** {len(X_test)}")

                                try:
                                    best_model = compare_models(sort="R2")
                                    result_df = pull()
                                except Exception as e:
                                    st.warning(f"⚠️ compare_models failed: {e}")
                                    st.info("➡️ Using fallback: LinearRegression")
                                    from sklearn.linear_model import LinearRegression
                                    X = df.drop(columns=[target_column])
                                    y = df[target_column]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
                                    best_model = LinearRegression()
                                    best_model.fit(X_train, y_train)
                                    model_fallback_used = True

                            if best_model is None:
                                raise ValueError("❌ No valid model found!")

                            if not model_fallback_used:
                                st.subheader("🏆 Model Leaderboard")
                                st.dataframe(result_df)

                            st.subheader("🥇 Best Model Summary")
                            st.code(str(best_model), language="python")

                            st.subheader("📊 Evaluation Plots")
                            try:
                                if task_type == "classification" and not model_fallback_used:
                                    plot_model(best_model, plot='confusion_matrix', display_format='streamlit')
                                    if hasattr(best_model, "predict_proba"):
                                        plot_model(best_model, plot='auc', display_format='streamlit')
                                    plot_model(best_model, plot='learning', display_format='streamlit')
                                elif task_type == "regression" and not model_fallback_used:
                                    plot_model(best_model, plot='residuals', display_format='streamlit')
                                    plot_model(best_model, plot='error', display_format='streamlit')
                            except Exception as e:
                                st.warning(f"⚠️ Plotting failed: {e}")

                            # 🔍 Feature Importance
                            st.subheader("🔍 Feature Importance")
                            fi_df = None
                            try:
                                if hasattr(best_model, "feature_importances_"):
                                    features = getattr(best_model, 'feature_names_in_', df.drop(columns=[target_column]).columns)
                                    fi_df = pd.DataFrame({
                                        "Feature": features,
                                        "Importance": best_model.feature_importances_
                                    })
                                elif hasattr(best_model, "coef_"):
                                    coef = best_model.coef_
                                    features = getattr(best_model, 'feature_names_in_', df.drop(columns=[target_column]).columns)
                                    if len(coef.shape) > 1:
                                        coef = coef.mean(axis=0)
                                    fi_df = pd.DataFrame({
                                        "Feature": features,
                                        "Importance": np.abs(coef)
                                    })
                                else:
                                    st.info("ℹ️ This model does not support feature importances.")
                            except Exception as e:
                                st.warning(f"⚠️ Feature importance error: {e}")

                            if fi_df is not None:
                                st.dataframe(fi_df.sort_values("Importance", ascending=False))

                            # 💾 Export Model
                            st.subheader("💾 Export Best Model")
                            model_path = "best_model.pkl"
                            if not model_fallback_used:
                                save_model(best_model, model_path.replace(".pkl", ""))
                            else:
                                import joblib
                                joblib.dump(best_model, model_path)

                            with open(model_path, "rb") as f:
                                st.download_button("📥 Download Model", f, file_name="best_model.pkl")

                        except Exception as e:
                            st.error(f"❌ AutoML failed: {e}")
            except Exception as e:
                st.error(f"❌ File processing error: {e}")

    # 🔄 Reset
    st.sidebar.header("🔄 Reset Options")
    if st.sidebar.button("Reset AutoML"):
        st.session_state.automl_file_uploaded = False
        st.session_state.automl_df = None
        st.rerun()
    # ================================= Hyperparameter tuning ============================================
    # --- Hyperparameter Tuning Section ---

    

    # Session state for UI control
    if "show_tuning_ui" not in st.session_state:
        st.session_state.show_tuning_ui = False

    st.markdown("## 🎛️ Hyperparameter Tuning")

    if st.button("🔁 Run Hyperparameter Tuning"):
        st.session_state.show_tuning_ui = True

    if st.session_state.show_tuning_ui:
        tune_option = st.radio("Choose tuning strategy:",
                            ["🔮 PyCaret Auto Tune",
                                "🧪 GridSearchCV Fallback",
                                "🧾 Generate Tuning Code"])

        uploaded_file = st.file_uploader("📂 Upload your dataset for tuning", type=["csv"])

        if uploaded_file is not None:
            df_tune = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded `{uploaded_file.name}` with shape {df_tune.shape}")

            target_col = st.selectbox("🎯 Select target column for tuning", df_tune.columns)

            task_type = "classification" if type_of_target(df_tune[target_col]) in ["binary", "multiclass"] else "regression"

            # --- PyCaret Auto Tune ---
            if tune_option == "🔮 PyCaret Auto Tune":
                from pycaret.classification import setup as cls_setup, create_model as cls_create_model, tune_model as cls_tune_model, evaluate_model as cls_evaluate_model
                from pycaret.regression import setup as reg_setup, create_model as reg_create_model, tune_model as reg_tune_model, plot_model as reg_plot_model

                with st.spinner("🔍 Tuning"):
                    try:
                        if task_type == "classification":
                            cls_setup(data=df_tune, target=target_col, session_id=123, html=False, verbose=False)
                            model = cls_create_model("rf")
                            try:
                                tuned_model = cls_tune_model(model)
                            except AttributeError as e:
                                if "_memory_full_transform" in str(e):
                                    st.warning("⚠️ Tuning failed due to sklearn pipeline issue. Using base model instead.")
                                    tuned_model = model
                                else:
                                    raise e
                            st.success("✅ Tuning Complete (PyCaret)")
                            cls_evaluate_model(tuned_model)
                            st.code(str(tuned_model), language="python")
                        else:
                            reg_setup(data=df_tune, target=target_col, session_id=123, html=False, verbose=False)
                            model = reg_create_model("rf")
                            try:
                                tuned_model = reg_tune_model(model)
                            except AttributeError as e:
                                if "_memory_full_transform" in str(e):
                                    st.warning("⚠️ Tuning failed due to sklearn pipeline issue. Using base model instead.")
                                    tuned_model = model
                                else:
                                    raise e
                            st.success("✅ Tuning Complete (PyCaret)")
                            reg_plot_model(tuned_model, plot='learning', display_format='streamlit')
                            st.code(str(tuned_model), language="python")

                    except Exception as e:
                        st.error(f"❌ PyCaret tuning failed: {e}")

            # --- GridSearchCV Fallback ---
            elif tune_option == "🧪 GridSearchCV Fallback":
                from sklearn.model_selection import GridSearchCV, train_test_split
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                from sklearn.linear_model import LogisticRegression, LinearRegression
                from sklearn.metrics import classification_report, mean_squared_error

                try:
                    X = df_tune.drop(columns=[target_col])
                    y = df_tune[target_col]
                    X_encoded = pd.get_dummies(X)
                    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

                    if task_type == "classification":
                        models_and_params = {
                            "RandomForest": (RandomForestClassifier(), {
                                'n_estimators': [50, 100],
                                'max_depth': [None, 10, 20],
                                'min_samples_split': [2, 5]
                            }),
                            "GradientBoosting": (GradientBoostingClassifier(), {
                                'n_estimators': [50, 100],
                                'learning_rate': [0.05, 0.1],
                                'max_depth': [3, 5]
                            }),
                            "DecisionTree": (DecisionTreeClassifier(), {
                                'max_depth': [None, 10, 20],
                                'min_samples_split': [2, 5]
                            }),
                            "LogisticRegression": (LogisticRegression(max_iter=1000), {
                                'C': [0.1, 1.0, 10.0],
                                'penalty': ['l2'],
                                'solver': ['lbfgs']
                            }),
                        }
                    else:
                        models_and_params = {
                            "RandomForest": (RandomForestRegressor(), {
                                'n_estimators': [50, 100],
                                'max_depth': [None, 10, 20],
                                'min_samples_split': [2, 5]
                            }),
                            "GradientBoosting": (GradientBoostingRegressor(), {
                                'n_estimators': [50, 100],
                                'learning_rate': [0.05, 0.1],
                                'max_depth': [3, 5]
                            }),
                            "DecisionTree": (DecisionTreeRegressor(), {
                                'max_depth': [None, 10, 20],
                                'min_samples_split': [2, 5]
                            }),
                            "LinearRegression": (LinearRegression(), {})  # Safe fallback
                        }

                    for model_name, (model, param_grid) in models_and_params.items():
                        st.write(f"🔍 Training and tuning: **{model_name}**")
                        try:
                            grid = GridSearchCV(model, param_grid, cv=3, verbose=1, n_jobs=-1, error_score='raise')
                            grid.fit(X_train, y_train)
                            st.success(f"✅ {model_name} tuning complete!")
                            st.write("🔧 Best Parameters:", grid.best_params_)

                            y_pred = grid.predict(X_test)
                            if task_type == "classification":
                                st.text("📊 Classification Report")
                                st.text(classification_report(y_test, y_pred))
                            else:
                                mse = mean_squared_error(y_test, y_pred)
                                st.write(f"📉 MSE ({model_name}): {mse:.4f}")
                        except Exception as e:
                            st.error(f"❌ {model_name} failed: {e}")

                except Exception as e:
                    st.error(f"❌ GridSearchCV failed: {e}")

            # --- Generate Tuning Code (LLM) ---
            elif tune_option == "🧾 Generate Tuning Code":
                st.subheader("🧾 Auto-Generated Tuning Code with LLM")

                llm = ChatGroq(
                    temperature=0.2,
                    groq_api_key=groq_api_key,
                    model_name="deepseek-r1-distill-llama-70b"
                )

                model_type = "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor"

                prompt = f"""
    You're a Python and ML expert. Generate a complete, clean Python script to perform hyperparameter tuning using GridSearchCV on a dataset.

    Details:
    - The dataset is a CSV file named 'your_dataset.csv'
    - The target column is: '{target_col}'
    - The task is: '{task_type}'
    - Use sklearn's '{model_type}' for modeling
    - Use train_test_split (80/20)
    - Encode categorical variables using pandas.get_dummies()
    - Include fitting, best params printing, and model evaluation (accuracy or MSE)
    - Add all necessary imports

    Output only the code without explanations.
    """

                with st.spinner("🧠 Generating code with LLM..."):
                    response = llm([HumanMessage(content=prompt)])
                    st.code(response.content.strip(), language="python")

        # Optional reset
        if st.button("🔄 Reset Tuning UI"):
            st.session_state.show_tuning_ui = False
