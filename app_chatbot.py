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
import uuid
from datetime import datetime

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def run():
    # Initialize LLM
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192"
    )



    # Custom CSS for modern interface
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            background-color: #fafafa;
        }
        
        /* ===== Enhanced File Uploader Styling ===== */
        /* Main button style */
        div[data-testid="stFileUploader"] > label {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            border: none !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
            min-width: 200px !important;
            height: auto !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
        }
        
        /* Button hover state */
        div[data-testid="stFileUploader"] > label:hover {
            background: linear-gradient(135deg, #5a6fd1 0%, #6a4199 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        }
        
        /* Button active state */
        div[data-testid="stFileUploader"] > label:active {
            transform: translateY(0) !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        }
        
        /* Hide the "Drag and drop files here" text */
        div[data-testid="stFileUploader"] > div > div > div {
            display: none !important;
        }
        
        /* Remove the default border and padding */
        div[data-testid="stFileUploader"] {
            border: none !important;
            padding: 0 !important;
        }
        
        /* File uploader icon */
        div[data-testid="stFileUploader"] > label > div:first-child {
            margin-right: 8px !important;
        }
        
        /* ===== Rest of your styles ===== */
        .chat-input-container {
            position: sticky;
            bottom: 0;
            background: white;
            padding: 1rem;
            border-top: 1px solid #e0e0e0;
        }
        
        .sidebar-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .chat-history-item {
            padding: 0.5rem;
            margin: 0.2rem 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .chat-history-item:hover {
            background-color: #f0f2f6;
            transform: translateX(5px);
        }
        
        .new-chat-btn {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.7rem;
            border-radius: 8px;
            font-weight: bold;
            margin-bottom: 1rem;
            cursor: pointer;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background-color: #4CAF50; }
        .status-processing { background-color: #FF9800; }
        .status-error { background-color: #F44336; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    def initialize_session_state():
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
        
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        
        if 'uploaded_files_data' not in st.session_state:
            st.session_state.uploaded_files_data = {}
        
        if 'system_status' not in st.session_state:
            st.session_state.system_status = "online"

    initialize_session_state()

    def create_new_chat_session():
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        st.session_state.chat_sessions[session_id] = {
            'id': session_id,
            'title': f"Chat {len(st.session_state.chat_sessions) + 1}",
            'messages': [],
            'created_at': datetime.now(),
            'dataframe': None,
            'files': []
        }
        st.session_state.current_session_id = session_id
        return session_id

    def get_current_session():
        """Get current chat session"""
        if not st.session_state.current_session_id or st.session_state.current_session_id not in st.session_state.chat_sessions:
            create_new_chat_session()
        return st.session_state.chat_sessions[st.session_state.current_session_id]

    def update_session_title(session_id, title):
        """Update session title based on first message"""
        if session_id in st.session_state.chat_sessions:
            st.session_state.chat_sessions[session_id]['title'] = title[:50] + "..." if len(title) > 50 else title

    def delete_chat_session(session_id):
        """Delete a chat session"""
        if session_id in st.session_state.chat_sessions:
            del st.session_state.chat_sessions[session_id]
        
        if st.session_state.current_session_id == session_id:
            if st.session_state.chat_sessions:
                st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]
            else:
                create_new_chat_session()

    def process_uploaded_file(uploaded_file, session_id):
        """Process uploaded file and return appropriate response"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['csv']:
                df = pd.read_csv(uploaded_file)
                st.session_state.chat_sessions[session_id]['dataframe'] = df
                st.session_state.chat_sessions[session_id]['files'].append(uploaded_file.name)
                return f"📊 CSV file '{uploaded_file.name}' uploaded successfully!\n\n**Dataset Info:**\n- Shape: {df.shape}\n- Columns: {', '.join(df.columns.tolist())}\n\n**Preview:**\n{df.head(3).to_string()}"
            
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
                st.session_state.chat_sessions[session_id]['dataframe'] = df
                st.session_state.chat_sessions[session_id]['files'].append(uploaded_file.name)
                return f"📈 Excel file '{uploaded_file.name}' uploaded successfully!\n\n**Dataset Info:**\n- Shape: {df.shape}\n- Columns: {', '.join(df.columns.tolist())}\n\n**Preview:**\n{df.head(3).to_string()}"
            
            elif file_extension in ['txt']:
                content = uploaded_file.read().decode('utf-8')
                st.session_state.chat_sessions[session_id]['files'].append(uploaded_file.name)
                return f"📄 Text file '{uploaded_file.name}' uploaded successfully!\n\n**Content preview:**\n{content[:1000]}{'...' if len(content) > 1000 else ''}"
            
            elif file_extension in ['pdf']:
                st.session_state.chat_sessions[session_id]['files'].append(uploaded_file.name)
                return f"📑 PDF file '{uploaded_file.name}' uploaded successfully! Note: PDF text extraction requires additional libraries. I can help you analyze the structure or provide guidance on processing PDF files."
            
            elif file_extension in ['png', 'jpg', 'jpeg']:
                st.session_state.chat_sessions[session_id]['files'].append(uploaded_file.name)
                return f"🖼️ Image file '{uploaded_file.name}' uploaded successfully! I can help you with image analysis concepts, though direct image processing would require computer vision libraries."
            
            else:
                st.session_state.chat_sessions[session_id]['files'].append(uploaded_file.name)
                return f"📎 File '{uploaded_file.name}' uploaded. I can provide guidance on how to work with {file_extension} files in data science projects."
                
        except Exception as e:
            return f"❌ Error processing file '{uploaded_file.name}': {str(e)}"

    def generate_response(user_input, session_id, use_dataframe_agent=False):
        """Generate response using the LLM"""
        try:
            st.session_state.system_status = "processing"
            session = st.session_state.chat_sessions[session_id]
            
            # Prepare context
            context = "You are a helpful Data Scientist Assistant. You help with data analysis, machine learning, statistics, visualization, and general data science tasks. Provide clear, practical, and actionable advice."
            
            # Add dataframe context if available
            if session['dataframe'] is not None:
                df_info = f"\n\n**Current DataFrame Info:**\n- Shape: {session['dataframe'].shape}\n- Columns: {', '.join(session['dataframe'].columns.tolist())}\n- Data types: {dict(session['dataframe'].dtypes)}"
                context += df_info
            
            # Add chat history context
            if session['messages']:
                recent_history = session['messages'][-6:]  # Last 6 exchanges
                history_context = "\n\n**Recent conversation:**\n"
                for msg in recent_history:
                    history_context += f"{msg['role']}: {msg['content'][:200]}...\n"
                context += history_context
            
            # Create the full prompt
            full_prompt = f"{context}\n\nUser: {user_input}\n\nAssistant:"
            
            # Generate response
            message = HumanMessage(content=full_prompt)
            response = llm.invoke([message])
            
            st.session_state.system_status = "online"
            return response.content
            
        except Exception as e:
            st.session_state.system_status = "error"
            return f"❌ Error generating response: {str(e)}"

    def create_dataframe_agent_response(user_input, session_id):
        """Create response using pandas dataframe agent"""
        try:
            session = st.session_state.chat_sessions[session_id]
            if session['dataframe'] is not None:
                st.session_state.system_status = "processing"
                agent = create_pandas_dataframe_agent(
                    llm=llm,
                    df=session['dataframe'],
                    verbose=True,
                    allow_dangerous_code=True
                )
                response = agent.run(user_input)
                st.session_state.system_status = "online"
                return response
            else:
                return "❌ No dataframe loaded. Please upload a CSV or Excel file first."
        except Exception as e:
            st.session_state.system_status = "error"
            return f"❌ Error with dataframe agent: {str(e)}"

    def generate_data_report(session_id):
        """Generate automated data analysis report"""
        session = st.session_state.chat_sessions[session_id]
        if session['dataframe'] is not None:
            try:
                df = session['dataframe']
                
                # Basic info
                report = f"## 📊 Automated Data Analysis Report\n\n"
                report += f"**📈 Dataset Overview:**\n"
                report += f"- **Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n"
                report += f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
                
                # Data types
                report += f"**🔍 Column Information:**\n"
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    report += f"- {dtype}: {count} columns\n"
                
                # Missing values
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    report += f"\n**⚠️ Missing Values:**\n"
                    for col, count in missing[missing > 0].items():
                        percentage = count/len(df)*100
                        report += f"- **{col}:** {count} ({percentage:.1f}%)\n"
                else:
                    report += f"\n**✅ No missing values found!**\n"
                
                # Numerical summary
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    report += f"\n**📊 Numerical Statistics:**\n"
                    desc = df[numeric_cols].describe()
                    report += f"```\n{desc.to_string()}\n```\n"
                
                # Categorical summary
                cat_cols = df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    report += f"\n**📝 Categorical Columns:**\n"
                    for col in cat_cols[:5]:  # Show first 5 categorical columns
                        unique_count = df[col].nunique()
                        report += f"- **{col}:** {unique_count} unique values\n"
                
                return report
                
            except Exception as e:
                return f"❌ Error generating report: {str(e)}"
        else:
            return "❌ No dataframe loaded. Please upload a data file first."

    # Sidebar for chat management
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h2>🤖 Data Scientist</h2><p>AI Assistant</p></div>', unsafe_allow_html=True)
        
        # System status
        status_class = f"status-{st.session_state.system_status}"
        status_text = st.session_state.system_status.title()
        st.markdown(f'<div><span class="status-indicator {status_class}"></span>{status_text}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # New chat button
        if st.button("➕ New Chat", key="new_chat", help="Start a new conversation"):
            create_new_chat_session()
            st.rerun()
        
        # Chat history
        st.subheader("💬 Chat History")
        
        if st.session_state.chat_sessions:
            for session_id, session in st.session_state.chat_sessions.items():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    is_current = session_id == st.session_state.current_session_id
                    button_style = "primary" if is_current else "secondary"
                    
                    if st.button(session['title'], key=f"chat_{session_id}", 
                            help=f"Created: {session['created_at'].strftime('%Y-%m-%d %H:%M')}", 
                            type=button_style):
                        st.session_state.current_session_id = session_id
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{session_id}", help="Delete chat"):
                        delete_chat_session(session_id)
                        st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        use_dataframe_agent = st.toggle("🤖 DataFrame Agent", help="Enable for direct data manipulation")
        show_timestamps = st.toggle("🕒 Show Timestamps", value=True)
        
        # Current dataset info
        current_session = get_current_session()
        if current_session['dataframe'] is not None:
            st.subheader("📊 Current Dataset")
            df = current_session['dataframe']
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
            st.metric("Files", len(current_session['files']))
            
            if st.button("📈 Generate Report", key="generate_report"):
                report = generate_data_report(st.session_state.current_session_id)
                current_session['messages'].append({
                    'role': 'assistant',
                    'content': report,
                    'timestamp': datetime.now()
                })
                st.rerun()

    # Main chat interface
    st.markdown('<div class="main-header"><h1>🤖 Data Scientist Assistant</h1><p>Your AI companion for data analysis, machine learning, and insights</p></div>', unsafe_allow_html=True)

    # Chat messages container
    current_session = get_current_session()

    # File upload in chat area - replace your current col1, col2, col3 section with this
    upload_col, spacer_col = st.columns([6, 4])

    with upload_col:
        uploaded_files = st.file_uploader(
            "📤 Upload Files",
            type=['csv', 'xlsx', 'xls', 'txt', 'pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload files for analysis (CSV, Excel, Text, PDF, Images)"
        )

    with spacer_col:
        st.write("")  # Spacer

    # Display chat messages
    chat_container = st.container()

    with chat_container:
        if current_session['messages']:
            for i, message in enumerate(current_session['messages']):
                timestamp = message.get('timestamp', datetime.now())
                
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                        if show_timestamps:
                            st.caption(f"🕒 {timestamp.strftime('%H:%M:%S')}")
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        if show_timestamps:
                            st.caption(f"🕒 {timestamp.strftime('%H:%M:%S')}")
        else:
            # Welcome message for new chats
            with st.chat_message("assistant"):
                st.write("""
                👋 **Welcome to your Data Scientist Assistant!**
                
                I'm here to help you with:
                - 📊 **Data Analysis** - Upload CSV/Excel files for instant insights
                - 🤖 **Machine Learning** - Get guidance on ML algorithms and implementation
                - 📈 **Visualization** - Create compelling charts and graphs
                - 🔍 **Statistical Analysis** - Perform hypothesis testing and statistical modeling
                - 💡 **Data Science Consulting** - Best practices and methodology advice
                
                **Quick Actions:**
                - Upload a file using the 📎 button
                - Ask me anything about your data
                - Type "help" for more examples
                
                Let's start exploring your data! 🚀
                """)

    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in current_session['files']:
                file_response = process_uploaded_file(file, st.session_state.current_session_id)
                current_session['messages'].append({
                    'role': 'assistant',
                    'content': file_response,
                    'timestamp': datetime.now()
                })
                st.rerun()

    # Chat input
    user_input = st.chat_input("💬 Ask me anything about data science, upload files, or analyze your data...")

    # Process user input
    if user_input:
        # Update session title if it's the first message
        if len(current_session['messages']) == 0:
            update_session_title(st.session_state.current_session_id, user_input)
        
        # Add user message
        current_session['messages'].append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Generate and add assistant response
        with st.spinner("🤔 Thinking..."):
            if use_dataframe_agent and current_session['dataframe'] is not None:
                response = create_dataframe_agent_response(user_input, st.session_state.current_session_id)
            else:
                response = generate_response(user_input, st.session_state.current_session_id)
        
        current_session['messages'].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        })
        
        st.rerun()

    # Quick suggestions for empty chats
    if len(current_session['messages']) <= 1:
        st.markdown("### 💡 Try these examples:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Analyze my dataset", key="suggestion1"):
                if current_session['dataframe'] is not None:
                    user_input = "Please analyze my dataset and provide key insights"
                else:
                    user_input = "How do I start analyzing a dataset? What are the key steps?"
                
                current_session['messages'].append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
                })
                
                response = generate_response(user_input, st.session_state.current_session_id)
                current_session['messages'].append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
                st.rerun()
        
        with col2:
            if st.button("🤖 ML Algorithm Help", key="suggestion2"):
                user_input = "What machine learning algorithm should I choose for my problem?"
                current_session['messages'].append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
                })
                
                response = generate_response(user_input, st.session_state.current_session_id)
                current_session['messages'].append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
                st.rerun()
        
        with col3:
            if st.button("📈 Create Visualization", key="suggestion3"):
                user_input = "How do I create effective data visualizations?"
                current_session['messages'].append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
                })
                
                response = generate_response(user_input, st.session_state.current_session_id)
                current_session['messages'].append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("*🚀 Powered by Groq AI & Streamlit | Enhanced Data Scientist Assistant v2.0*")