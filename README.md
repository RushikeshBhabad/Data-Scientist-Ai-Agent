# 🧠 Data Scientist AI Agent

An all-in-one, AI-powered data science platform built with **Streamlit** and **LangChain**. Upload your data and get instant EDA, ML model training, dataset discovery, and an intelligent chatbot — all in one app.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3-orange)

---

## ✨ Features

### 📊 EDA Assistant
- Upload a CSV and get **automated exploratory data analysis** with AI insights
- Missing values, duplicates, correlations, outliers, distribution checks
- Variable-level deep dives with interactive Plotly charts
- Auto-generated EDA Python scripts (downloadable)
- Sweetviz one-click HTML reports
- LLM-powered feature engineering suggestions

### 📁 Dataset Finder
- Search datasets from **Kaggle**, **GitHub**, **Data.gov**, and **UCI ML Repository**
- AI-enhanced search — LLM generates related keywords automatically
- AI-ranked results by relevance
- Export search results as CSV or Markdown report

### 🤖 ML Model Training
- **LLM-generated training code** — AI writes sklearn code tailored to your dataset
- Automatic classification vs. regression detection
- AutoML pipeline with PyCaret (Python 3.9–3.11)
- Hyperparameter tuning: Auto Tune, GridSearchCV, or LLM-generated code
- Model download (`.pkl`)

### 💬 Chatbot Assistant
- Chat with your data using natural language
- Multi-session chat management with history
- Upload CSV/Excel/TXT/PDF files directly in chat
- DataFrame Agent mode for direct data manipulation
- Auto-generated data analysis reports

### 🔥 Data Analysis Agent *(Enterprise-Grade)*
- **Autonomous AI agent** with self-correcting code execution
- Two specialized agents: Pandas Analysis + Visualization
- Intelligence layers: Critic, Planner, Confidence Scorer, Self-Reflection
- **Enterprise EDA**: AI plans → generates code → executes on full data → interprets results
- Auto-generated visualizations (histograms, KDE, boxplots, violins, QQ plots, heatmaps, scatter, pairplots)
- Interactive chat with data — ask anything, get code + results + plots

---

## 🏗️ Project Structure

```
├── app_main.py                  # Main Streamlit entry point
├── app_chatbot.py               # Chatbot Assistant module
├── app_Dataset.py               # Dataset Finder module
├── app_EDA.py                   # EDA Assistant module
├── app_model.py                 # ML Model Training module
├── apikey.py                    # API key loader (from .env)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
│
├── pages/                       # Streamlit multi-page navigation
│   ├── Chatbot_Assistant.py
│   ├── Dataset_Finder.py
│   ├── EDA_Assistant.py
│   ├── Model_Training.py
│   └── Data_Analysis_Agent.py
│
└── data_analysis_agent/         # Enterprise-grade analysis agent
    ├── agent.py                 # Core agent logic (1600+ lines)
    ├── app.py                   # Streamlit UI for the agent
    └── __init__.py
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-data-science-agent.git
cd ai-data-science-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
GROQ_API_KEY=your_groq_api_key_here          # Required — https://console.groq.com
GITHUB_TOKEN=your_github_token_here          # Optional — for dataset search
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here  # Optional — for HuggingFace models
```

### 5. Run the app

```bash
streamlit run app_main.py
```

Open http://localhost:8501 in your browser.

---

## 🔑 API Keys

| Key | Required | Get it from |
|-----|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) |
| `GITHUB_TOKEN` | Optional | [github.com/settings/tokens](https://github.com/settings/tokens) |
| `HUGGINGFACEHUB_API_TOKEN` | Optional | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

For Kaggle dataset search, place your `kaggle.json` in `~/.kaggle/`.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (LLaMA 3.3 70B Versatile)
- **Orchestration**: LangChain, LangChain Experimental
- **Data Science**: pandas, NumPy, scikit-learn, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly, Sweetviz
- **AutoML**: PyCaret (Python 3.9–3.11)
- **APIs**: Kaggle, GitHub, Data.gov

---

## 📋 Notes

- **PyCaret** requires Python 3.9–3.11. If using a newer Python version, AutoML features are gracefully disabled and all other features work normally.
- The app uses **Groq** as the primary LLM provider (free tier available). The `llama-3.3-70b-versatile` model is used across all components.
- The Data Analysis Agent can also use **HuggingFace** models as a fallback if `HUGGINGFACEHUB_API_TOKEN` is set.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
