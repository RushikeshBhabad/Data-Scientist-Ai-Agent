# 🔥 Autonomous Data Analysis Agent — Architecture Guide

---

## 📂 Project File Structure

```
data_analy/
├── .env                 # HuggingFace API token (secret)
├── requirements.txt     # Python dependencies
├── agent.py             # Core logic — LLM, agents, tools, EDA functions
├── app.py               # Streamlit UI — tabs, layout, user interaction
└── ARCHITECTURE.md      # This file
```

---

## 🏗 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       STREAMLIT UI  (app.py)                         │
│  ┌─────────┐ ┌────────┐ ┌──────────────┐ ┌──────┐ ┌──────┐ ┌─────┐ │
│  │ Preview │ │Deep EDA│ │Enterprise EDA│ │ Viz  │ │ Chat │ │Scrip│ │
│  └────┬────┘ └───┬────┘ └──────┬───────┘ └──┬───┘ └──┬───┘ └──┬──┘ │
└───────┼──────────┼─────────────┼────────────┼────────┼────────┼─────┘
        │          │             │            │        │        │
        ▼          ▼             ▼            ▼        ▼        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       AGENT LAYER  (agent.py)                        │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │              INTELLIGENCE LAYERS                               │   │
│  │  Dataset Profiler · Strategic Planner · Critic / Evaluator    │   │
│  │  Confidence Scorer · Self-Reflection · Column Matcher         │   │
│  │  Adaptive Viz · Enterprise EDA Pipeline                       │   │
│  └───────────────────────────────┬───────────────────────────────┘   │
│                                  │                                    │
│    ┌─────────────────────────────┼─────────────────────────┐         │
│    │              PLANNER (LLM)  │                          │         │
│    │         Routes to correct agent(s)                     │         │
│    └──────┬──────────────────────┼──────────────┬───────────┘         │
│           │                      │              │                     │
│    ┌──────▼──────┐  ┌────────────▼────────┐  ┌──▼──────────────┐     │
│    │  AGENT 1:   │  │  ENTERPRISE EDA:    │  │  AGENT 2:       │     │
│    │  Pandas     │  │  AI Plan → Execute  │  │  Visualization  │     │
│    │  Analysis   │  │  → Interpret        │  │  Agent          │     │
│    └──────┬──────┘  └────────────┬────────┘  └──┬──────────────┘     │
│           │                      │              │                     │
│     ┌─────▼──────────────────────▼──────────────▼─────┐              │
│     │         SAFE EXECUTION ENGINES                   │              │
│     │  execute_pandas_code()  |  execute_viz_code()    │              │
│     └─────────────────────┬───────────────────────────┘              │
│                           │                                           │
│     ┌─────────────────────▼───────────────────────────┐              │
│     │         SELF-HEALING ERROR CORRECTION            │              │
│     │    (LLM fixes errors → re-execute, up to 5x)    │              │
│     └─────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
        │                                          │
        ▼                                          ▼
┌──────────────────┐                  ┌──────────────────────────┐
│  Full DataFrame  │                  │  HuggingFace LLM         │
│  (in memory)     │                  │  (openai/gpt-oss-120b)   │
│  Never sent      │                  │  via ChatHuggingFace     │
│  to LLM          │                  │  Gets ONLY 25-row sample │
│                  │                  │  + metadata              │
└──────────────────┘                  └──────────────────────────┘
```

---

## 🔄 Complete Request Flow

### Chat Query Example: "What is the average salary by department?"

```
User types query
       │
       ▼
┌─────────────────────────────────┐
│ 1. MASTER PLANNER               │
│    Input: query + df metadata   │
│    Output: JSON plan            │
│    {                            │
│      needs_pandas: true,        │
│      needs_visualization: false,│
│      pandas_query: "...",       │
│      viz_query: ""              │
│    }                            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 2. PANDAS AGENT                 │
│    - Receives: query + 25-row   │
│      sample + column names +    │
│      dtypes + statistics        │
│    - LLM generates pandas code  │
│    - Code executed on FULL df   │
│    - Result stored in `result`  │
└──────────────┬──────────────────┘
               │
          ┌────┴────┐
          │ Success? │
          └────┬────┘
         Yes   │   No
          │    │    │
          │    │    ▼
          │    │  ┌───────────────────────┐
          │    │  │ ERROR CORRECTION      │
          │    │  │ - Capture error msg   │
          │    │  │ - Send code + error + │
          │    │  │   columns + dtypes    │
          │    │  │ - LLM generates fix   │
          │    │  │ - Re-execute          │
          │    │  │ - Repeat up to 5x     │
          │    │  └───────────┬───────────┘
          │    │              │
          ▼    ▼              ▼
┌─────────────────────────────────┐
│ 3. DISPLAY RESULTS              │
│    - Text output in chat        │
│    - Code in expander           │
│    - Logs in expander           │
└─────────────────────────────────┘
```

### Visualization Query Example: "Plot histogram of age"

Same flow, but planner sets `needs_visualization: true`, so after
the pandas agent (optional), the **Visualization Agent** runs:

```
VIZ AGENT
  │
  ├── LLM generates matplotlib/seaborn code
  ├── Code executed in sandboxed namespace
  ├── plt.gcf() captures the figure
  ├── If error → same retry loop (up to 5x)
  └── Figure rendered via st.pyplot()
```

---

## 🧩 Component Deep Dive

### 📄 agent.py — Core Modules

| Section | Lines | What It Does |
|---------|-------|-------------|
| **LLM Factory** | `create_llm()` | Creates HuggingFaceEndpoint via LangChain, loads API key from `.env` |
| **Context Builder** | `get_df_context()`, `format_context()` | Extracts 25-row sample + metadata (shape, dtypes, missing, stats). This is ALL the LLM ever sees. |
| **Code Helpers** | `extract_code()`, `clean_code()`, `extract_json()` | Parses LLM output — strips markdown fences, removes redundant imports |
| **Pandas Executor** | `execute_pandas_code()` | Runs generated code in sandboxed `exec()` with only `df, pd, np, stats` available |
| **Viz Executor** | `execute_viz_code()` | Same sandbox but with `plt, sns` added. Captures matplotlib figure. |
| **Prompt Templates** | 15 templates | `PANDAS_PROMPT`, `VIZ_PROMPT`, `ERROR_FIX_PROMPT`, `CRITIC_PROMPT`, `CONFIDENCE_PROMPT`, `SELF_REFLECTION_PROMPT`, `COLUMN_MATCH_PROMPT`, `ADAPTIVE_VIZ_PROMPT`, `DATASET_PROFILING_PROMPT`, `STRATEGIC_PLANNER_PROMPT`, `ENTERPRISE_PLAN_PROMPT`, `ENTERPRISE_STEP_PROMPT`, `ENTERPRISE_INTERPRETATION_PROMPT`, `FEATURE_PROMPT`, `EDA_SCRIPT_PROMPT` |
| **Agent 1: Pandas** | `setup_pandas_agent()` | LCEL chain + `Tool("PandasAnalysis")` + confidence + critic + reflection + retry loop |
| **Agent 2: Viz** | `setup_viz_agent()` | LCEL chain + `Tool("Visualization")` + adaptive viz + retry loop |
| **Master Planner** | `setup_planner()` | Strategic planner with profiling awareness — routes queries to agents |
| **Enterprise EDA** | `enterprise_eda()` | 100% AI-driven: LLM plans → generates code → executes → interprets |
| **EDA Functions** | `eda_overview()`, `eda_cleaning()`, `eda_summary()`, `eda_insights()` | Pure pandas — no LLM needed, fast |
| **LLM EDA** | `llm_feature_suggestions()`, `generate_eda_script()` | AI feature engineering ideas, downloadable script |
| **Auto Viz** | `auto_visualizations()` | Generates ALL standard plots (histogram, KDE, boxplot, violin, QQ, countplot, pie, scatter, heatmap, pairplot, missing values) without LLM |
| **Chat Orchestrator** | `handle_chat()` | Planner → Pandas Agent → Viz Agent → Critic → Reflection → merged result |

### 📄 app.py — Streamlit UI

| Section | What It Does |
|---------|-------------|
| **Sidebar** | File uploader, Agent init button, dataset quick info, column list |
| **Tab 1: Preview** | `df.head(n)`, column types table, `describe()` |
| **Tab 2: Deep EDA** | Runs all 4 EDA functions + LLM feature suggestions. Displays in expandable sections. |
| **Tab 3: Enterprise EDA** | 100% AI-driven: LLM plans → generates code → executes on full data → interprets. Shows plan, per-task results+code, AI interpretation. |
| **Tab 4: Visualizations** | Auto-generate all plots OR custom LLM-generated plots + AI viz recommendations |
| **Tab 5: Chat** | Full AI chat — planner → pandas/viz agents → critic → self-reflection → display results + code + logs |
| **Tab 6: Script** | LLM generates downloadable Python EDA script |

---

## 🔐 Safety & Sandboxing

```python
# Pandas execution namespace — ONLY these are available:
namespace = {
    "df": df.copy(),        # Copy, not reference
    "pd": pd,
    "np": np,
    "stats": scipy_stats,
}

# Viz execution namespace — adds plotting:
namespace = {
    "df": df.copy(),
    "pd": pd, "np": np,
    "plt": plt, "sns": sns,
    "stats": scipy_stats,
}

# ❌ No file system access
# ❌ No os module
# ❌ No subprocess
# ❌ No network access
# ❌ No __import__
```

---

## 🔁 Self-Healing Error Correction Loop

```
Attempt 1: Execute generated code
  │
  ├── ✅ Success → return result
  │
  └── ❌ Error captured
         │
         ▼
      Send to LLM:
        - Original code
        - Error message
        - Column names
        - Column dtypes
         │
         ▼
      LLM returns fixed code
         │
         ▼
Attempt 2: Execute fixed code
  │
  ├── ✅ Success → return result
  └── ❌ Error → repeat...
         │
         ▼
      ... up to 5 attempts total
```

**What gets auto-fixed:**
- Wrong column names → LLM corrects spelling
- Dtype mismatches → LLM adds `.astype()` / `pd.to_numeric()`
- Missing values → LLM adds `.dropna()` / `.fillna()`
- Syntax errors → LLM rewrites code
- Empty results → fallback `df.head(10)` added

---

## 🧠 LangChain Components Used

| LangChain Concept | Where Used |
|-------------------|-----------|
| `HuggingFaceEndpoint` | LLM backend (agent.py → `create_llm()`) |
| `ChatHuggingFace` | Chat wrapper for Groq-hosted models (chat completion API) |
| `PromptTemplate` | All 15 prompt templates |
| LCEL `prompt \| llm \| StrOutputParser()` | All chains — Pandas, Viz, Planner, Critic, Confidence, Reflection, Enterprise Plan/Step/Interpret, Feature, Script |
| `Tool` | `Tool("PandasAnalysis")`, `Tool("Visualization")` |
| `StrOutputParser` | Converts LLM output (including `AIMessage`) to plain strings |

---

## 📊 All Visualizations Generated

### Auto-Generated (no LLM, instant):
| Plot Type | For | Count |
|-----------|-----|-------|
| Histogram + KDE + Boxplot | Each numeric col (top 8) | up to 8 |
| Violin plot | Each numeric col (top 6) | up to 6 |
| QQ plot | Each numeric col (top 4) | up to 4 |
| Bar count plot | Each categorical col (top 6) | up to 6 |
| Pie chart | Low-cardinality categoricals (≤8 values) | up to 4 |
| Correlation heatmap | All numeric columns | 1 |
| Top-correlated scatter plots | Top 3 correlated pairs | up to 3 |
| Pairplot | If 2–5 numeric cols | 1 |
| Missing values heatmap | If any NaN exists | 1 |

### LLM-Generated (custom):
The Visualization Agent can generate **any** plot the user describes,
including: line plots, stacked bar charts, rolling mean, seasonal
decomposition suggestions, scatter matrices, etc.

---

## 💬 Chat Workflow Diagram

```
User: "Which department has highest average salary?"
  │
  ▼
PLANNER (LLMChain):
  "needs_pandas=true, needs_visualization=false"
  "pandas_query: group by department, compute mean salary, find max"
  │
  ▼
PANDAS AGENT (LLMChain → generates code):
  result = df.groupby('department')['salary'].mean().sort_values(ascending=False)
  │
  ▼
EXECUTE on full DataFrame (safe exec)
  │
  ▼
DISPLAY: table of departments + average salaries
  │
  ▼
User: "Now plot that as a bar chart"
  │
  ▼
PLANNER:
  "needs_pandas=true, needs_visualization=true"
  │
  ▼
PANDAS AGENT → computes data
VIZ AGENT → generates bar chart code → renders figure
  │
  ▼
DISPLAY: bar chart + code in expander
```

---

## ⚡ Performance Design

| Concern | Solution |
|---------|----------|
| Large datasets (millions of rows) | Only 25-row sample sent to LLM; all computation via pandas on full data |
| Wide datasets (100+ columns) | Sample truncated to 15 cols; describe truncated to 20 cols |
| Long string values | Truncated to 50 chars in sample |
| Memory | `df.copy()` in exec namespace; figures closed after use |
| Slow LLM | Separate auto-viz (no LLM) for instant plots |
| LLM errors | Up to 5 auto-retries with error context |
| Pairplot on big data | Sampled to 500 rows |
| Scatter on big data | Sampled to 2000 rows |

---

## 🏢 Enterprise EDA — 100% AI-Driven Pipeline

The Enterprise EDA uses the **exact same architecture** as the Chat system.
Zero hardcoded analysis logic — the LLM plans, generates code, and interprets everything.

```
User clicks "Run Enterprise EDA"
       │
       ▼
┌──────────────────────────────────────────────┐
│  STAGE 1: AI PLANS THE ANALYSIS              │
│                                              │
│  LLM receives: dataset metadata (25-row      │
│  sample + shape + dtypes + missing + stats)  │
│                                              │
│  LLM returns: JSON plan with 7-9 tasks       │
│  [                                           │
│    {id: "overview",    name: "Data Overview"} │
│    {id: "quality",     name: "Quality Audit"} │
│    {id: "statistics",  name: "Statistics"}    │
│    {id: "outliers",    name: "Outlier Detect"}│
│    {id: "correlation", name: "Correlations"}  │
│    {id: "categorical", name: "Cat Analysis"}  │
│    {id: "risk",        name: "Risk Assess"}   │
│  ]                                           │
│                                              │
│  Fallback: DEFAULT_ENTERPRISE_PLAN if parse  │
│  fails (still AI-executed, just default plan)│
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  STAGE 2: FOR EACH TASK — AI GENERATES CODE  │
│                                              │
│  For task_i in plan:                         │
│    │                                         │
│    ├─ LLM receives:                          │
│    │  • Task name + objective                │
│    │  • Dataset context (metadata)           │
│    │  • Previous task results (for context)  │
│    │                                         │
│    ├─ LLM generates: pandas code             │
│    │  (specific to this analysis task)       │
│    │                                         │
│    ├─ Code executed on FULL DataFrame        │
│    │  (sandboxed: df, pd, np, stats only)    │
│    │                                         │
│    ├─ If error:                              │
│    │  └─ Self-healing loop (up to 5 retries) │
│    │     LLM sees error → generates fix      │
│    │     → re-execute → repeat               │
│    │                                         │
│    └─ Result stored in all_results dict      │
│                                              │
│  Progress: 10% → 80% (per-task updates)      │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  STAGE 3: AI INTERPRETS ALL RESULTS          │
│                                              │
│  LLM receives: ALL computed results from     │
│  every task (real numbers, not estimates)     │
│                                              │
│  LLM returns: Professional 9-section report  │
│  • Data Overview (citing exact numbers)      │
│  • Data Quality Audit                        │
│  • Statistical Analysis                      │
│  • Outlier Analysis                          │
│  • Correlation Analysis                      │
│  • Categorical Analysis                      │
│  • Risk Assessment                           │
│  • Feature Engineering Opportunities         │
│  • Executive Summary (scores + top findings) │
│                                              │
│  Every claim references a computed number.    │
│  Zero hallucination — only real data.        │
└──────────────────────────────────────────────┘
```

### Key Design Decisions:
- **No hardcoded pandas logic** — LLM generates ALL analysis code
- **Self-healing** — if generated code fails, LLM auto-fixes (up to 5 retries per task)
- **Context chaining** — each task sees results from previous tasks
- **Fallback plan** — if LLM plan parsing fails, uses DEFAULT_ENTERPRISE_PLAN
- **Full proof** — raw computed data shown alongside AI interpretation

---

## 💬 Chat System — AI-Driven Orchestration

The Chat system demonstrates the full AI pipeline:
**User Query → Planner → Agent Selection → Code Generation → Execution → Critic → Reflection**

### Complete Flow Example: "give the box plot of age, are there any outliers?"

```
User types: "give the box plot of age, are there any outliers?"
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  1. STRATEGIC PLANNER (LLM)                         │
│                                                     │
│  Input: query + dataset metadata + profiling data   │
│  LLM decides:                                       │
│  {                                                  │
│    "analysis_strategy": "multi_stage",              │
│    "needs_pandas": true,                            │
│    "needs_visualization": true,                     │
│    "pandas_query": "Compute IQR bounds, list        │
│      outlier ages, count outliers",                 │
│    "viz_query": "Generate box plot of Age column",  │
│    "explanation": "We'll compute the IQR bounds     │
│      and generate a box plot"                       │
│  }                                                  │
└──────────────────────────┬──────────────────────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
┌──────────────────────┐  ┌──────────────────────────┐
│  2. PANDAS AGENT     │  │  3. VISUALIZATION AGENT  │
│                      │  │                          │
│  a. CONFIDENCE CHECK │  │  a. LLM generates code:  │
│     Score: 88/100    │  │     plt.figure(...)      │
│     "Safe to run"    │  │     plt.boxplot(...)     │
│                      │  │     plt.title(...)       │
│  b. LLM generates:   │  │                          │
│     Q1 = df['Age']   │  │  b. Execute in sandbox   │
│       .quantile(0.25)│  │     (df, pd, plt, sns)   │
│     Q3 = ...         │  │                          │
│     IQR = Q3 - Q1    │  │  c. If error → auto-fix  │
│     result = {...}   │  │     (up to 5 retries)    │
│                      │  │                          │
│  c. Execute on FULL  │  │  d. Capture figure via   │
│     df (sandbox)     │  │     plt.gcf()            │
│                      │  │                          │
│  d. If error:        │  │  Result:                 │
│     → KeyError?      │  │   ✅ Box plot rendered    │
│     → Column Match!  │  └──────────────────────────┘
│     → LLM fixes code │
│     → Retry (5x max) │
│                      │
│  Result:             │
│  {Q1: 35, Q3: 68,   │
│   IQR: 33,          │
│   outlier_count: 0}  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  4. CRITIC EVALUATION (LLM)                          │
│                                                      │
│  Input: query + code + output + dataset context      │
│  LLM judges quality:                                 │
│  {                                                   │
│    "is_sufficient": true,                            │
│    "reason": "Correctly computes Q1, Q3, IQR...",    │
│    "missing_aspects": [],                            │
│    "improvement_suggestion": "Add mean, median..."   │
│  }                                                   │
│                                                      │
│  If insufficient → runs improvement query            │
└──────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  5. SELF-REFLECTION (LLM)                            │
│                                                      │
│  LLM evaluates completeness:                         │
│  {                                                   │
│    "analysis_complete": false,                       │
│    "missed_opportunities": ["central tendency...",   │
│      "distribution shape...", "data quality..."],    │
│    "next_step_suggestion": "Add histogram, stats",   │
│    "should_visualize_more": true                     │
│  }                                                   │
│                                                      │
│  If should_visualize_more → auto-triggers viz agent  │
└──────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  6. DISPLAY TO USER                                  │
│                                                      │
│  📊 Result: {Q1: 35, Q3: 68, IQR: 33, outliers: 0}  │
│  📈 Box plot rendered in UI                           │
│  🎯 Critic: Sufficient ✅                             │
│  🔄 Reflection: Missing deeper analysis              │
│  📝 Code shown in expander                            │
│  📋 Full execution logs in expander                   │
└──────────────────────────────────────────────────────┘
```

### Intelligence Layers in Chat:

| Layer | Purpose | When Applied |
|-------|---------|-------------|
| **Confidence Scorer** | Pre-execution risk check (0-100) | Before running generated code |
| **Semantic Column Matcher** | Fixes column name typos via LLM | On KeyError during execution |
| **Critic / Evaluator** | Judges if output answers the question | After successful execution |
| **Self-Reflection** | Identifies missed analysis opportunities | After critic evaluation |
| **Auto-Improvement** | Re-runs query with critic's suggestion | If critic says "insufficient" |
| **Auto-Viz Trigger** | Triggers visualization agent | If reflection says "should_visualize_more" |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your HuggingFace API token in .env
#    HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxx

# 3. Launch the app
streamlit run app.py

# 4. Open browser → http://localhost:8501
# 5. Upload CSV → Click "Initialize Agent" → Explore!
```

---

## 🔧 Configuration

Edit these in `agent.py` (top of file):

```python
MAX_RETRIES = 5           # Error correction attempts
SAMPLE_ROWS = 25          # Rows sent to LLM
MAX_SAMPLE_COLS = 15      # Max columns in sample
DEFAULT_MODEL = "..."     # HuggingFace model repo ID
```

---

*Built with LangChain + HuggingFace + Streamlit + Pandas + Matplotlib + Seaborn*
