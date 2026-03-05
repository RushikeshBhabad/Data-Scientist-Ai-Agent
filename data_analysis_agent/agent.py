"""
agent.py — Autonomous Data Analysis Agent (Production Grade)
=============================================================
Two LangChain agents + Critic + Planner + Profiler + Self-Reflection.

Agents:
  1. Pandas Analysis Agent   — code gen, execution, stats
  2. Visualization Agent     — matplotlib / seaborn charts

Intelligence Layers:
  - Dataset Profiler          — one-time complexity scoring
  - Strategic Planner         — multi-step reasoning
  - Critic / Evaluator        — judges output quality
  - Confidence Scorer         — pre-execution risk check
  - Self-Reflection           — post-execution gap analysis
  - Semantic Column Matcher   — fuzzy column resolution
  - Adaptive Viz Intelligence — data-aware chart selection
  - Enterprise EDA            — Fortune-500-grade deep analysis
"""

import os, re, json, traceback, warnings
from io import StringIO
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from dotenv import load_dotenv

# --- LangChain ---
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser

# Optional imports for different LLM providers
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
except ImportError:
    try:
        from langchain_community.llms import HuggingFaceEndpoint
    except ImportError:
        HuggingFaceEndpoint = None
    ChatHuggingFace = None

warnings.filterwarnings("ignore")
load_dotenv()


# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
MAX_RETRIES = 5
SAMPLE_ROWS = 25
MAX_SAMPLE_COLS = 15
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Model registry:
#   openai/gpt-oss-120b              (Groq-hosted, chat-only)
#   Qwen/Qwen2.5-Coder-32B-Instruct (best for code)
#   mistralai/Mixtral-8x7B-Instruct-v0.1
#   meta-llama/Meta-Llama-3-8B-Instruct


# ════════════════════════════════════════════════════════════════
# 1. LLM FACTORY  (auto-detects provider task type)
# ════════════════════════════════════════════════════════════════

def create_llm(temperature=0.2, max_tokens=3072, model_id=None):
    """Create LLM instance. Tries Groq first (fast), then HuggingFace.

    Strategy:
      1. Try Groq (ChatGroq) — fast, reliable, uses GROQ_API_KEY.
      2. Try HuggingFace text-generation endpoint.
      3. Try ChatHuggingFace for chat-only providers.
    """
    errors = []
    temp = max(temperature, 0.01)

    # --- Attempt 1: Groq (preferred — fast and reliable) ---
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and ChatGroq is not None:
        try:
            groq_model = model_id or "llama-3.3-70b-versatile"
            llm = ChatGroq(
                temperature=temp,
                groq_api_key=groq_key,
                model_name=groq_model,
                max_tokens=max_tokens,
            )
            llm.invoke("Say OK")
            return llm
        except Exception as e:
            errors.append(f"Groq ({groq_model}): {e}")

    # --- Attempt 2: HuggingFace text-generation ---
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if api_key and HuggingFaceEndpoint is not None:
        repo = model_id or DEFAULT_MODEL
        try:
            llm = HuggingFaceEndpoint(
                repo_id=repo,
                task="text-generation",
                max_new_tokens=max_tokens,
                temperature=temp,
                huggingfacehub_api_token=api_key,
            )
            llm.invoke("Say OK")
            return llm
        except Exception as e:
            errors.append(f"HF text-generation: {e}")

        # --- Attempt 3: ChatHuggingFace (for chat-only providers) ---
        if ChatHuggingFace is not None:
            try:
                endpoint = HuggingFaceEndpoint(
                    repo_id=repo,
                    task="conversational",
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    huggingfacehub_api_token=api_key,
                )
                llm = ChatHuggingFace(llm=endpoint)
                llm.invoke("Say OK")
                return llm
            except Exception as e:
                errors.append(f"ChatHuggingFace: {e}")

    raise ValueError(
        "No LLM could be loaded. Set GROQ_API_KEY or HUGGINGFACEHUB_API_TOKEN in .env\n"
        + "\n".join(errors)
    )


# ════════════════════════════════════════════════════════════════
# 2. DATAFRAME CONTEXT BUILDER  (never sends full data)
# ════════════════════════════════════════════════════════════════

def get_df_context(df):
    """Build metadata + sample — safe for LLM consumption."""
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    date_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)

    sample = df.head(SAMPLE_ROWS).copy()
    if len(sample.columns) > MAX_SAMPLE_COLS:
        sample = sample.iloc[:, :MAX_SAMPLE_COLS]
    for col in sample.select_dtypes(include=["object"]).columns:
        sample[col] = sample[col].astype(str).str[:50]

    desc = df.describe(include="all")
    if len(desc.columns) > 20:
        desc = desc.iloc[:, :20]

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "datetime_cols": date_cols,
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "nunique": df.nunique().to_dict(),
        "sample_str": sample.to_string(),
        "describe_str": desc.to_string(),
    }


def format_context(ctx):
    """Render context dict to a compact string."""
    return "\n".join([
        f"Shape: {ctx['shape'][0]} rows x {ctx['shape'][1]} columns",
        f"Columns: {ctx['columns']}",
        f"Dtypes: {json.dumps(ctx['dtypes'], indent=2)}",
        f"Numeric: {ctx['numeric_cols']}",
        f"Categorical: {ctx['categorical_cols']}",
        f"Datetime: {ctx['datetime_cols']}",
        f"Missing: {json.dumps(ctx['missing'])}",
        f"Missing%: {json.dumps(ctx['missing_pct'])}",
        f"Unique: {json.dumps(ctx['nunique'])}",
        f"\nSample ({SAMPLE_ROWS} rows):\n{ctx['sample_str']}",
        f"\nDescribe:\n{ctx['describe_str']}",
    ])


# ════════════════════════════════════════════════════════════════
# 3. HELPERS  (extract code / json, chain runner)
# ════════════════════════════════════════════════════════════════

def extract_code(text):
    """Pull Python code from LLM response."""
    if not text or not text.strip():
        return ""
    for pat in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        m = re.findall(pat, text, re.DOTALL)
        if m:
            return m[0].strip()
    lines, started = [], False
    for line in text.strip().split("\n"):
        s = line.strip()
        if s.startswith("```"):
            started = not started
            continue
        if not started and not lines:
            if s and s[0].isupper() and not any(k in s for k in
                    ["import", "from ", "result", "df", "plt", "sns", "pd.", "np."]):
                continue
        lines.append(line)
        started = True
    return "\n".join(lines).strip() if lines else text.strip()


def clean_code(code):
    """Remove import / read_csv lines (already in namespace)."""
    out = []
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            continue
        if "read_csv" in s or "read_excel" in s:
            continue
        out.append(line)
    return "\n".join(out)


def extract_json(text):
    """Extract JSON object from LLM text."""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    for pat in [r"```json\s*\n(.*?)```", r"(\{[\s\S]*?\})"]:
        for m in re.findall(pat, text, re.DOTALL):
            try:
                return json.loads(m.strip())
            except Exception:
                continue
    return None


def make_chain(prompt, llm):
    """LCEL chain: prompt | llm | StrOutputParser."""
    return prompt | llm | StrOutputParser()


def run_chain(chain, **kwargs):
    """Invoke chain, always return string."""
    try:
        result = chain.invoke(kwargs)
    except Exception as e:
        return f"LLM_ERROR: {e}"
    if isinstance(result, dict):
        return str(result.get("text", result))
    if hasattr(result, "content"):
        return result.content
    return str(result)


# ════════════════════════════════════════════════════════════════
# 4. SAFE EXECUTION ENGINES
# ════════════════════════════════════════════════════════════════

def execute_pandas_code(code, df):
    """Run pandas code in sandboxed namespace."""
    ns = {"__builtins__": __builtins__, "df": df.copy(),
          "pd": pd, "np": np, "stats": scipy_stats}
    buf = StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, ns)
        printed = buf.getvalue()
        result = ns.get("result", None)
        if result is not None:
            if isinstance(result, (pd.DataFrame, pd.Series)):
                return {"success": True, "output": result.to_string(max_rows=200),
                        "data": result, "type": "dataframe"}
            return {"success": True, "output": str(result), "data": result, "type": "value"}
        if printed.strip():
            return {"success": True, "output": printed.strip(), "data": None, "type": "print"}
        return {"success": True, "output": "Code executed (no output).", "data": None, "type": "none"}
    except Exception as e:
        return {"success": False, "output": "", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def execute_viz_code(code, df):
    """Run matplotlib/seaborn code. Returns figure or error."""
    plt.close("all")
    ns = {"__builtins__": __builtins__, "df": df.copy(),
          "pd": pd, "np": np, "plt": plt, "sns": sns, "stats": scipy_stats}
    try:
        exec(code, ns)
        fig = plt.gcf()
        if fig.get_axes():
            plt.tight_layout()
            return {"success": True, "figure": fig, "type": "plot"}
        plt.close(fig)
        return {"success": False, "error": "No plot created."}
    except Exception as e:
        plt.close("all")
        return {"success": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


# ════════════════════════════════════════════════════════════════
# 5. ALL PROMPT TEMPLATES
# ════════════════════════════════════════════════════════════════

# --- Core Agent Prompts ---

PANDAS_PROMPT = PromptTemplate(
    input_variables=["df_context", "query"],
    template="""You are an expert Python data analyst.
Write pandas code to answer the query.

RULES:
- DataFrame is `df`. `pd`, `np`, `stats` (scipy.stats) are available.
- Do NOT import anything. Do NOT redefine df.
- Store final answer in variable `result`.
- Handle NaN with dropna()/fillna().
- Output ONLY executable Python code. No markdown.

DataFrame Info:
{df_context}

Query: {query}

Python code:"""
)

VIZ_PROMPT = PromptTemplate(
    input_variables=["df_context", "query"],
    template="""You are an expert data visualization engineer.
Generate matplotlib/seaborn code.

RULES:
- `df`, `plt`, `sns`, `pd`, `np` are available. Do NOT import.
- Start with: plt.figure(figsize=(10, 6))
- Set title, xlabel, ylabel.
- Use dropna() before plotting.
- For high-cardinality categoricals, plot top 15 only.
- Use color palettes like 'viridis', 'Set2'.
- Output ONLY executable Python code. No markdown.

DataFrame Info:
{df_context}

Plot request: {query}

Python code:"""
)

ERROR_FIX_PROMPT = PromptTemplate(
    input_variables=["original_code", "error_message", "columns", "dtypes"],
    template="""Fix this Python code that produced an error.

Code:
```python
{original_code}
```

Error: {error_message}
Available columns: {columns}
Dtypes: {dtypes}

RULES:
- Return ONLY corrected Python code. No markdown.
- Use correct column names from the list.
- Handle dtype mismatches.

Fixed code:"""
)

# --- Intelligence Layer Prompts ---

CRITIC_PROMPT = PromptTemplate(
    input_variables=["query", "code", "output", "df_context"],
    template="""You are a senior data analysis evaluator.
Judge whether the analysis truly answers the user's question.

User Question: {query}

Generated Code:
```python
{code}
```

Output: {output}

Dataset Metadata:
{df_context}

Return ONLY valid JSON:
{{
  "is_sufficient": true or false,
  "reason": "short explanation",
  "missing_aspects": ["aspect1", "aspect2"],
  "improvement_suggestion": "what should be done next",
  "requires_visualization": true or false
}}

JSON:"""
)

CONFIDENCE_PROMPT = PromptTemplate(
    input_variables=["query", "code", "df_context"],
    template="""Evaluate this generated pandas code BEFORE execution.

User Question: {query}

Generated Code:
```python
{code}
```

Dataset Context:
{df_context}

Return ONLY valid JSON:
{{
  "confidence_score": 0 to 100,
  "risk_factors": ["factor1", "factor2"],
  "should_refine_before_execution": true or false
}}

JSON:"""
)

SELF_REFLECTION_PROMPT = PromptTemplate(
    input_variables=["query", "code", "output"],
    template="""Reflect on the analysis result.

User Question: {query}
Code: {code}
Result: {output}

Did we fully answer the question? Are there deeper insights?

Return ONLY valid JSON:
{{
  "analysis_complete": true or false,
  "missed_opportunities": ["item1", "item2"],
  "next_step_suggestion": "what to do next",
  "should_visualize_more": true or false
}}

JSON:"""
)

COLUMN_MATCH_PROMPT = PromptTemplate(
    input_variables=["user_column", "columns"],
    template="""You are a data schema matching expert.

User referenced column: "{user_column}"
Available columns: {columns}

Choose the best semantic match.

Return ONLY valid JSON:
{{
  "matched_column": "exact_column_name_from_list",
  "confidence": 0 to 100,
  "reason": "why this column matches"
}}

JSON:"""
)

ADAPTIVE_VIZ_PROMPT = PromptTemplate(
    input_variables=["query", "df_context"],
    template="""You are an expert visualization strategist.

User Request: {query}
Dataset Metadata:
{df_context}

Choose best visualization types based on data types and size.
Avoid overcrowded charts. Limit high-cardinality to top categories.
If time column exists, consider time-series.
If strong correlations, suggest scatter with regression.

Return ONLY valid JSON:
{{
  "recommended_plots": [
    {{
      "type": "histogram | scatter | heatmap | boxplot | violin | line | bar | pairplot | qqplot | pie",
      "columns": ["col1", "col2"],
      "reason": "why appropriate"
    }}
  ],
  "plot_priority_order": ["plot1", "plot2"]
}}

JSON:"""
)

DATASET_PROFILING_PROMPT = PromptTemplate(
    input_variables=["df_context"],
    template="""You are a dataset profiler. Analyze complexity and quality.

Dataset Context:
{df_context}

Return ONLY valid JSON:
{{
  "complexity_level": "low | medium | high",
  "data_quality_score": 0 to 100,
  "major_issues": ["issue1", "issue2"],
  "risk_areas": ["imbalance", "high_missing", "high_cardinality", "skewed_distribution"],
  "analysis_recommendations": ["rec1", "rec2"]
}}

JSON:"""
)

STRATEGIC_PLANNER_PROMPT = PromptTemplate(
    input_variables=["query", "df_context", "profiling_summary"],
    template="""You are a senior AI data strategist.

User Question: {query}
Dataset Context:
{df_context}

Dataset Profiling:
{profiling_summary}

Return ONLY valid JSON:
{{
  "analysis_strategy": "multi_stage | quick_answer | deep_exploration",
  "stages": [
    {{
      "stage": 1,
      "objective": "what we want to learn",
      "tool": "pandas | visualization",
      "expected_insight": "expected outcome"
    }}
  ],
  "should_iterate": true or false,
  "needs_pandas": true or false,
  "needs_visualization": true or false,
  "pandas_query": "task for pandas agent",
  "viz_query": "task for viz agent",
  "explanation": "one sentence for user"
}}

JSON:"""
)

ENTERPRISE_PLAN_PROMPT = PromptTemplate(
    input_variables=["df_context"],
    template="""You are a Principal Data Scientist at a Fortune-500 company.
Plan a comprehensive Enterprise-grade EDA for this dataset.

Dataset Context:
{df_context}

Create a detailed multi-step analysis plan. Think about what a $300K/year data
science director would analyze before putting data into production.

Return ONLY valid JSON:
{{
  "tasks": [
    {{
      "id": "unique_id",
      "name": "Human-readable task name",
      "objective": "Detailed description of EXACTLY what to compute — specific metrics, formulas, thresholds",
      "priority": 1
    }}
  ]
}}

You MUST include tasks that cover ALL of these analysis areas:

1. DATA OVERVIEW — row count, column count, memory usage (MB), duplicate row count,
   data type summary, column listing with types
2. DATA QUALITY AUDIT — missing value count & percentage per column, total missing cells,
   completeness %, placeholder detection in object columns (scan for ?, NA, N/A, na, -,
   --, null, none, unknown, empty string)
3. STATISTICAL ANALYSIS — describe() for numerics (mean, std, min, 25%, 50%, 75%, max),
   skewness, kurtosis, variance, median per numeric column
4. OUTLIER DETECTION — IQR method: Q1, Q3, IQR, lower/upper bounds (1.5×IQR), count &
   percentage of outliers per numeric column. Z-score method: count values with |z|>3
   per numeric column
5. CORRELATION ANALYSIS — full correlation matrix for numeric columns, identify ALL pairs
   with |correlation| > 0.8, rank by absolute correlation
6. CATEGORICAL ANALYSIS — cardinality (nunique) per categorical column, top 10 value
   counts per column, class imbalance ratio (min_count/max_count) for columns with
   2-20 unique values
7. RISK ASSESSMENT — potential data leakage columns (nunique > 90% of row count),
   constant columns (nunique ≤ 1), low-variance numeric columns (variance < 0.01),
   columns that could cause ML pipeline issues
8. DISTRIBUTION ANALYSIS — normality assessment based on skewness/kurtosis,
   identify heavily skewed columns (|skew| > 1), identify heavy-tailed distributions
   (|kurtosis| > 3)
9. FEATURE ENGINEERING OPPORTUNITIES — suggest transformations based on detected
   patterns (log transform for skewed, interaction terms for correlated pairs,
   encoding strategies for categoricals)

Be SPECIFIC in each objective. State exactly what metrics and thresholds to compute.
Order by priority (1 = most important, compute first).

JSON:"""
)

ENTERPRISE_STEP_PROMPT = PromptTemplate(
    input_variables=["task_name", "task_objective", "df_context", "previous_results"],
    template="""You are a Principal Data Scientist executing one step of an Enterprise EDA.

Current Task: {task_name}
Objective: {task_objective}

Dataset Context:
{df_context}

Results from previous analysis steps (use these for context, do NOT recompute):
{previous_results}

Generate pandas code that performs this SPECIFIC analysis task on the FULL dataset.
Compute EVERYTHING described in the objective above. Be thorough — this is production work.

STRICT RULES:
- df is already loaded. `pd`, `np`, `stats` (scipy.stats) are available.
- Do NOT import anything. Do NOT redefine df.
- Compute on the FULL dataset. NOT samples.
- Handle NaN with dropna() where needed.
- Store ALL results in a single dict variable called `result`.
- Make values JSON-serializable: use int(), float(), str(), .tolist(), .to_dict().
- Do NOT use print(). Store everything in `result`.
- Output ONLY executable Python code. No markdown. No explanations.

Python code:"""
)

ENTERPRISE_INTERPRETATION_PROMPT = PromptTemplate(
    input_variables=["execution_result"],
    template="""You are a Senior Data Science Director reviewing computed EDA results.

Below are the ACTUAL computed outputs from running AI-generated pandas code on the
FULL dataset. Every number is REAL — computed, not estimated.

{execution_result}

Interpret this data professionally. Do NOT guess. Use ONLY the numbers provided.
If a task failed, note it and work with what succeeded.

Provide a detailed executive report covering:

SECTION 1: DATA OVERVIEW
- Exact rows, columns, memory, duplicates (cite the numbers)
- Column classification breakdown
- Structural observations and data schema assessment

SECTION 2: DATA QUALITY AUDIT
- Missing values: cite exact counts and percentages from computed results
- Duplicates: exact count and impact assessment
- Placeholders found (if any)
- Data Quality Score (0-100) — justify with specific metrics

SECTION 3: STATISTICAL ANALYSIS
- Distribution shape: cite exact skewness and kurtosis values
- Central tendency and spread analysis with exact numbers
- Identify columns that need transformation (cite skewness values)

SECTION 4: OUTLIER ANALYSIS
- IQR outliers: cite exact counts, percentages, and bounds per column
- Z-score outliers: cite exact counts per column
- Assessment of outlier severity and recommended handling

SECTION 5: CORRELATION ANALYSIS
- High-correlation pairs with exact coefficients
- Multicollinearity risk assessment
- Feature selection implications

SECTION 6: CATEGORICAL ANALYSIS
- Cardinality breakdown with exact unique counts
- Class imbalance with exact ratios (cite min/max classes)
- Encoding strategy recommendations based on actual cardinality

SECTION 7: RISK ASSESSMENT
- Data leakage columns with evidence (cite cardinality vs row count)
- Constant/low-variance columns with exact values
- Production readiness risks

SECTION 8: FEATURE ENGINEERING OPPORTUNITIES
- Log/power transforms for skewed columns (cite skewness)
- Interaction terms for correlated pairs (cite correlations)
- Encoding recommendations based on cardinality data

SECTION 9: EXECUTIVE SUMMARY
- Top 5 critical findings (with exact numbers as evidence)
- Top 5 data risks (with evidence from computed metrics)
- Top 5 recommended actions (prioritized)
- Overall Production Readiness Score (0-100) with justification

Be analytical and precise. Every claim MUST reference a specific number from the
computed results. No assumptions. No generic statements. This is a Fortune-500 report."""
)

FEATURE_PROMPT = PromptTemplate(
    input_variables=["df_context"],
    template="""You are a feature engineering expert. Suggest features.

DataFrame Info:
{df_context}

Return ONLY valid JSON:
{{
  "feature_suggestions": [
    {{"name": "feat", "description": "what it does", "code": "pandas code"}}
  ],
  "transformations": ["suggestion1"],
  "scaling_recommendations": ["rec1"],
  "dimensionality_notes": "notes"
}}

JSON:"""
)

EDA_SCRIPT_PROMPT = PromptTemplate(
    input_variables=["df_context"],
    template="""Generate a complete Python EDA script for this dataset.

DataFrame Info:
{df_context}

RULES:
- Jupyter-compatible, all imports at top.
- df = pd.read_csv("your_data.csv") placeholder.
- Include: shape, info, describe, missing, duplicates.
- Include: distribution + count plots, correlation heatmap.
- Include: outlier detection (IQR), skewness analysis.
- Well-commented, clean code.
- Output ONLY Python code:"""
)


# ════════════════════════════════════════════════════════════════
# 6. AGENT 1 — PANDAS ANALYSIS AGENT
# ════════════════════════════════════════════════════════════════

def setup_pandas_agent(llm):
    """Create Pandas Agent with confidence check + critic + self-reflection."""
    code_chain = make_chain(PANDAS_PROMPT, llm)
    fix_chain = make_chain(ERROR_FIX_PROMPT, llm)
    confidence_chain = make_chain(CONFIDENCE_PROMPT, llm)
    critic_chain = make_chain(CRITIC_PROMPT, llm)
    reflect_chain = make_chain(SELF_REFLECTION_PROMPT, llm)
    column_chain = make_chain(COLUMN_MATCH_PROMPT, llm)

    def match_column(user_col, columns):
        """Semantic column matching when exact name not found."""
        raw = run_chain(column_chain,
                        user_column=user_col, columns=str(columns))
        parsed = extract_json(raw)
        if parsed and parsed.get("confidence", 0) >= 60:
            return parsed.get("matched_column", user_col)
        return user_col

    def run_pandas(query, df):
        ctx = get_df_context(df)
        ctx_str = format_context(ctx)
        logs = []

        # Step 1: Generate code
        raw = run_chain(code_chain, df_context=ctx_str, query=query)
        if raw.startswith("LLM_ERROR"):
            return {"success": False, "error": raw, "code": "", "logs": [raw],
                    "critic": None, "reflection": None}

        code = clean_code(extract_code(raw))
        if not code.strip():
            return {"success": False, "error": "LLM returned empty code.",
                    "code": "", "logs": ["Empty response."],
                    "critic": None, "reflection": None}
        logs.append(f"[Pandas] Generated code:\n{code}")

        # Step 2: Confidence check (skip if LLM slow)
        confidence = None
        try:
            conf_raw = run_chain(confidence_chain,
                                 query=query, code=code, df_context=ctx_str)
            confidence = extract_json(conf_raw)
            if confidence:
                score = confidence.get("confidence_score", 100)
                logs.append(f"[Confidence] Score: {score}/100")
                if confidence.get("should_refine_before_execution") and score < 50:
                    logs.append("[Confidence] Low score — regenerating code...")
                    raw2 = run_chain(code_chain, df_context=ctx_str,
                                     query=query + " (be extra careful with column names and dtypes)")
                    code2 = clean_code(extract_code(raw2))
                    if code2.strip():
                        code = code2
                        logs.append(f"[Pandas] Regenerated code:\n{code}")
        except Exception:
            pass

        # Step 3: Execute with retry loop
        for attempt in range(MAX_RETRIES):
            result = execute_pandas_code(code, df)

            if result["success"]:
                if result["type"] == "none" and "result" not in code:
                    code += "\nresult = df.head(10)"
                    logs.append(f"Attempt {attempt+1}: no result — adding fallback.")
                    continue
                logs.append(f"Attempt {attempt+1}: ✅ Success")
                result["code"] = code
                result["logs"] = logs

                # Step 4: Critic evaluation
                critic = None
                try:
                    crit_raw = run_chain(critic_chain,
                                         query=query, code=code,
                                         output=str(result["output"])[:500],
                                         df_context=ctx_str)
                    critic = extract_json(crit_raw)
                    if critic:
                        logs.append(f"[Critic] Sufficient: {critic.get('is_sufficient')}")
                except Exception:
                    pass
                result["critic"] = critic

                # Step 5: Self-reflection
                reflection = None
                try:
                    ref_raw = run_chain(reflect_chain,
                                        query=query, code=code,
                                        output=str(result["output"])[:500])
                    reflection = extract_json(ref_raw)
                    if reflection:
                        logs.append(f"[Reflection] Complete: {reflection.get('analysis_complete')}")
                except Exception:
                    pass
                result["reflection"] = reflection

                return result

            # Step 3b: Auto-fix error
            err = result.get("error", "Unknown")
            logs.append(f"Attempt {attempt+1}: ❌ {err}")

            # Check for column name mismatch — try semantic match
            if "KeyError" in err:
                import re as _re
                col_match = _re.search(r"KeyError:\s*['\"](.+?)['\"]", err)
                if col_match:
                    bad_col = col_match.group(1)
                    good_col = match_column(bad_col, ctx["columns"])
                    if good_col != bad_col:
                        code = code.replace(f"'{bad_col}'", f"'{good_col}'")
                        code = code.replace(f'"{bad_col}"', f'"{good_col}"')
                        logs.append(f"[Column Match] '{bad_col}' → '{good_col}'")
                        continue

            if attempt < MAX_RETRIES - 1:
                fix_raw = run_chain(fix_chain,
                                    original_code=code, error_message=err,
                                    columns=str(ctx["columns"]),
                                    dtypes=json.dumps(ctx["dtypes"]))
                code = clean_code(extract_code(fix_raw))
                logs.append(f"Auto-fix #{attempt+1}:\n{code}")

        return {"success": False, "error": f"Failed after {MAX_RETRIES} retries.",
                "code": code, "logs": logs, "critic": None, "reflection": None}

    pandas_tool = Tool(
        name="PandasAnalysis",
        func=lambda q: json.dumps(
            {k: v for k, v in run_pandas(q, pd.DataFrame()).items()
             if k not in ("data", "critic", "reflection")}, default=str),
        description="Execute pandas analysis. Input: natural language query.",
    )
    return run_pandas, pandas_tool


# ════════════════════════════════════════════════════════════════
# 7. AGENT 2 — VISUALIZATION AGENT
# ════════════════════════════════════════════════════════════════

def setup_viz_agent(llm):
    """Create Visualization Agent with adaptive intelligence."""
    viz_chain = make_chain(VIZ_PROMPT, llm)
    fix_chain = make_chain(ERROR_FIX_PROMPT, llm)
    adaptive_chain = make_chain(ADAPTIVE_VIZ_PROMPT, llm)

    def get_viz_recommendations(query, df):
        """Get AI-recommended visualizations."""
        ctx_str = format_context(get_df_context(df))
        raw = run_chain(adaptive_chain, query=query, df_context=ctx_str)
        return extract_json(raw)

    def run_viz(query, df):
        ctx = get_df_context(df)
        ctx_str = format_context(ctx)
        logs = []

        raw = run_chain(viz_chain, df_context=ctx_str, query=query)
        if raw.startswith("LLM_ERROR"):
            return {"success": False, "error": raw, "code": "", "logs": [raw]}

        code = clean_code(extract_code(raw))
        if not code.strip():
            return {"success": False, "error": "LLM returned empty viz code.",
                    "code": "", "logs": ["Empty response."]}
        logs.append(f"[Viz] Generated code:\n{code}")

        for attempt in range(MAX_RETRIES):
            result = execute_viz_code(code, df)
            if result["success"]:
                logs.append(f"Attempt {attempt+1}: ✅ Plot created")
                result["code"] = code
                result["logs"] = logs
                return result

            err = result.get("error", "Unknown")
            logs.append(f"Attempt {attempt+1}: ❌ {err}")
            if attempt < MAX_RETRIES - 1:
                fix_raw = run_chain(fix_chain,
                                    original_code=code, error_message=err,
                                    columns=str(ctx["columns"]),
                                    dtypes=json.dumps(ctx["dtypes"]))
                code = clean_code(extract_code(fix_raw))
                logs.append(f"Auto-fix #{attempt+1}:\n{code}")

        return {"success": False, "error": f"Viz failed after {MAX_RETRIES} retries.",
                "code": code, "logs": logs}

    viz_tool = Tool(
        name="Visualization",
        func=lambda q: json.dumps(
            {k: v for k, v in run_viz(q, pd.DataFrame()).items()
             if k != "figure"}, default=str),
        description="Generate visualizations. Input: natural language description.",
    )
    return run_viz, viz_tool, get_viz_recommendations


# ════════════════════════════════════════════════════════════════
# 8. DATASET PROFILER  (runs once after upload)
# ════════════════════════════════════════════════════════════════

def profile_dataset(llm, df):
    """One-time dataset profiling for smarter planning."""
    ctx_str = format_context(get_df_context(df))
    chain = make_chain(DATASET_PROFILING_PROMPT, llm)
    raw = run_chain(chain, df_context=ctx_str)
    result = extract_json(raw)
    if result is None:
        result = {
            "complexity_level": "medium",
            "data_quality_score": 50,
            "major_issues": [],
            "risk_areas": [],
            "analysis_recommendations": [],
        }
    return result


# ════════════════════════════════════════════════════════════════
# 9. STRATEGIC PLANNER  (upgraded with profiling context)
# ════════════════════════════════════════════════════════════════

def setup_planner(llm):
    """Strategic planner with dataset profiling awareness."""
    plan_chain = make_chain(STRATEGIC_PLANNER_PROMPT, llm)

    def plan(query, df, profiling=None):
        ctx_str = format_context(get_df_context(df))
        prof_str = json.dumps(profiling, indent=2) if profiling else "Not available"
        raw = run_chain(plan_chain,
                        query=query, df_context=ctx_str,
                        profiling_summary=prof_str)
        parsed = extract_json(raw)

        if parsed is None:
            q = query.lower()
            needs_viz = any(w in q for w in [
                "plot", "chart", "graph", "visual", "histogram", "scatter",
                "heatmap", "bar", "pie", "boxplot", "violin", "distribution",
                "show", "draw", "display",
            ])
            parsed = {
                "analysis_strategy": "quick_answer",
                "stages": [],
                "needs_pandas": True,
                "needs_visualization": needs_viz,
                "pandas_query": query,
                "viz_query": query if needs_viz else "",
                "explanation": "Processing your request...",
            }
        # Ensure required keys exist
        parsed.setdefault("needs_pandas", True)
        parsed.setdefault("needs_visualization", False)
        parsed.setdefault("pandas_query", query)
        parsed.setdefault("viz_query", query)
        parsed.setdefault("explanation", "Analyzing...")
        return parsed

    return plan


# ════════════════════════════════════════════════════════════════
# 10. EDA FUNCTIONS  (pure pandas — fast, no LLM)
# ════════════════════════════════════════════════════════════════

def eda_overview(df):
    """Basic dataset overview."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(10),
        "memory": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        "duplicates": int(df.duplicated().sum()),
    }


def eda_cleaning(df):
    """Data cleaning report."""
    missing = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    report = {
        "missing_values": missing[missing > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "complete_rows": int(df.dropna().shape[0]),
        "complete_pct": round(df.dropna().shape[0] / max(len(df), 1) * 100, 2),
    }
    # Placeholder detection
    placeholders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        for val in ["?", "NA", "N/A", "na", "-", "--", "null", "none", "unknown", ""]:
            cnt = int((df[col].astype(str).str.strip().str.lower() == val.lower()).sum())
            if cnt > 0:
                placeholders.setdefault(col, {})[val] = cnt
    report["placeholders"] = placeholders
    return report


def eda_summary(df):
    """Statistical summary: describe, skewness, kurtosis, outliers, correlation."""
    num = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(include=["object", "category"])
    summary = {}

    if len(num.columns) > 0:
        summary["describe_numeric"] = num.describe().to_dict()
        summary["skewness"] = num.skew().round(3).to_dict()
        summary["kurtosis"] = num.kurtosis().round(3).to_dict()

    if len(cat.columns) > 0:
        summary["describe_categorical"] = cat.describe().to_dict()

    if len(num.columns) > 1:
        summary["correlation"] = num.corr().round(3).to_dict()

    # Outliers (IQR)
    outliers = {}
    for col in num.columns:
        data = num[col].dropna()
        if len(data) == 0:
            continue
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n = int(((data < lo) | (data > hi)).sum())
        if n > 0:
            outliers[col] = {
                "count": n,
                "pct": round(n / max(len(df), 1) * 100, 2),
                "lower": round(float(lo), 3),
                "upper": round(float(hi), 3),
            }
    summary["outliers"] = outliers

    # Z-score outliers
    zscore_outliers = {}
    for col in num.columns:
        data = num[col].dropna()
        if len(data) > 2:
            z = np.abs(scipy_stats.zscore(data))
            n = int((z > 3).sum())
            if n > 0:
                zscore_outliers[col] = {"count": n, "pct": round(n / max(len(df), 1) * 100, 2)}
    summary["zscore_outliers"] = zscore_outliers

    # Normality hint (Shapiro for small samples)
    normality = {}
    for col in num.columns[:10]:
        data = num[col].dropna()
        if 8 <= len(data) <= 5000:
            try:
                stat, p = scipy_stats.shapiro(data.sample(min(len(data), 5000), random_state=42))
                normality[col] = {
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p), 6),
                    "is_normal": p > 0.05,
                }
            except Exception:
                pass
    summary["normality"] = normality

    # High correlation pairs
    if len(num.columns) > 1:
        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = []
        for col_a in upper.columns:
            for col_b in upper.index:
                val = upper.at[col_b, col_a]
                if pd.notna(val) and val > 0.8:
                    high_corr.append({"col1": col_a, "col2": col_b, "corr": round(float(val), 3)})
        summary["high_correlation_pairs"] = sorted(high_corr, key=lambda x: x["corr"], reverse=True)

    return summary


def eda_insights(df):
    """Advanced insights: imbalance, cardinality, dtype suggestions."""
    insights = {}

    # High cardinality
    high_card = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        n = df[col].nunique()
        if n > 50:
            high_card[col] = n
    insights["high_cardinality"] = high_card

    # Class imbalance
    imbalance = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if 2 <= df[col].nunique() <= 20:
            counts = df[col].value_counts()
            ratio = counts.min() / max(counts.max(), 1)
            if ratio < 0.2:
                imbalance[col] = {
                    "distribution": counts.head(10).to_dict(),
                    "imbalance_ratio": round(float(ratio), 3),
                }
    insights["class_imbalance"] = imbalance

    # Constant columns
    insights["constant_columns"] = [c for c in df.columns if df[c].nunique() <= 1]

    # Low variance numeric
    low_var = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        v = df[col].var()
        if pd.notna(v) and v < 0.01:
            low_var[col] = round(float(v), 6)
    insights["low_variance"] = low_var

    # Dtype suggestions
    dtype_sugg = {}
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            pd.to_numeric(df[col], errors="raise")
            dtype_sugg[col] = "Convert to numeric"
            continue
        except Exception:
            pass
        try:
            pd.to_datetime(df[col], format="mixed", errors="raise")
            dtype_sugg[col] = "Convert to datetime"
            continue
        except Exception:
            pass
        if df[col].nunique() < 20:
            dtype_sugg[col] = "Convert to category"
    insights["dtype_suggestions"] = dtype_sugg

    return insights


# ════════════════════════════════════════════════════════════════
# 11. LLM-POWERED ENTERPRISE EDA  (100% AI-driven)
# ════════════════════════════════════════════════════════════════

# Default plan — fallback if LLM plan parsing fails
DEFAULT_ENTERPRISE_PLAN = [
    {"id": "overview", "name": "Data Overview & Structure",
     "objective": "Compute: row count, column count, memory usage in MB, duplicate row count, data type value_counts, list all column names with their dtypes. Store everything in result dict.",
     "priority": 1},
    {"id": "quality", "name": "Data Quality Audit",
     "objective": "Compute: missing value count per column, missing percentage per column, total missing cells, total cells, completeness percentage, complete row count. Detect placeholders in object columns — scan each for ?, NA, N/A, na, -, --, null, none, unknown, empty string. Store in result dict.",
     "priority": 2},
    {"id": "statistics", "name": "Statistical Analysis",
     "objective": "For numeric columns: compute describe(), skewness, kurtosis, variance, median. Flag columns with |skewness|>1 as highly skewed. Flag |kurtosis|>3 as heavy-tailed. Store all in result dict.",
     "priority": 3},
    {"id": "outliers", "name": "Outlier Detection",
     "objective": "For each numeric column: compute Q1, Q3, IQR, lower bound (Q1-1.5*IQR), upper bound (Q3+1.5*IQR), count outliers outside bounds, outlier percentage. Also compute Z-score outliers with |z|>3 count and percentage. Store in result dict.",
     "priority": 4},
    {"id": "correlation", "name": "Correlation Analysis",
     "objective": "Compute correlation matrix for numeric columns. Find ALL column pairs with |correlation|>0.8. Sort by absolute correlation descending. Store matrix and high-correlation pairs list in result dict.",
     "priority": 5},
    {"id": "categorical", "name": "Categorical Analysis",
     "objective": "For each categorical/object column: compute nunique (cardinality), top 10 value counts. For columns with 2-20 unique values: compute class imbalance ratio = min_count/max_count, identify min and max classes. Store all in result dict.",
     "priority": 6},
    {"id": "risk", "name": "Risk Assessment & Feature Engineering",
     "objective": "Identify: data leakage columns where nunique > 90% of row count, constant columns with nunique<=1, low-variance numeric columns with variance<0.01. Suggest log transforms for skewed columns, encoding strategies for categoricals. Store in result dict.",
     "priority": 7},
]


def enterprise_eda(llm, df, progress_callback=None):
    """100% LLM-Driven Enterprise EDA: PLAN → EXECUTE → INTERPRET.

    The AI plans what to analyze, generates pandas code for each step,
    executes it with self-healing retry, and interprets all real results.
    Zero hardcoded analysis logic — everything is AI-generated.

    Flow (identical pattern to chat system):
      1. LLM creates analysis PLAN (what tasks to run)
      2. For each task: LLM generates code → execute on FULL df → collect results
      3. Self-healing: if code fails → LLM fixes → retry (up to MAX_RETRIES)
      4. LLM interprets ALL computed results into professional report
    """
    logs = []
    ctx = get_df_context(df)
    ctx_str = format_context(ctx)

    # ── STAGE 1: AI PLANS the analysis ──────────────────────────
    if progress_callback:
        progress_callback("Stage 1/3: AI planning analysis strategy...")
    logs.append("[Enterprise] ══ Stage 1: AI creating analysis plan ══")

    plan_chain = make_chain(ENTERPRISE_PLAN_PROMPT, llm)
    plan_raw = run_chain(plan_chain, df_context=ctx_str)

    if plan_raw.startswith("LLM_ERROR"):
        logs.append(f"[Enterprise] ⚠️ Plan LLM error: {plan_raw[:100]}")
        plan = {"tasks": DEFAULT_ENTERPRISE_PLAN}
    else:
        plan = extract_json(plan_raw)
        if plan is None or not plan.get("tasks"):
            logs.append("[Enterprise] ⚠️ Plan JSON parsing failed — using intelligent defaults")
            plan = {"tasks": DEFAULT_ENTERPRISE_PLAN}

    tasks = sorted(plan["tasks"], key=lambda t: t.get("priority", 99))
    logs.append(f"[Enterprise] Plan created — {len(tasks)} analysis tasks:")
    for t in tasks:
        logs.append(f"  📋 [{t.get('id', '?')}] {t.get('name', '?')}")

    # ── STAGE 2: For each task, AI GENERATES & EXECUTES code ───
    fix_chain = make_chain(ERROR_FIX_PROMPT, llm)
    step_chain = make_chain(ENTERPRISE_STEP_PROMPT, llm)

    all_results = {}
    total_tasks = len(tasks)

    for i, task in enumerate(tasks):
        task_id = task.get("id", f"task_{i+1}")
        task_name = task.get("name", f"Task {i+1}")
        task_obj = task.get("objective", task_name)

        if progress_callback:
            pct = int(10 + (i / max(total_tasks, 1)) * 70)
            progress_callback(f"Stage 2/3: {task_name} ({i+1}/{total_tasks})...")
        logs.append(f"\n[Enterprise] ── Task {i+1}/{total_tasks}: {task_name} ──")

        # Compact summary of previous results for LLM context
        prev_summary = _summarize_results(all_results) if all_results else "No previous results yet."

        # LLM generates code for this specific task
        raw_code = run_chain(
            step_chain,
            task_name=task_name,
            task_objective=task_obj,
            df_context=ctx_str,
            previous_results=prev_summary,
        )

        if raw_code.startswith("LLM_ERROR"):
            logs.append(f"  ❌ Code generation failed: {raw_code[:100]}")
            all_results[task_id] = {"task_name": task_name, "error": raw_code[:200]}
            continue

        code = clean_code(extract_code(raw_code))
        if not code.strip():
            logs.append(f"  ❌ LLM returned empty code")
            all_results[task_id] = {"task_name": task_name, "error": "Empty code generated"}
            continue

        logs.append(f"  [Code] Generated ({len(code)} chars)")

        # Execute with self-healing retry loop (same as chat system)
        task_result = None
        final_code = code
        for attempt in range(MAX_RETRIES):
            exec_result = execute_pandas_code(final_code, df)

            if exec_result["success"] and exec_result.get("data") is not None:
                task_result = exec_result["data"]
                logs.append(f"  ✅ Executed successfully (attempt {attempt+1})")
                break
            elif exec_result["success"] and exec_result["type"] == "print":
                try:
                    task_result = json.loads(exec_result["output"])
                except Exception:
                    task_result = {"raw_output": exec_result["output"][:2000]}
                logs.append(f"  ✅ Output captured (attempt {attempt+1})")
                break
            elif exec_result["success"] and exec_result["type"] == "none":
                # Code ran but no result variable — try adding it
                final_code += "\nresult = 'Code executed — no result variable set'"
                logs.append(f"  ⚠️ No result variable — retrying with fallback")
                continue
            else:
                err = exec_result.get("error", "Unknown error")
                logs.append(f"  ❌ Attempt {attempt+1}: {err}")
                if attempt < MAX_RETRIES - 1:
                    fix_raw = run_chain(
                        fix_chain,
                        original_code=final_code,
                        error_message=err,
                        columns=str(ctx["columns"]),
                        dtypes=json.dumps(ctx["dtypes"]),
                    )
                    final_code = clean_code(extract_code(fix_raw))
                    logs.append(f"  🔧 Auto-fixed code (attempt {attempt+2})")

        if task_result is not None:
            all_results[task_id] = {
                "task_name": task_name,
                "data": _make_serializable(task_result),
                "code": final_code,
            }
        else:
            all_results[task_id] = {
                "task_name": task_name,
                "error": f"Failed after {MAX_RETRIES} attempts",
                "code": final_code,
            }
            logs.append(f"  ❌ Task failed after all retries")

    # Count successes/failures
    successes = sum(1 for v in all_results.values() if "data" in v)
    failures = sum(1 for v in all_results.values() if "error" in v)
    logs.append(f"\n[Enterprise] ══ Execution complete: {successes}✅ {failures}❌ ══")

    # ── STAGE 3: AI INTERPRETS all computed results ─────────────
    if progress_callback:
        progress_callback("Stage 3/3: AI generating professional interpretation...")
    logs.append("[Enterprise] ══ Stage 3: AI interpretation of computed results ══")

    results_str = _format_all_results(all_results)
    interp_chain = make_chain(ENTERPRISE_INTERPRETATION_PROMPT, llm)
    interpretation = run_chain(interp_chain, execution_result=results_str)

    if interpretation.startswith("LLM_ERROR"):
        logs.append(f"[Enterprise] Interpretation failed: {interpretation[:100]}")
        interpretation = (
            "⚠️ AI interpretation unavailable. "
            "Review the raw computed data below for all metrics.\n\n"
            f"Tasks completed: {successes}/{total_tasks}"
        )

    return {
        "error": None,
        "report": interpretation,
        "computed_data": all_results,
        "plan": plan,
        "logs": logs,
    }


def _make_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if isinstance(obj, float) else False:
        return None
    return obj


def _summarize_results(all_results):
    """Create compact summary of previous task results for LLM context."""
    parts = []
    for task_id, info in all_results.items():
        name = info.get("task_name", task_id)
        if info.get("error"):
            parts.append(f"• {name}: FAILED — {info['error'][:100]}")
        else:
            data = info.get("data", {})
            data_str = json.dumps(data, default=str)
            if len(data_str) > 800:
                data_str = data_str[:800] + "...(truncated)"
            parts.append(f"• {name}: {data_str}")
    return "\n".join(parts) if parts else "No previous results."


def _format_all_results(all_results):
    """Format all task results into a single string for LLM interpretation."""
    parts = []
    for task_id, info in all_results.items():
        name = info.get("task_name", task_id)
        parts.append(f"\n{'='*60}")
        parts.append(f"TASK: {name}")
        parts.append('='*60)
        if info.get("error"):
            parts.append(f"STATUS: ❌ FAILED — {info['error']}")
        else:
            data = info.get("data", {})
            parts.append("STATUS: ✅ SUCCESS")
            parts.append("COMPUTED RESULTS:")
            data_str = json.dumps(data, indent=2, default=str)
            # Truncate very large results to stay within LLM context
            if len(data_str) > 4000:
                data_str = data_str[:4000] + "\n...(truncated for brevity)"
            parts.append(data_str)
    return "\n".join(parts)


def llm_feature_suggestions(llm, df):
    """Feature engineering suggestions via LLM."""
    ctx_str = format_context(get_df_context(df))
    chain = make_chain(FEATURE_PROMPT, llm)
    raw = run_chain(chain, df_context=ctx_str)
    result = extract_json(raw)
    if result is None:
        result = {"feature_suggestions": [], "transformations": [],
                  "scaling_recommendations": [], "dimensionality_notes": raw[:500]}
    return result


def generate_eda_script(llm, df):
    """Generate downloadable EDA script."""
    ctx_str = format_context(get_df_context(df))
    chain = make_chain(EDA_SCRIPT_PROMPT, llm)
    raw = run_chain(chain, df_context=ctx_str)
    return extract_code(raw)


# ════════════════════════════════════════════════════════════════
# 12. AUTO VISUALIZATIONS  (no LLM — instant)
# ════════════════════════════════════════════════════════════════

def auto_visualizations(df):
    """Generate all standard EDA plots. Returns list of {title, figure}."""
    figs = []
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    plt.close("all")

    # --- Histogram + KDE + Boxplot for numerics ---
    for col in num_cols[:8]:
        try:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            data = df[col].dropna()
            axes[0].hist(data, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
            axes[0].set_title(f"Histogram — {col}")
            axes[0].set_xlabel(col); axes[0].set_ylabel("Frequency")
            if len(data) > 1:
                sns.kdeplot(data, ax=axes[1], fill=True, color="coral")
            axes[1].set_title(f"KDE — {col}"); axes[1].set_xlabel(col)
            axes[2].boxplot(data, vert=True, patch_artist=True,
                            boxprops=dict(facecolor="lightblue"))
            axes[2].set_title(f"Boxplot — {col}"); axes[2].set_ylabel(col)
            plt.tight_layout()
            figs.append({"title": f"Distribution: {col}", "figure": fig})
        except Exception:
            plt.close("all")

    # --- Violin plots ---
    for col in num_cols[:6]:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(y=df[col].dropna(), ax=ax, color="mediumpurple")
            ax.set_title(f"Violin — {col}"); ax.set_ylabel(col)
            plt.tight_layout()
            figs.append({"title": f"Violin: {col}", "figure": fig})
        except Exception:
            plt.close("all")

    # --- QQ plots ---
    for col in num_cols[:4]:
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            scipy_stats.probplot(df[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f"QQ Plot — {col}")
            plt.tight_layout()
            figs.append({"title": f"QQ Plot: {col}", "figure": fig})
        except Exception:
            plt.close("all")

    # --- Categorical bar plots ---
    for col in cat_cols[:6]:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            top = df[col].value_counts().head(15)
            sns.barplot(x=top.index.astype(str), y=top.values, ax=ax,
                        hue=top.index.astype(str), palette="viridis", legend=False)
            ax.set_title(f"Counts — {col}")
            ax.set_xlabel(col); ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            figs.append({"title": f"Counts: {col}", "figure": fig})
        except Exception:
            plt.close("all")

    # --- Pie charts for low-cardinality ---
    for col in cat_cols[:4]:
        if df[col].nunique() <= 8:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                counts = df[col].value_counts()
                ax.pie(counts.values, labels=counts.index.astype(str),
                       autopct="%1.1f%%", startangle=140,
                       colors=sns.color_palette("Set2", len(counts)))
                ax.set_title(f"Pie — {col}")
                plt.tight_layout()
                figs.append({"title": f"Pie: {col}", "figure": fig})
            except Exception:
                plt.close("all")

    # --- Correlation heatmap ---
    if len(num_cols) > 1:
        try:
            size = max(8, min(len(num_cols), 16))
            fig, ax = plt.subplots(figsize=(size, size * 0.8))
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=len(num_cols) <= 15, cmap="RdBu_r",
                        center=0, fmt=".2f", square=True, ax=ax, linewidths=0.5)
            ax.set_title("Correlation Heatmap")
            plt.tight_layout()
            figs.append({"title": "Correlation Heatmap", "figure": fig})
        except Exception:
            plt.close("all")

    # --- Top-correlated scatter plots ---
    if len(num_cols) > 1:
        try:
            corr = df[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.stack().sort_values(ascending=False).head(3)
            for (c1, c2), _ in pairs.items():
                fig, ax = plt.subplots(figsize=(8, 6))
                s = df[[c1, c2]].dropna()
                if len(s) > 2000:
                    s = s.sample(2000, random_state=42)
                ax.scatter(s[c1], s[c2], alpha=0.4, s=15, color="teal")
                ax.set_title(f"Scatter — {c1} vs {c2}")
                ax.set_xlabel(c1); ax.set_ylabel(c2)
                plt.tight_layout()
                figs.append({"title": f"Scatter: {c1} vs {c2}", "figure": fig})
        except Exception:
            plt.close("all")

    # --- Pairplot ---
    if 2 <= len(num_cols) <= 5:
        try:
            s = df[num_cols].dropna()
            if len(s) > 500:
                s = s.sample(500, random_state=42)
            g = sns.pairplot(s, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15})
            g.figure.suptitle("Pairplot", y=1.02)
            figs.append({"title": "Pairplot", "figure": g.figure})
        except Exception:
            plt.close("all")

    # --- Missing values heatmap ---
    if df.isnull().sum().sum() > 0:
        try:
            fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 0.4), 6))
            sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="YlOrRd", ax=ax)
            ax.set_title("Missing Values Heatmap")
            plt.tight_layout()
            figs.append({"title": "Missing Values Heatmap", "figure": fig})
        except Exception:
            plt.close("all")

    return figs


# ════════════════════════════════════════════════════════════════
# 13. CHAT ORCHESTRATOR  (planner → agents → critic → reflect)
# ════════════════════════════════════════════════════════════════

def handle_chat(query, df, llm, pandas_fn, viz_fn, planner_fn, profiling=None):
    """Full orchestration: plan → execute → evaluate → reflect."""
    result = {"plan": None, "pandas_result": None, "viz_result": None,
              "explanation": "", "logs": [], "critic": None, "reflection": None}

    # Plan
    try:
        plan = planner_fn(query, df, profiling)
    except Exception as e:
        q = query.lower()
        plan = {
            "needs_pandas": True,
            "needs_visualization": any(w in q for w in
                ["plot", "chart", "graph", "visual", "hist", "scatter", "bar", "heatmap"]),
            "pandas_query": query,
            "viz_query": query,
            "explanation": "Processing your request...",
        }
        result["logs"].append(f"Planner fallback: {e}")

    result["plan"] = plan
    result["logs"].append(f"Strategy: {plan.get('analysis_strategy', 'quick_answer')}")
    result["explanation"] = plan.get("explanation", "")

    # Pandas Agent
    if plan.get("needs_pandas", True):
        pq = plan.get("pandas_query") or query
        pr = pandas_fn(pq, df)
        result["pandas_result"] = pr
        result["logs"].extend(pr.get("logs", []))
        result["critic"] = pr.get("critic")
        result["reflection"] = pr.get("reflection")

        # If critic says insufficient and we have suggestion, try again
        if (pr.get("critic") and not pr["critic"].get("is_sufficient", True)
                and pr["critic"].get("improvement_suggestion")):
            result["logs"].append("[Critic] Insufficient — running improvement...")
            pr2 = pandas_fn(pr["critic"]["improvement_suggestion"], df)
            if pr2["success"]:
                result["pandas_result_followup"] = pr2
                result["logs"].extend(pr2.get("logs", []))

    # Viz Agent
    if plan.get("needs_visualization", False):
        vq = plan.get("viz_query") or query
        vr = viz_fn(vq, df)
        result["viz_result"] = vr
        result["logs"].extend(vr.get("logs", []))

    # Auto-trigger viz if reflection suggests it
    if (result.get("reflection") and result["reflection"].get("should_visualize_more")
            and not result.get("viz_result")):
        result["logs"].append("[Reflection] Triggering visualization...")
        vr = viz_fn(query, df)
        result["viz_result"] = vr
        result["logs"].extend(vr.get("logs", []))

    return result
