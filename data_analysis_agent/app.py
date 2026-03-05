"""
app.py — Streamlit UI for Autonomous Data Analysis Agent
=========================================================
Tabs: Preview | Deep EDA | Enterprise EDA | Visualizations | Chat | Script
"""

import sys
import os
import streamlit as st
import pandas as pd
import json

# Ensure agent.py can be found regardless of where this is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    create_llm,
    setup_pandas_agent,
    setup_viz_agent,
    setup_planner,
    profile_dataset,
    eda_overview,
    eda_cleaning,
    eda_summary,
    eda_insights,
    enterprise_eda,
    llm_feature_suggestions,
    generate_eda_script,
    auto_visualizations,
    handle_chat,
)

def run():
    """Main entry point for the Data Analysis Agent page."""

    # ═══════════════════════════════════════════════════════════
    # SESSION STATE
    # ═══════════════════════════════════════════════════════════
    DEFAULTS = {
        "df": None, "file_key": None,
        "llm": None, "pandas_fn": None, "viz_fn": None,
        "viz_rec_fn": None, "planner_fn": None,
        "profiling": None, "chat_history": [],
        # EDA cache
        "eda_done": False, "overview": None, "cleaning": None,
        "summary_stats": None, "insights": None,
        "feature_sugg": None, "auto_viz": None,
        "eda_script": None, "enterprise_report": None,
    }
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


    # ═══════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════
    with st.sidebar:
        st.title("🔥 Data Analysis Agent")
        st.caption("AI-Powered • Self-Correcting • Enterprise Grade")
        st.markdown("---")

        uploaded = st.file_uploader("📁 Upload CSV", type=["csv"])

        if uploaded is not None:
            fk = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.file_key != fk:
                try:
                    df = pd.read_csv(uploaded)
                    st.session_state.df = df
                    st.session_state.file_key = fk
                    for key in ["eda_done", "overview", "cleaning", "summary_stats",
                                "insights", "feature_sugg", "auto_viz",
                                "eda_script", "enterprise_report", "profiling"]:
                        st.session_state[key] = DEFAULTS[key]
                    st.session_state.chat_history = []
                    st.success(f"✅ **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} cols")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        st.markdown("---")

        if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
            if st.session_state.df is None:
                st.warning("Upload a dataset first.")
            else:
                with st.spinner("Connecting to LLM..."):
                    try:
                        llm = create_llm()
                        pandas_fn, pandas_tool = setup_pandas_agent(llm)
                        viz_fn, viz_tool, viz_rec_fn = setup_viz_agent(llm)
                        planner_fn = setup_planner(llm)

                        st.session_state.llm = llm
                        st.session_state.pandas_fn = pandas_fn
                        st.session_state.viz_fn = viz_fn
                        st.session_state.viz_rec_fn = viz_rec_fn
                        st.session_state.planner_fn = planner_fn

                        st.success("✅ Agents ready!")

                        # Auto-profile dataset
                        with st.spinner("Profiling dataset..."):
                            prof = profile_dataset(llm, st.session_state.df)
                            st.session_state.profiling = prof
                            score = prof.get("data_quality_score", "?")
                            level = prof.get("complexity_level", "?")
                            st.info(f"📊 Quality: **{score}/100** | Complexity: **{level}**")

                    except Exception as e:
                        st.error(f"Init failed: {e}")

        st.markdown("---")

        # Dataset info
        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown("### 📊 Dataset")
            st.markdown(f"**Rows:** {df.shape[0]:,}")
            st.markdown(f"**Cols:** {df.shape[1]}")
            mem = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.markdown(f"**Mem:** {mem:.2f} MB")

            # Profiling summary
            if st.session_state.profiling:
                p = st.session_state.profiling
                st.markdown("---")
                st.markdown("### 🧪 Profile")
                st.markdown(f"Quality: **{p.get('data_quality_score', '?')}/100**")
                st.markdown(f"Complexity: **{p.get('complexity_level', '?')}**")
                if p.get("risk_areas"):
                    st.markdown(f"Risks: {', '.join(p['risk_areas'][:3])}")

            st.markdown("---")
            st.markdown("### 📋 Columns")
            for col in df.columns:
                st.markdown(f"- `{col}` *({df[col].dtype})*")


    # ═══════════════════════════════════════════════════════════
    # MAIN
    # ═══════════════════════════════════════════════════════════
    st.title("🔥 Autonomous Data Analysis Agent")
    st.markdown("Upload CSV → Initialize → Explore with **Deep EDA**, **Enterprise Analysis**, **AI Chat**, **Auto Viz**")

    if st.session_state.df is None:
        st.info("👈 Upload a CSV file in the sidebar to begin.")
        st.stop()

    df = st.session_state.df

    # ═══════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════
    tab_preview, tab_eda, tab_enterprise, tab_viz, tab_chat, tab_script = st.tabs([
        "📋 Preview", "🔍 Deep EDA", "🏢 Enterprise EDA",
        "📊 Visualizations", "💬 Chat", "📜 Script",
    ])


    # ═══════════════════════════════════════════════════════════
    # TAB 1: PREVIEW
    # ═══════════════════════════════════════════════════════════
    with tab_preview:
        st.subheader("Dataset Preview")
        n = st.slider("Rows to display", 5, min(200, len(df)), 20)
        st.dataframe(df.head(n), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Column Types")
            dtypes_df = df.dtypes.astype(str).reset_index()
            dtypes_df.columns = ["Column", "Type"]
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Quick Statistics")
            st.dataframe(df.describe(include="all").T, use_container_width=True)


    # ═══════════════════════════════════════════════════════════
    # TAB 2: DEEP EDA
    # ═══════════════════════════════════════════════════════════
    with tab_eda:
        if st.button("🔍 Run Full EDA", type="primary", use_container_width=True):
            with st.spinner("Computing overview..."):
                st.session_state.overview = eda_overview(df)
            with st.spinner("Generating cleaning report..."):
                st.session_state.cleaning = eda_cleaning(df)
            with st.spinner("Computing statistics..."):
                st.session_state.summary_stats = eda_summary(df)
            with st.spinner("Detecting patterns..."):
                st.session_state.insights = eda_insights(df)

            if st.session_state.llm:
                with st.spinner("🧪 AI feature suggestions..."):
                    try:
                        st.session_state.feature_sugg = llm_feature_suggestions(st.session_state.llm, df)
                    except Exception as e:
                        st.session_state.feature_sugg = {"feature_suggestions": [],
                            "transformations": [], "scaling_recommendations": [],
                            "dimensionality_notes": str(e)}
            st.session_state.eda_done = True
            st.success("✅ EDA complete!")

        if not st.session_state.eda_done:
            st.info("Click **Run Full EDA** to analyze the dataset.")
        else:
            # --- 1. Overview ---
            with st.expander("1️⃣ Data Overview", expanded=True):
                ov = st.session_state.overview
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows", f"{ov['shape'][0]:,}")
                m2.metric("Columns", ov["shape"][1])
                m3.metric("Duplicates", f"{ov['duplicates']:,}")
                m4.metric("Memory", ov["memory"])
                st.dataframe(ov["head"], use_container_width=True)

            # --- 2. Cleaning ---
            with st.expander("2️⃣ Data Cleaning Report"):
                cl = st.session_state.cleaning
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Rows", f"{cl['total_rows']:,}")
                m2.metric("Complete", f"{cl['complete_rows']:,} ({cl['complete_pct']}%)")
                m3.metric("Duplicates", f"{cl['duplicates']:,}")

                if cl["missing_values"]:
                    st.markdown("**Missing Values:**")
                    miss_df = pd.DataFrame({
                        "Column": list(cl["missing_values"].keys()),
                        "Count": list(cl["missing_values"].values()),
                        "Percent": [cl["missing_pct"].get(k, 0) for k in cl["missing_values"]],
                    }).sort_values("Count", ascending=False)
                    st.dataframe(miss_df, use_container_width=True, hide_index=True)
                else:
                    st.success("No missing values! 🎉")

                if cl["placeholders"]:
                    st.markdown("**⚠️ Placeholders Detected:**")
                    st.json(cl["placeholders"])

            # --- 3. Summary ---
            with st.expander("3️⃣ Statistical Summary"):
                sm = st.session_state.summary_stats

                if sm.get("describe_numeric"):
                    st.markdown("**Numeric:**")
                    st.dataframe(pd.DataFrame(sm["describe_numeric"]).T,
                                 use_container_width=True)

                if sm.get("describe_categorical"):
                    st.markdown("**Categorical:**")
                    st.dataframe(pd.DataFrame(sm["describe_categorical"]).T,
                                 use_container_width=True)

                if sm.get("skewness"):
                    st.markdown("**Skewness & Kurtosis:**")
                    skew_data = sm["skewness"]
                    kurt_data = sm.get("kurtosis", {})
                    skew_df = pd.DataFrame({
                        "Skewness": skew_data,
                        "Kurtosis": {k: kurt_data.get(k, None) for k in skew_data},
                    })
                    skew_df["Interpretation"] = skew_df["Skewness"].apply(
                        lambda x: "⚠️ Highly Skewed" if abs(x) > 1
                        else ("Moderate" if abs(x) > 0.5 else "≈ Symmetric"))
                    st.dataframe(skew_df, use_container_width=True)

                if sm.get("normality"):
                    st.markdown("**Normality Tests (Shapiro-Wilk):**")
                    norm_df = pd.DataFrame(sm["normality"]).T
                    st.dataframe(norm_df, use_container_width=True)

                if sm.get("outliers"):
                    st.markdown("**🔴 Outliers (IQR):**")
                    st.dataframe(pd.DataFrame(sm["outliers"]).T,
                                 use_container_width=True)

                if sm.get("zscore_outliers"):
                    st.markdown("**🔴 Outliers (Z-Score > 3):**")
                    st.dataframe(pd.DataFrame(sm["zscore_outliers"]).T,
                                 use_container_width=True)

                if sm.get("high_correlation_pairs"):
                    st.markdown("**🔗 High Correlation Pairs (>0.8):**")
                    for p in sm["high_correlation_pairs"][:10]:
                        st.markdown(f"- **{p['col1']}** ↔ **{p['col2']}**: `{p['corr']}`")

                if sm.get("correlation"):
                    st.markdown("**Correlation Matrix:**")
                    corr_df = pd.DataFrame(sm["correlation"])
                    st.dataframe(corr_df.style.background_gradient(
                        cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)

            # --- 4. Insights ---
            with st.expander("4️⃣ Advanced Insights"):
                ins = st.session_state.insights

                if ins.get("class_imbalance"):
                    st.markdown("**⚠️ Class Imbalance:**")
                    for col, info in ins["class_imbalance"].items():
                        st.markdown(f"- **{col}**: ratio = `{info['imbalance_ratio']}`")
                        st.json(info["distribution"])

                if ins.get("high_cardinality"):
                    st.markdown("**⚠️ High Cardinality:**")
                    for col, n in ins["high_cardinality"].items():
                        st.markdown(f"- **{col}**: {n:,} unique values")

                if ins.get("constant_columns"):
                    st.markdown("**⚠️ Constant Columns:**")
                    for col in ins["constant_columns"]:
                        st.markdown(f"- `{col}`")

                if ins.get("low_variance"):
                    st.markdown("**⚠️ Low Variance:**")
                    for col, v in ins["low_variance"].items():
                        st.markdown(f"- **{col}**: variance = `{v}`")

                if ins.get("dtype_suggestions"):
                    st.markdown("**💡 Dtype Suggestions:**")
                    for col, s in ins["dtype_suggestions"].items():
                        st.markdown(f"- **{col}**: {s}")

                # Feature suggestions
                if st.session_state.feature_sugg:
                    fs = st.session_state.feature_sugg
                    if fs.get("feature_suggestions"):
                        st.markdown("#### 🧪 Feature Engineering")
                        for feat in fs["feature_suggestions"]:
                            st.markdown(f"- **{feat.get('name', '?')}**: {feat.get('description', '')}")
                            if feat.get("code"):
                                st.code(feat["code"], language="python")
                    if fs.get("transformations"):
                        st.markdown("**Transformations:**")
                        for t in fs["transformations"]:
                            st.markdown(f"- {t}")
                    if fs.get("scaling_recommendations"):
                        st.markdown("**Scaling:**")
                        for s in fs["scaling_recommendations"]:
                            st.markdown(f"- {s}")


    # ═══════════════════════════════════════════════════════════
    # TAB 3: ENTERPRISE EDA  (100% AI-Driven: PLAN → EXECUTE → INTERPRET)
    # ═══════════════════════════════════════════════════════════
    with tab_enterprise:
        st.subheader("🏢 Enterprise-Grade Deep Analysis")
        st.markdown(
            "**100% AI-Driven: PLAN → EXECUTE → INTERPRET** — "
            "The AI plans what to analyze, generates code for each step, "
            "executes it on the FULL dataset, and interprets all results. "
            "Same architecture as the Chat system."
        )

        if st.session_state.llm is None:
            st.warning("Initialize the agent first (sidebar).")
        else:
            if st.button("🏢 Run Enterprise EDA", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Starting enterprise analysis...")

                def _progress(msg):
                    if "Stage 1" in msg:
                        progress_bar.progress(10, text=msg)
                    elif "Stage 2" in msg:
                        # Extract task progress from message
                        try:
                            import re as _re
                            m = _re.search(r"\((\d+)/(\d+)\)", msg)
                            if m:
                                cur, tot = int(m.group(1)), int(m.group(2))
                                pct = int(10 + (cur / max(tot, 1)) * 70)
                                progress_bar.progress(min(pct, 80), text=msg)
                            else:
                                progress_bar.progress(50, text=msg)
                        except Exception:
                            progress_bar.progress(50, text=msg)
                    elif "Stage 3" in msg:
                        progress_bar.progress(85, text=msg)

                try:
                    result = enterprise_eda(
                        st.session_state.llm, df,
                        progress_callback=_progress,
                    )
                    st.session_state.enterprise_report = result
                    progress_bar.progress(100, text="✅ Enterprise EDA complete!")
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        # Count results
                        computed = result.get("computed_data", {})
                        ok = sum(1 for v in computed.values() if "data" in v)
                        fail = sum(1 for v in computed.values() if "error" in v)
                        st.success(f"✅ AI completed {ok} analysis tasks ({fail} skipped) + professional interpretation!")
                except Exception as e:
                    progress_bar.progress(100, text="❌ Error")
                    st.error(f"Error: {e}")

        if st.session_state.enterprise_report and not st.session_state.enterprise_report.get("error"):
            ent = st.session_state.enterprise_report
            computed = ent.get("computed_data", {})
            report = ent.get("report", "")
            plan = ent.get("plan", {})

            # ── AI ANALYSIS PLAN ──
            with st.expander("🧠 AI Analysis Plan", expanded=False):
                st.caption("The AI created this plan autonomously based on your dataset.")
                tasks = plan.get("tasks", [])
                for t in tasks:
                    tid = t.get("id", "?")
                    task_info = computed.get(tid, {})
                    if "data" in task_info:
                        icon = "✅"
                    elif "error" in task_info:
                        icon = "❌"
                    else:
                        icon = "⏳"
                    st.markdown(f"{icon} **{t.get('name', '?')}** — _{t.get('objective', '')[:120]}_")

            # ── AI-GENERATED & EXECUTED RESULTS (per task) ──
            with st.expander("📊 Computed Results (AI-Generated Code + Execution)", expanded=False):
                st.caption(
                    "For each task: the AI generated pandas code → executed on FULL dataset → "
                    "results captured. Same flow as Chat system."
                )
                for task_id, info in computed.items():
                    name = info.get("task_name", task_id)
                    st.markdown(f"---")
                    if info.get("error"):
                        st.error(f"**{name}**: {info['error']}")
                        if info.get("code"):
                            with st.expander(f"Failed code for {name}"):
                                st.code(info["code"], language="python")
                    else:
                        st.markdown(f"**✅ {name}**")
                        data = info.get("data", {})
                        if isinstance(data, dict):
                            # Try to render nested dicts nicely
                            for key, val in data.items():
                                if isinstance(val, dict) and len(val) > 0:
                                    try:
                                        val_df = pd.DataFrame(val)
                                        if len(val_df) > 0:
                                            st.markdown(f"*{key}:*")
                                            st.dataframe(val_df, use_container_width=True, hide_index=False)
                                            continue
                                    except Exception:
                                        pass
                                    try:
                                        val_df = pd.DataFrame(val, index=[0])
                                        if len(val_df.columns) <= 20:
                                            st.markdown(f"*{key}:*")
                                            st.dataframe(val_df, use_container_width=True, hide_index=True)
                                            continue
                                    except Exception:
                                        pass
                                if isinstance(val, (list, dict)):
                                    st.markdown(f"*{key}:*")
                                    st.json(val)
                                else:
                                    st.markdown(f"*{key}:* `{val}`")
                        elif isinstance(data, str):
                            st.text(data[:3000])
                        else:
                            st.json(data)

                        if info.get("code"):
                            with st.expander(f"AI-Generated code for {name}"):
                                st.code(info["code"], language="python")

            # ── AI INTERPRETATION ──
            st.markdown("---")
            st.subheader("🧠 AI Interpretation (based on computed data)")

            # Parse sections
            sections = [
                ("SECTION 1", "📊 Data Overview"),
                ("SECTION 2", "🧹 Data Quality Audit"),
                ("SECTION 3", "📈 Statistical Analysis"),
                ("SECTION 4", "🔍 Outlier Analysis"),
                ("SECTION 5", "🔗 Correlation Analysis"),
                ("SECTION 6", "🏷️ Categorical Analysis"),
                ("SECTION 7", "⚠️ Risk Assessment"),
                ("SECTION 8", "🎨 Feature Engineering"),
                ("SECTION 9", "📋 Executive Summary"),
            ]

            section_texts = {}
            for i, (marker, title) in enumerate(sections):
                start_idx = report.upper().find(marker)
                if start_idx >= 0:
                    next_idx = len(report)
                    if i + 1 < len(sections):
                        next_start = report.upper().find(sections[i + 1][0])
                        if next_start > start_idx:
                            next_idx = next_start
                    section_texts[title] = report[start_idx:next_idx].strip()

            if section_texts:
                for title, text in section_texts.items():
                    with st.expander(title, expanded=title == "📋 Executive Summary"):
                        st.markdown(text)
            else:
                st.markdown(report)

            # Execution pipeline logs
            if ent.get("logs"):
                with st.expander("⚙️ AI Execution Pipeline Logs"):
                    for log in ent["logs"]:
                        st.text(log)

            # Download
            st.download_button(
                "⬇️ Download Enterprise Report",
                data=report,
                file_name="enterprise_eda_report.md",
                mime="text/markdown",
            )


    # ═══════════════════════════════════════════════════════════
    # TAB 4: VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════
    with tab_viz:
        c1, c2 = st.columns([1, 1])

        with c1:
            if st.button("📊 Auto-Generate All Plots", type="primary",
                          use_container_width=True):
                with st.spinner("Generating visualizations..."):
                    st.session_state.auto_viz = auto_visualizations(df)
                st.success(f"✅ Generated {len(st.session_state.auto_viz)} plots!")

        with c2:
            custom_q = st.text_input("Custom plot:",
                                     placeholder="e.g. scatter of age vs income colored by gender")
            if st.button("🎨 Generate", use_container_width=True) and custom_q:
                if st.session_state.viz_fn is None:
                    st.warning("Initialize agent first.")
                else:
                    with st.spinner("Generating..."):
                        vr = st.session_state.viz_fn(custom_q, df)
                    if vr["success"]:
                        st.pyplot(vr["figure"])
                        with st.expander("Code"):
                            st.code(vr.get("code", ""), language="python")
                    else:
                        st.error(f"Failed: {vr.get('error', 'Unknown')}")
                        with st.expander("Logs"):
                            for log in vr.get("logs", []):
                                st.text(log)

        # AI Viz Recommendations
        if st.session_state.viz_rec_fn:
            with st.expander("🧠 AI Visualization Recommendations"):
                if st.button("Get Smart Recommendations"):
                    with st.spinner("Analyzing best visualizations..."):
                        recs = st.session_state.viz_rec_fn("suggest best visualizations", df)
                    if recs and recs.get("recommended_plots"):
                        for i, plot in enumerate(recs["recommended_plots"], 1):
                            st.markdown(
                                f"**{i}. {plot.get('type', '?')}** — "
                                f"Columns: `{plot.get('columns', [])}` — "
                                f"_{plot.get('reason', '')}_"
                            )
                    else:
                        st.info("No recommendations available.")

        # Display auto plots
        if st.session_state.auto_viz:
            st.markdown("---")
            for viz in st.session_state.auto_viz:
                st.markdown(f"### {viz['title']}")
                st.pyplot(viz["figure"])
                st.markdown("---")


    # ═══════════════════════════════════════════════════════════
    # TAB 5: CHAT
    # ═══════════════════════════════════════════════════════════
    with tab_chat:
        st.subheader("💬 Chat with your Data")

        if st.session_state.llm is None:
            st.warning("Initialize the agent from the sidebar first.")
        else:
            st.markdown(
                "Ask anything: *average salary? • correlation between X and Y? "
                "• which category has highest sales? • plot age distribution • "
                "suggest features for ML • how many outliers?*"
            )

            # Chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    if msg.get("content"):
                        st.markdown(msg["content"])
                    if msg.get("figure"):
                        st.pyplot(msg["figure"])
                    if msg.get("code"):
                        with st.expander("Code"):
                            st.code(msg["code"], language="python")

            # Input
            user_input = st.chat_input("Ask anything about your data...")

            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = handle_chat(
                            user_input, df,
                            st.session_state.llm,
                            st.session_state.pandas_fn,
                            st.session_state.viz_fn,
                            st.session_state.planner_fn,
                            st.session_state.profiling,
                        )

                    parts = []
                    figure = None
                    code = ""

                    # Explanation
                    if result.get("explanation"):
                        st.markdown(f"**{result['explanation']}**")
                        parts.append(result["explanation"])

                    # Pandas result
                    pr = result.get("pandas_result")
                    if pr:
                        if pr["success"]:
                            st.markdown("**📊 Result:**")
                            out = pr["output"]
                            if len(out) > 3000:
                                out = out[:3000] + "\n...(truncated)"
                            st.text(out)
                            parts.append(out[:500])
                            code = pr.get("code", "")
                        else:
                            st.error(f"Error: {pr.get('error', 'Unknown')}")
                            parts.append(f"Error: {pr.get('error', '')}")

                    # Follow-up result
                    pr2 = result.get("pandas_result_followup")
                    if pr2 and pr2.get("success"):
                        st.markdown("**📊 Follow-up Analysis:**")
                        st.text(pr2["output"][:2000])

                    # Critic feedback
                    critic = result.get("critic")
                    if critic:
                        with st.expander("🎯 Critic Evaluation"):
                            st.markdown(f"**Sufficient:** {critic.get('is_sufficient', '?')}")
                            st.markdown(f"**Reason:** {critic.get('reason', '')}")
                            if critic.get("missing_aspects"):
                                st.markdown("**Missing:** " + ", ".join(critic["missing_aspects"]))
                            if critic.get("improvement_suggestion"):
                                st.markdown(f"**Suggestion:** {critic['improvement_suggestion']}")

                    # Reflection
                    reflection = result.get("reflection")
                    if reflection:
                        with st.expander("🔄 Self-Reflection"):
                            st.markdown(f"**Complete:** {reflection.get('analysis_complete', '?')}")
                            if reflection.get("missed_opportunities"):
                                st.markdown("**Missed:** " + ", ".join(reflection["missed_opportunities"]))
                            if reflection.get("next_step_suggestion"):
                                st.markdown(f"**Next:** {reflection['next_step_suggestion']}")

                    # Viz result
                    vr = result.get("viz_result")
                    if vr:
                        if vr["success"]:
                            st.pyplot(vr["figure"])
                            figure = vr["figure"]
                            code = code or vr.get("code", "")
                        else:
                            st.warning(f"Viz issue: {vr.get('error', '')}")

                    # Code
                    if code:
                        with st.expander("Code"):
                            st.code(code, language="python")

                    # Logs
                    if result.get("logs"):
                        with st.expander("Execution Logs"):
                            for log in result["logs"]:
                                st.text(log)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "\n".join(parts) if parts else "Done.",
                        "figure": figure,
                        "code": code,
                    })


    # ═══════════════════════════════════════════════════════════
    # TAB 6: SCRIPT
    # ═══════════════════════════════════════════════════════════
    with tab_script:
        st.subheader("📜 Generate Full EDA Script")
        st.markdown("Get a clean, downloadable Python script.")

        if st.session_state.llm is None:
            st.warning("Initialize agent first.")
        else:
            if st.button("📝 Generate Script", type="primary", use_container_width=True):
                with st.spinner("Generating..."):
                    try:
                        code = generate_eda_script(st.session_state.llm, df)
                        st.session_state.eda_script = code
                        st.success("✅ Script ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.eda_script:
            st.code(st.session_state.eda_script, language="python")
            st.download_button(
                "⬇️ Download Script",
                data=st.session_state.eda_script,
                file_name="eda_script.py",
                mime="text/x-python",
            )


    # ═══════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        "🔥 **Autonomous Data Analysis Agent** — "
        "LangChain + HuggingFace + Streamlit | "
        "Self-correcting • Enterprise-grade • Critic + Reflection loops"
    )


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(
        page_title="🔥 Data Analysis Agent",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    run()
