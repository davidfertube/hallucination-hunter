"""
Hallucination Hunter: LLM Evaluation Dashboard
===============================================

EvalOps platform for measuring groundedness, relevance, 
and coherence of LLM responses in industrial applications.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Hallucination Hunter",
    page_icon="",
    layout="wide"
)

# ============================================
# SAMPLE DATA
# ============================================

SAMPLE_EVAL_DATA = pd.DataFrame({
    "model": ["GPT-4", "GPT-4", "GPT-4", "Gemini Pro", "Gemini Pro", "Gemini Pro", 
              "Claude 3", "Claude 3", "Claude 3"],
    "test_case": ["Contract Q1", "Contract Q2", "Contract Q3"] * 3,
    "groundedness": [0.92, 0.88, 0.95, 0.89, 0.85, 0.91, 0.95, 0.93, 0.96],
    "relevance": [0.88, 0.91, 0.85, 0.91, 0.88, 0.93, 0.87, 0.89, 0.86],
    "coherence": [0.94, 0.92, 0.96, 0.90, 0.88, 0.92, 0.93, 0.95, 0.91],
    "latency_ms": [450, 480, 520, 380, 350, 400, 620, 580, 650]
})

SAMPLE_TRACES = [
    {
        "id": "trace_001",
        "query": "What is the indemnification cap in the contract?",
        "context": "The Customer shall indemnify Provider for claims up to $1,000,000...",
        "response": "The indemnification cap is $1,000,000 as stated in Section 8.2.",
        "groundedness_score": 0.95,
        "reasoning": "Response directly quotes the cap amount from the source document.",
        "verdict": " Grounded"
    },
    {
        "id": "trace_002",
        "query": "What are the termination rights?",
        "context": "Either party may terminate with 30 days written notice...",
        "response": "The contract can be terminated immediately by either party.",
        "groundedness_score": 0.35,
        "reasoning": "Response contradicts source. Source says 30 days notice required, response claims immediate termination.",
        "verdict": " Hallucination"
    },
    {
        "id": "trace_003",
        "query": "What data processing is allowed?",
        "context": "Provider may process data for service delivery and improvement...",
        "response": "Data can be used for service delivery. The contract also mentions service improvement, though specifics are vague.",
        "groundedness_score": 0.82,
        "reasoning": "Response is mostly accurate but adds subjective interpretation ('vague') not in source.",
        "verdict": "️ Partially Grounded"
    }
]

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/391/direct-hit_1f3af.png", width=80)
    st.title("Hallucination Hunter")
    st.markdown("---")
    
    st.subheader(" Data Source")
    data_source = st.radio(
        "Select data source:",
        ["Sample Data", "Upload CSV", "API Connection"]
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload evaluation results", type=['csv'])
        if uploaded_file:
            eval_data = pd.read_csv(uploaded_file)
        else:
            eval_data = SAMPLE_EVAL_DATA
    else:
        eval_data = SAMPLE_EVAL_DATA
    
    st.markdown("---")
    
    st.subheader("️ Settings")
    threshold = st.slider("Hallucination Threshold", 0.0, 1.0, 0.7)
    
    st.markdown("---")
    st.markdown("""
    **Author:** David Fernandez  
    [Portfolio](https://davidfernandez.dev) | [GitHub](https://github.com/davidfertube)
    """)


# ============================================
# MAIN CONTENT
# ============================================

st.title(" Hallucination Hunter")
st.markdown("### Automated Groundedness & Relevance Testing for Industrial Agents")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([" Overview", " Trace Analysis", " Trends", " Export"])


# ============================================
# TAB 1: OVERVIEW
# ============================================

with tab1:
    st.subheader("Evaluation Summary")
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_groundedness = eval_data["groundedness"].mean()
        st.metric(
            "Avg Groundedness",
            f"{avg_groundedness:.1%}",
            delta=f"+{(avg_groundedness - 0.85):.1%}" if avg_groundedness > 0.85 else f"{(avg_groundedness - 0.85):.1%}"
        )
    
    with col2:
        avg_relevance = eval_data["relevance"].mean()
        st.metric(
            "Avg Relevance", 
            f"{avg_relevance:.1%}",
            delta=f"+{(avg_relevance - 0.85):.1%}" if avg_relevance > 0.85 else f"{(avg_relevance - 0.85):.1%}"
        )
    
    with col3:
        avg_coherence = eval_data["coherence"].mean()
        st.metric(
            "Avg Coherence",
            f"{avg_coherence:.1%}",
            delta=f"+{(avg_coherence - 0.85):.1%}" if avg_coherence > 0.85 else f"{(avg_coherence - 0.85):.1%}"
        )
    
    with col4:
        hallucination_rate = (eval_data["groundedness"] < threshold).mean()
        st.metric(
            "Hallucination Rate",
            f"{hallucination_rate:.1%}",
            delta=f"{hallucination_rate:.1%}",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Model comparison chart
    st.subheader("Model Comparison")
    
    model_metrics = eval_data.groupby("model").agg({
        "groundedness": "mean",
        "relevance": "mean", 
        "coherence": "mean",
        "latency_ms": "mean"
    }).reset_index()
    
    fig = px.bar(
        model_metrics.melt(id_vars=["model"], value_vars=["groundedness", "relevance", "coherence"]),
        x="model",
        y="value",
        color="variable",
        barmode="group",
        title="Model Performance Comparison",
        labels={"value": "Score", "variable": "Metric", "model": "Model"}
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    
    # Latency comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_latency = px.bar(
            model_metrics,
            x="model",
            y="latency_ms",
            title="Average Latency by Model",
            color="model"
        )
        fig_latency.update_layout(showlegend=False)
        st.plotly_chart(fig_latency, use_container_width=True)
    
    with col2:
        # Radar chart
        categories = ["Groundedness", "Relevance", "Coherence"]
        
        fig_radar = go.Figure()
        
        for model in model_metrics["model"].unique():
            row = model_metrics[model_metrics["model"] == model].iloc[0]
            values = [row["groundedness"], row["relevance"], row["coherence"]]
            values.append(values[0])  # Close the radar
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=model
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Capabilities Radar"
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ============================================
# TAB 2: TRACE ANALYSIS
# ============================================

with tab2:
    st.subheader("Individual Trace Analysis")
    
    for trace in SAMPLE_TRACES:
        with st.expander(f"{trace['verdict']} {trace['id']}: {trace['query'][:50]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Query:**")
                st.info(trace["query"])
                
                st.markdown("**Retrieved Context:**")
                st.warning(trace["context"])
                
                st.markdown("**Model Response:**")
                st.success(trace["response"])
            
            with col2:
                st.metric("Groundedness Score", f"{trace['groundedness_score']:.0%}")
                
                if trace["groundedness_score"] >= 0.8:
                    st.success(" Well Grounded")
                elif trace["groundedness_score"] >= 0.5:
                    st.warning("️ Partially Grounded")
                else:
                    st.error(" Hallucination Detected")
                
                st.markdown("**Analysis:**")
                st.write(trace["reasoning"])


# ============================================
# TAB 3: TRENDS
# ============================================

with tab3:
    st.subheader("Performance Trends Over Time")
    
    # Generate fake time series
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    trend_data = pd.DataFrame({
        "date": dates,
        "groundedness": [0.85 + 0.08 * (i/30) + 0.02 * (i % 5 - 2) for i in range(30)],
        "relevance": [0.82 + 0.05 * (i/30) + 0.03 * (i % 3 - 1) for i in range(30)],
        "coherence": [0.88 + 0.04 * (i/30) + 0.01 * (i % 7 - 3) for i in range(30)]
    })
    
    fig = px.line(
        trend_data.melt(id_vars=["date"], value_vars=["groundedness", "relevance", "coherence"]),
        x="date",
        y="value",
        color="variable",
        title="Evaluation Metrics Over Time",
        labels={"value": "Score", "variable": "Metric", "date": "Date"}
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Improvement suggestions
    st.subheader(" Improvement Recommendations")
    
    recommendations = [
        {"priority": " High", "issue": "Vague contract terms causing hallucinations", 
         "solution": "Add explicit few-shot examples for ambiguous clauses"},
        {"priority": "🟡 Medium", "issue": "Latency variance in peak hours",
         "solution": "Implement response caching for common queries"},
        {"priority": "🟢 Low", "issue": "Minor coherence drops on multi-step reasoning",
         "solution": "Consider chain-of-thought prompting for complex questions"}
    ]
    
    for rec in recommendations:
        st.markdown(f"**{rec['priority']}**: {rec['issue']}")
        st.markdown(f"  → *Solution:* {rec['solution']}")


# ============================================
# TAB 4: EXPORT
# ============================================

with tab4:
    st.subheader("Export Evaluation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Download Data")
        
        csv = eval_data.to_csv(index=False)
        st.download_button(
            label=" Download CSV",
            data=csv,
            file_name=f"hallucination_eval_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        json_data = json.dumps(SAMPLE_TRACES, indent=2)
        st.download_button(
            label=" Download Traces (JSON)",
            data=json_data,
            file_name=f"eval_traces_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col2:
        st.markdown("###  Generate Report")
        
        report_format = st.selectbox("Report Format", ["Markdown", "HTML", "PDF (Coming Soon)"])
        
        if st.button("Generate Report"):
            report = f"""
# Hallucination Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary
- Total Test Cases: {len(eval_data)}
- Average Groundedness: {eval_data['groundedness'].mean():.1%}
- Average Relevance: {eval_data['relevance'].mean():.1%}
- Average Coherence: {eval_data['coherence'].mean():.1%}
- Hallucination Rate: {(eval_data['groundedness'] < threshold).mean():.1%}

## Models Evaluated
{', '.join(eval_data['model'].unique())}

## Recommendations
1. Continue monitoring groundedness on contract clauses
2. Consider fine-tuning for domain-specific terminology
3. Implement human-in-the-loop for high-stakes decisions
            """
            
            st.markdown(report)
            st.download_button(
                label=" Download Report",
                data=report,
                file_name=f"eval_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )


# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <strong>Hallucination Hunter</strong> | Powered by Streamlit & Plotly<br>
    <a href="https://davidfernandez.dev">David Fernandez</a> | Industrial AI Engineer
</div>
""", unsafe_allow_html=True)
