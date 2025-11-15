import streamlit as st
import pandas as pd
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Project Janus: Advanced Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Data Loading ---
@st.cache_data
def load_all_data():
    """Loads all project CSVs from the data directory."""
    data = {}
    data_dir = "data"
    
    files = {
        "returns": "portfolio_log_returns.csv",
        "hybrid_var": "phase4_final_hybrid_var.csv",
        "advanced_vars": "phase3_advanced_var_timeseries.csv",
        "basic_vars": "phase1_all_risk_measures.csv",
        "hmm_signals": "phase4_hmm_signals.csv",
        "stress_test": "phase5_stress_test_summary.csv",
        "attribution": "phase6_attribution_summary.csv",
        "capital": "phase7_final_capital_requirement.csv"
    }

    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            try:
                if key in ["stress_test", "attribution"]:
                    data[key] = pd.read_csv(path, index_col=0)
                else:
                    data[key] = pd.read_csv(path, index_col=0, parse_dates=True)
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
                data[key] = None
        else:
            st.warning(f"⚠️ File not found: {filename}")
            data[key] = None
            
    return data

# Load data once
data = load_all_data()

# --- 3. Sidebar Navigation ---
st.sidebar.title("🛡️ Project Janus")
st.sidebar.caption("Regime-Aware Risk Engine")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Executive Summary",
    "Risk Models Analysis",
    "HMM Regime Detector",
    "Stress Testing",
    "Risk Attribution",
    "Regulatory Capital"
])

st.sidebar.markdown("---")
st.sidebar.info("v1.0 | Basel III Compliant")

# --- 4. Page Routing ---

if page == "Executive Summary":
    st.title("Executive Risk Summary")
    st.markdown("### Daily Risk Monitor")
    
    if data['returns'] is not None and data['hybrid_var'] is not None:
        # Get latest values safely
        latest_date = data['returns'].index[-1].strftime('%Y-%m-%d')
        latest_return = data['returns'].iloc[-1, 0]
        latest_var = data['hybrid_var'].iloc[-1, -1]
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Date", latest_date)
        col2.metric("Daily Return", f"{latest_return:.2%}")
        col3.metric("99% Hybrid VaR", f"{latest_var:.2%}", delta="Regime-Adjusted", delta_color="off")
        
        # --- RESTORED SIMPLE CHART ---
        st.subheader("Portfolio Performance vs. Risk Limit (Last 1 Year)")
        
        # Prepare clean DataFrame for st.line_chart
        # This ensures both lines are on the same index and valid floats
        recent_returns = data['returns'].iloc[-252:, 0]
        recent_var = data['hybrid_var'].iloc[-252:, -1]
        
        chart_data = pd.DataFrame({
            "Daily Returns": recent_returns,
            "99% Hybrid VaR": recent_var
        })
        
        # Streamlit native chart - handles colors/theme automatically
        st.line_chart(chart_data)
        
    else:
        st.error("Critical data missing. Please check the data folder.")

elif page == "Risk Models Analysis":
    st.title("Model Comparison")
    st.write("Coming soon...")

elif page == "HMM Regime Detector":
    st.title("Regime Detection")
    st.write("Coming soon...")

elif page == "Stress Testing":
    st.title("Stress Test Scenarios")
    st.write("Coming soon...")

elif page == "Risk Attribution":
    st.title("Risk Attribution")
    st.write("Coming soon...")

elif page == "Regulatory Capital":
    st.title("Capital Requirement")
    st.write("Coming soon...")