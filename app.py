# ============================================
# MCA Insights Engine - AI Powered Dashboard (Groq AI)
# ============================================

import os
import re
import traceback
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from groq import Groq

# --------------------------------------------
# CONSTANTS & CONFIG
# --------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PAGE_CONFIG = dict(
    page_title="MCA Insights Engine",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSV_FILES = {
    "master": "MCA_Master_Assignment.csv",
    "day2_log": "Daily_Change_Log_Day2.csv",
    "day3_log": "Daily_Change_Log_Day3.csv",
    "enriched": "Enriched_Companies.csv",
}

# --------------------------------------------
# STREAMLIT SETUP
# --------------------------------------------
st.set_page_config(**PAGE_CONFIG)

# --------------------------------------------
# STYLES
# --------------------------------------------
st.markdown(
    """
<style>
/* Main Background */
.main { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem 2rem; 
}

/* Headers */
h1 { 
    color: #ffffff !important; 
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    font-size: 2.5rem !important;
}
h2, h3 { 
    color: #ffffff !important; 
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

/* Metric Cards */
.stMetric { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px; 
    border-radius: 15px; 
    color: #ffffff !important;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.2);
}
.stMetric label {
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: bold !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] { 
    gap: 8px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 55px;
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    border-radius: 12px 12px 0px 0px;
    padding: 12px 24px;
    color: #1a1a2e !important;
    font-weight: bold;
    font-size: 1rem;
    border: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: #ffffff !important;
}

/* DataFrame Styling */
.dataframe {
    background-color: rgba(255,255,255,0.95) !important;
    border-radius: 10px;
}
.dataframe thead th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #ffffff !important;
    font-weight: bold !important;
    padding: 12px !important;
}
.dataframe tbody tr:hover {
    background-color: rgba(102, 126, 234, 0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 10px 24px;
    border-radius: 8px;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}
.css-1d391kg .stSelectbox label, 
[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Error & Success Boxes */
.error-box {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
    color: white;
    border-left: 5px solid #ff0000;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.success-box {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    color: #1a1a2e;
    border-left: 5px solid #00d084;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    font-weight: 600;
}

/* Info boxes */
.stAlert {
    background-color: rgba(255,255,255,0.9);
    border-radius: 8px;
}

/* Text areas and inputs */
.stTextArea textarea, .stTextInput input {
    background-color: rgba(255,255,255,0.95) !important;
    border-radius: 8px !important;
    border: 2px solid rgba(102, 126, 234, 0.3) !important;
    color: #000000 !important;
    font-size: 16px !important;
}
.stTextArea textarea::placeholder {
    color: #666666 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: rgba(255,255,255,0.1);
    border-radius: 8px;
    color: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------
# GROQ CLIENT INIT 
# --------------------------------------------
def init_groq(api_key: str):
    if not api_key:
        st.warning("⚠️ GROQ_API_KEY not found in environment variables. AI features will be disabled.")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {e}")
        return None

client = init_groq(GROQ_API_KEY)

# --------------------------------------------
# HELPERS 
# --------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to lowercase with underscores."""
    try:
        df = df.copy()
        df.columns = (
            df.columns.str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("/", "_", regex=False)
            .str.lower()
        )
        return df
    except Exception as e:
        st.error(f"❌ Error normalizing columns: {e}")
        return df

def sanitize_code(code_string: str) -> str:
    """Clean and sanitize AI-generated code."""
    try:
        clean = code_string.replace("```python", "").replace("```", "")
        replacements = {
            "-": "-",  
            "–": "-", 
            "—": "-", 
            "−": "-",  
            "\r": "",
            "\u2018": "'",
            "\u2019": "'",
            "\u201C": '"',
            "\u201D": '"',
        }
        for old, new in replacements.items():
            clean = clean.replace(old, new)
        clean = re.sub(r'[^\x00-\x7F]+', '-', clean)
        return clean.strip()
    except Exception as e:
        st.error(f"❌ Error sanitizing code: {e}")
        return code_string

def safe_execute_code(code: str, local_vars: dict, description: str = "code"):
    """Safely execute code with comprehensive error handling (keeps original behavior)."""
    try:
        exec_globals = {}
        exec(code, exec_globals, local_vars)

        result = local_vars.get("result", None)

        # Fallback: find last meaningful output
        if result is None:
            for val in reversed(list(local_vars.values())):
                if isinstance(val, (pd.DataFrame, pd.Series, int, float, str, list, dict)):
                    result = val
                    break
        return True, result, None
    except SyntaxError as e:
        error_msg = f"Syntax Error: {str(e)}\nLine {e.lineno}: {e.text}"
        return False, None, error_msg
    except NameError as e:
        error_msg = f"Name Error: {str(e)}\nPossible undefined variable or column name."
        return False, None, error_msg
    except KeyError as e:
        error_msg = f"Key Error: {str(e)}\nColumn or key not found in data."
        return False, None, error_msg
    except Exception as e:
        error_msg = f"Execution Error: {str(e)}\n{traceback.format_exc()}"
        return False, None, error_msg

def styled_plotly(fig: px.scatter, title: str):
    """Apply the custom plot style used in the app."""
    fig.update_layout(
        title=title,
        title_font=dict(color='#F1F1F1', size=20, family='Poppins'),
        plot_bgcolor='rgba(20, 20, 30, 0.7)',
        paper_bgcolor='rgba(20, 20, 30, 0.7)',
        font=dict(color='#E8E8E8', size=13),
        xaxis=dict(
            title_font_color='#E8E8E8',
            tickfont_color='#E8E8E8',
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title_font_color='#E8E8E8',
            tickfont_color='#E8E8E8',
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white', size=12))
    )
    return fig

# --------------------------------------------
# DATA LOADING
# --------------------------------------------
@st.cache_data(ttl=3600)
def load_data(files_map):
    loaded = {}
    errors = []
    for key, filename in files_map.items():
        if not os.path.exists(filename):
            errors.append(f"⚠️ File not found: {filename}")
            loaded[key] = None
            continue
        try:
            if key == "master":
                usecols = [
                    "CIN", "CompanyName", "CompanyStateCode", "CompanyStatus",
                    "AuthorizedCapital", "PaidupCapital", "CompanyClass",
                    "CompanyIndustrialClassification", "CompanyROCcode",
                    "CompanyCategory", "CompanyRegistrationdate_date"
                ]
                df = pd.read_csv(filename, usecols=usecols, low_memory=False)
            else:
                df = pd.read_csv(filename, low_memory=False)
            df = normalize_columns(df)
            loaded[key] = df
        except Exception as e:
            errors.append(f"❌ Error loading {filename}: {e}")
            loaded[key] = None
    return loaded, errors

with st.spinner("📂 Loading data..."):
    data_map, load_errors = load_data(CSV_FILES)
    master = data_map.get("master")
    day2_log = data_map.get("day2_log")
    day3_log = data_map.get("day3_log")
    enriched = data_map.get("enriched")

# Show load issues if any (same UX)
if load_errors:
    with st.expander("⚠️ Data Loading Issues", expanded=False):
        for e in load_errors:
            st.write(e)

# --------------------------------------------
# HEADER
# --------------------------------------------
st.title("🏢 MCA Insights Engine - AI Powered Dashboard")
st.markdown("### Smart company data explorer combining AI intelligence with interactive visual analytics")
st.markdown("---")

# --------------------------------------------
# SIDEBAR FILTERS 
# --------------------------------------------
with st.sidebar:
    st.header("🔍 Filters & Navigation")
    if master is not None:
        try:
            states = ["All"] + sorted(master["companystatecode"].dropna().unique().tolist())
            selected_state = st.selectbox("🏙️ Select State", states)
            statuses = ["All"] + sorted(master["companystatus"].dropna().unique().tolist())
            selected_status = st.selectbox("🏷️ Company Status", statuses)
            classes = ["All"] + sorted(master["companyclass"].dropna().unique().tolist())
            selected_class = st.selectbox("🏛️ Company Class", classes)
            industries = ["All"] + sorted(master["companyindustrialclassification"].dropna().unique().tolist())
            selected_industry = st.selectbox("🏭 Industry", industries)
        except Exception as e:
            st.error(f"❌ Error creating filters: {e}")
            selected_state = selected_status = selected_class = selected_industry = "All"
    else:
        st.warning("⚠️ Master dataset not available. Filters disabled.")
        selected_state = selected_status = selected_class = selected_industry = "All"

# --------------------------------------------
# FILTER DATA 
# --------------------------------------------
if master is not None:
    try:
        filtered = master.copy()
        if selected_state != "All":
            filtered = filtered[filtered["companystatecode"] == selected_state]
        if selected_status != "All":
            filtered = filtered[filtered["companystatus"] == selected_status]
        if selected_class != "All":
            filtered = filtered[filtered["companyclass"] == selected_class]
        if selected_industry != "All":
            filtered = filtered[filtered["companyindustrialclassification"] == selected_industry]
    except Exception as e:
        st.error(f"❌ Error filtering data: {e}")
        filtered = master
else:
    filtered = None

# --------------------------------------------
# AI SUMMARY 
# --------------------------------------------
def groq_generate_summary(data: pd.DataFrame, label: str = "Day"):
    if not client:
        return "⚠️ Groq API client not initialized. Check your API key."
    try:
        prompt = f"""
        You are an MCA data analyst.
        Summarize the following change log data concisely in 3-5 bullet points:
        {data.head(25).to_string()}
        
        Focus on:
        - Total changes
        - Key trends
        - Notable patterns
        """
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return "### 🤖 AI Summary for " + label + "\n\n" + response.choices[0].message.content
    except Exception as e:
        return f"❌ Summary generation failed: {str(e)}\n\nPlease check your API key and internet connection."

# --------------------------------------------
# TABS
# --------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "📅 Daily Changes",
    "🌐 Enriched Data",
    "🤖 AI Assistant",
    "📈 Analytics",
    "🔍 Data Explorer"
])

# --------------------------------------------
# TAB 1: DASHBOARD 
# --------------------------------------------
with tab1:
    st.header("📊 MCA Master Dataset Overview")
    if filtered is not None:
        try:
            st.markdown("""
            <div style="
                background: linear-gradient(145deg, rgba(0,0,0,0.8), rgba(30,30,30,0.8));
                border-radius:20px;
                padding:25px;
                margin-bottom:25px;
                box-shadow:0 0 25px rgba(0,255,150,0.15);
                ">
            """, unsafe_allow_html=True)

            # KPI Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Companies", f"{len(filtered):,}")
            with c2:
                active_count = len(filtered[filtered['companystatus'] == 'Active'])
                st.metric("Active Companies", f"{active_count:,}")
            with c3:
                try:
                    avg_cap = filtered['authorizedcapital'].astype(float).mean()
                    st.metric("Avg. Authorized Capital", f"₹{avg_cap:,.0f}")
                except Exception:
                    st.metric("Avg. Authorized Capital", "N/A")
            with c4:
                state_count = filtered["companystatecode"].nunique()
                st.metric("States Covered", state_count)

            st.markdown("---")

            col1, col2 = st.columns(2)

            # Left Chart: Top 10 States
            with col1:
                try:
                    top_states = filtered["companystatecode"].value_counts().head(10)
                    fig = px.bar(
                        x=top_states.index,
                        y=top_states.values,
                        title="🏙️ Top 10 States by Company Count",
                        labels={"x": "State", "y": "Count"},
                        text=top_states.values,
                        color=top_states.values,
                        color_continuous_scale=px.colors.sequential.Tealgrn
                    )
                    fig.update_traces(
                        texttemplate='%{text:,}',
                        textposition='outside',
                        marker_line_color='rgba(255,255,255,0.3)',
                        marker_line_width=1.2
                    )
                    fig = styled_plotly(fig, "🏙️ Top 10 States by Company Count")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error creating state chart: {e}")

            # Right Chart: Status Distribution
            with col2:
                try:
                    status_counts = filtered["companystatus"].value_counts()
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="📊 Company Status Distribution",
                        hole=0.45,
                        color_discrete_sequence=[
                            '#00C2A8', '#00E5FF', '#F39C12', '#E74C3C', '#9B59B6',
                            '#1ABC9C', '#2ECC71', '#3498DB', '#95A5A6', '#16A085'
                        ]
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        textfont=dict(color='black', size=13),
                        hoverinfo='label+percent',
                        marker=dict(line=dict(color='rgba(255,255,255,0.2)', width=1))
                    )
                    fig = styled_plotly(fig, "📊 Company Status Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error creating status chart: {e}")

            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error rendering dashboard: {e}")
    else:
        st.warning("⚠️ Master data not loaded. Please check the data files.")

# --------------------------------------------
# TAB 2: DAILY CHANGES
# --------------------------------------------
with tab2:
    st.header("📅 Daily Change Logs & AI Summary")
    selected_day = st.radio("Select Log to View:", ["Day 2", "Day 3"])
    data = day2_log if selected_day == "Day 2" else day3_log

    if data is not None:
        try:
            st.subheader(f"📘 Change Log - {selected_day}")
            st.metric("Total Changes", len(data))
            st.dataframe(data.head(50), use_container_width=True)

            if st.button(f"🧠 Generate AI Summary for {selected_day}"):
                with st.spinner("🤖 Analyzing change patterns..."):
                    summary = groq_generate_summary(data, selected_day)
                    st.markdown(summary)
        except Exception as e:
            st.error(f"❌ Error displaying change log: {e}")
    else:
        st.info(f"⚠️ Change log for {selected_day} not found.")

# --------------------------------------------
# TAB 3: ENRICHED DATA
# --------------------------------------------
with tab3:
    st.header("🌐 Enriched Company Data")
    if enriched is not None:
        try:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Enriched Records", len(enriched))
            with c2:
                if 'status' in enriched.columns:
                    active = len(enriched[enriched["status"] == "Active"])
                    st.metric("Active Companies", active)
            st.dataframe(enriched.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error displaying enriched data: {e}")
    else:
        st.warning("⚠️ Enriched data not found. Please upload Enriched_Companies.csv")

# --------------------------------------------
# TAB 4: AI ASSISTANT 
# --------------------------------------------
with tab4:
    st.header("🤖 AI Query Assistant - Multi-Dataset")
    with st.expander("💡 Example Queries", expanded=False):
        st.markdown("""
        **Try asking:**
        - Show new incorporations in Maharashtra
        - List companies with authorized capital > 10 lakh
        - How many companies were struck off?
        - Show top 5 states by company count
        - Find companies in the Finance sector
        """)

    query = st.text_area("💬 Ask your question:", height=100,
                         placeholder="Example: How many active companies are in Karnataka?")
    ask_button = st.button("🚀 Ask AI", type="primary", use_container_width=True)

    if ask_button and query:
        if not client:
            st.error("❌ AI features unavailable. Please set GROQ_API_KEY in your .env file.")
        elif master is None:
            st.warning("⚠️ Master dataset not loaded. Cannot process query.")
        else:
            with st.spinner("🧠 Analyzing your question..."):
                try:
                    latest_data = day3_log.copy() if day3_log is not None else pd.DataFrame()
                    status_values = master['companystatus'].dropna().unique().tolist()[:10]
                    state_values = master['companystatecode'].dropna().unique().tolist()[:10]

                    combined_prompt = f"""
You are an MCA data analyst. You have access to these DataFrames:
- master: Main company dataset (shape: {master.shape})
- day2_log: Day 2 change log
- day3_log: Day 3 change log
- enriched: Enriched company data
- latest_data: Copy of day3_log

IMPORTANT DATA CONTEXT:
- All column names are lowercase with underscores
- Key columns: 'cin', 'companystatecode', 'companystatus', 'companyname', 'authorizedcapital', 'paidupCapital'
- Actual status values in data: {status_values}
- Actual state values in data: {state_values}
- For "struck off" or "striked off" queries, use 'Strike Off' (exact match)
- For "active" queries, use 'Active' (exact match)
- State codes are lowercase (e.g., 'maharashtra', 'delhi', 'karnataka')

Generate Python pandas code to answer this query: {query}

Requirements:
1. Store the final result in a variable named 'result'
2. Use exact string matches from the data context above
3. For counts, use .shape[0] or len()
4. For filtering, use == for exact matches (case-sensitive after checking data)
5. Handle missing values with .dropna() if needed
6. Return a DataFrame, Series, or scalar value

Return ONLY the Python code, no explanations.
"""

                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": combined_prompt}],
                        temperature=0.3,
                        max_tokens=1000
                    )

                    code = response.choices[0].message.content.strip()
                    clean_code = sanitize_code(code)

                    safe_locals = {
                        "master": master.copy(),
                        "day2_log": day2_log.copy() if day2_log is not None else pd.DataFrame(),
                        "day3_log": day3_log.copy() if day3_log is not None else pd.DataFrame(),
                        "enriched": enriched.copy() if enriched is not None else pd.DataFrame(),
                        "latest_data": latest_data,
                        "pd": pd,
                        "np": np
                    }

                    success, result, error = safe_execute_code(clean_code, safe_locals, "AI query")

                    if success:
                        st.markdown('<div class="success-box">✅ Query executed successfully!</div>',
                                    unsafe_allow_html=True)

                        if isinstance(result, pd.DataFrame):
                            st.markdown(f'<div style="color:green; font-weight:bold;">📊 Found {len(result):,} results</div>', unsafe_allow_html=True)
                            st.dataframe(result.head(100), use_container_width=True)
                            csv = result.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download Full Results",
                                csv,
                                "query_results.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        elif isinstance(result, pd.Series):
                            st.success("📊 Results:")
                            st.dataframe(result.to_frame(), use_container_width=True)
                        else:
                            if isinstance(result, (int, float)):
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.metric("Answer", f"{result:,}" if isinstance(result, int) else f"{result:.2f}")
                            else:
                                st.success(f"✅ Answer: **{result}**")
                    else:
                        st.markdown(f'<div class="error-box"><strong>❌ Execution Error</strong><br>{error}</div>',
                                    unsafe_allow_html=True)
                        with st.expander("🔍 View Generated Code (Debug)", expanded=False):
                            st.code(clean_code, language="python")
                        st.info("💡 Try rephrasing your question or check the Data Explorer tab for column names.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.info("💡 Try using the example queries or check the Data Explorer tab.")

# --------------------------------------------
# TAB 5: TREND ANALYTICS
# --------------------------------------------
with tab5:
    st.header("📈 Trend Analytics")
    if master is not None:
        try:
            master_copy = master.copy()
            master_copy['regdate'] = pd.to_datetime(master_copy['companyregistrationdate_date'], errors='coerce')
            valid_dates = master_copy.dropna(subset=['regdate'])

            if len(valid_dates) > 0:
                yearly = valid_dates.groupby(valid_dates['regdate'].dt.year).size().reset_index(name='count')
                yearly = yearly[yearly['regdate'] > 1900]

                fig = px.line(
                    yearly, x='regdate', y='count',
                    title="Company Registrations Over Time",
                    labels={"regdate": "Year", "count": "Number of Companies"},
                    markers=True, line_shape='spline',
                    color_discrete_sequence=['#ff66b3']
                )
                fig.update_layout(hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    peak_year = yearly.loc[yearly['count'].idxmax(), 'regdate']
                    st.metric("Peak Registration Year", int(peak_year))
                with col2:
                    total_reg = yearly['count'].sum()
                    st.metric("Total Registrations", f"{total_reg:,}")
                with col3:
                    avg_per_year = yearly['count'].mean()
                    st.metric("Avg. per Year", f"{avg_per_year:,.0f}")
            else:
                st.warning("⚠️ No valid registration dates found in the dataset.")
        except Exception as e:
            st.error(f"❌ Error creating trend analysis: {e}")
    else:
        st.warning("⚠️ No data available for trend analysis.")

# --------------------------------------------
# TAB 6: DATA EXPLORER
# --------------------------------------------
with tab6:
    st.header("🔍 Data Explorer & Debugger")
    st.info("💡 Use this tab to explore actual values in your dataset")

    if master is not None:
        try:
            st.subheader("📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(master):,}")
            with col2:
                st.metric("Total Columns", len(master.columns))
            with col3:
                st.metric("Memory Usage", f"{master.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            st.markdown("---")

            selected_col = st.selectbox("🔍 Select Column to Explore", master.columns.tolist())

            if selected_col:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"📋 Unique Values in '{selected_col}'")
                    unique_vals = master[selected_col].dropna().unique()
                    st.metric("Unique Count", len(unique_vals))
                    if len(unique_vals) <= 100:
                        st.write("All unique values:")
                        for val in sorted(unique_vals):
                            st.write(f"- `{val}`")
                    else:
                        st.write(f"Top 50 unique values (out of {len(unique_vals)}):")
                        for val in sorted(unique_vals)[:50]:
                            st.write(f"- `{val}`")
                with col2:
                    st.subheader(f"📊 Value Counts for '{selected_col}'")
                    value_counts = master[selected_col].value_counts().head(20)
                    st.dataframe(value_counts, use_container_width=True)
                    if len(value_counts) <= 20:
                        fig = px.bar(
                            x=value_counts.index.astype(str),
                            y=value_counts.values,
                            title=f"Distribution of '{selected_col}'",
                            labels={"x": selected_col, "y": "Count"}
                        )
                        st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("🔍 Quick Data Checks")
            check1, check2, check3 = st.columns(3)
            with check1:
                st.write("**Company Status Values:**")
                for status in master['companystatus'].dropna().unique()[:10]:
                    count = len(master[master['companystatus'] == status])
                    st.write(f"- `{status}`: {count:,}")
            with check2:
                st.write("**State Values:**")
                for state in master['companystatecode'].dropna().unique()[:10]:
                    count = len(master[master['companystatecode'] == state])
                    st.write(f"- `{state}`: {count:,}")
            with check3:
                st.write("**Company Class Values:**")
                for cls in master['companyclass'].dropna().unique()[:10]:
                    count = len(master[master['companyclass'] == cls])
                    st.write(f"- `{cls}`: {count:,}")

            st.markdown("---")
            st.subheader("👀 Sample Data Preview")
            st.dataframe(master.head(100), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error in data explorer: {e}")
    else:
        st.warning("⚠️ Master dataset not loaded.")

# --------------------------------------------
# FOOTER 
# --------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4b0082;'> MCA Insights Engine | "
    "Built with Streamlit + Groq AI | © 2025</div>",
    unsafe_allow_html=True
)
