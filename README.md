# 🏢 MCA Insights Engine — AI Powered Dashboard (Groq AI + Streamlit)

An interactive, AI-driven data analytics dashboard for **Ministry of Corporate Affairs (MCA)** datasets.  
This app enables **company-level insights**, **trend analytics**, and **automated data enrichment** — all through a user-friendly Streamlit interface powered by **Groq AI (LLaMA-3 model)**.

---

## 🚀 Overview

**MCA Insights Engine** is a comprehensive data intelligence tool that:
- Analyzes MCA company datasets (`master`, `daily change logs`, `enriched`).
- Provides rich visual analytics and KPIs using Plotly.
- Supports natural language queries through **Groq’s LLaMA-3.3-70B Versatile model**.
- Identifies **new incorporations**, **state-level growth trends**, and **company class patterns**.
- Generates **AI-driven summaries** and performs **data enrichment** dynamically.

---

## 🧩 Project Architecture

```bash
📦 MCA_Insights_Engine/
├── app.py                     # Main Streamlit application (frontend + AI backend)
├── assignment.ipynb           # Notebook for data enrichment and transformation logic
├── data/                      # Folder containing all dataset files
│   ├── MCA_Master_Assignment.csv   # Master dataset (baseline)
│   ├── Daily_Change_Log_Day2.csv   # Incremental changes (Day 2)
│   ├── Daily_Change_Log_Day3.csv   # Incremental changes (Day 3)
│   └── Enriched_Companies.csv      # Output of enrichment logic
├── .env                       # Environment file containing GROQ_API_KEY
└── README.md                  # Documentation file (this file)
```

### 🧠 Core Technologies
| Layer | Tools |
|-------|--------|
| **Frontend/UI** | Streamlit, Plotly Express, HTML/CSS |
| **Backend Processing** | Pandas, NumPy |
| **AI Assistant** | Groq API (LLaMA 3.3 70B Versatile) |
| **Data Caching** | Streamlit’s `@st.cache_data` |
| **Visualization** | Plotly (bar, pie, line, distribution charts) |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/MCA-Insights-Engine.git
cd MCA-Insights-Engine
```
### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate          # On Windows
```
### 3️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```
If you don’t have a requirements.txt file yet:
```bash
pip install streamlit pandas numpy plotly python-dotenv groq
```
### 4️⃣ Add Environment Variables

Create a .env file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
You can generate an API key at https://console.groq.com.

### 5️⃣ Run the Application
```bash
streamlit run app.py
```
The app will be accessible at:
```bash
http://localhost:8501
```
# 🧠 Application Workflow

## 🔹 Step 1: Data Loading

The app automatically loads the following CSV files:

- `MCA_Master_Assignment.csv` — Baseline MCA master data  
- `Daily_Change_Log_Day2.csv` — Daily change log (Day 2)  
- `Daily_Change_Log_Day3.csv` — Daily change log (Day 3)  
- `Enriched_Companies.csv` — Output after enrichment pipeline  

All columns are normalized using:

```python
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
```
## 🔹 Step 2: Filtering and Visualization

Interactive filters allow users to select:

- **Company State**
- **Status** *(Active, Strike Off, etc.)*
- **Company Class**
- **Industry Classification**

---

### 📊 Key Charts Include:
- 🏙️ **Top 10 States by Company Count**  
- 📈 **Company Status Distribution**  
- 💰 **Authorized Capital Trends**

---

### 📋 KPIs Displayed:
- **Total Companies** — Overall count after filters applied  
- **Active Companies** — Count of companies with status “Active”  
- **Average Authorized Capital** — Mean authorized capital of filtered records  
- **States Covered** — Number of unique states present in the filtered dataset  

---

These filters and visualizations are rendered dynamically using **Streamlit widgets** and **Plotly charts**, allowing users to interactively explore MCA datasets in real time.

## 🔹 Step 3: AI Assistant (Groq-Powered)

Natural language queries are processed through **Groq’s LLaMA-3.3 model**, which dynamically generates executable **Python (Pandas)** code to answer data-driven questions.

---

### 💬 Example Queries
- “Show all active companies in Maharashtra.”  
- “List new incorporations from Day 3.”  
- “Find top 5 industries by company count.”  

---

### ⚙️ Execution Process

The AI assistant:
1. Takes the user’s natural language query.  
2. Sends it to the **Groq LLaMA-3.3 model** for interpretation.  
3. Receives dynamically generated Python code as output.  
4. Executes the code securely in a sandbox using the function:

```python
success, result, error = safe_execute_code(clean_code, safe_locals)
```
### 🧠 AI Summary Generation (Powered by Groq)

The app uses the **Groq LLaMA-3.3 model** to automatically analyze uploaded daily change logs.  
It generates concise, human-readable summaries in **3–5 bullet points**, focusing on:

- ✅ Total number of changes  
- 📈 Key trends or growth areas  
- 🏙️ State or sector-wise activity patterns  
- ⚙️ Notable data anomalies or sudden shifts  

---

### 🧾 AI Prompt Example
The underlying AI summary logic uses the following prompt internally:

> **Prompt:**  
> “Summarize the following change log data in 3–5 bullet points focusing on total changes, key trends, and notable patterns.”

---
## 🔹 Step 5: Enrichment Logic (from `assignment.ipynb`)

The **Enrichment Notebook** is responsible for detecting new incorporations, filtering based on state, and preparing an enriched dataset ready for analysis or visualization.

---

### 🧩 Workflow Overview

1️⃣ **Identify New CINs (Company Identification Numbers)**  
Compare the latest dataset (`latest_data`) with the baseline (`master`) to find newly registered companies:

```python
new_cins = set(latest_data['cin']).difference(set(master['cin']))
```
2️⃣ **Filter by State (Example: Maharashtra)**
Extract companies that belong to the selected state — in this case, Maharashtra (`state code: 'mh'`):
```python
mh_new = latest_data[
    (latest_data['cin'].isin(new_cins)) &
    (latest_data['companystatecode'].str.lower() == 'mh')
]
```
3️⃣ **Enrich & Merge**
Combine the filtered data with external or derived attributes such as industry classification, capital structure, or category data, and then export to CSV:
```python
mh_new.to_csv("Enriched_Companies.csv", index=False)
```
 4️⃣ **Validation & Quality Checks**

Ensures type consistency, removes duplicates, and maintains column normalization.

## 🔹 Step 6: Analytics & Trend Insights
---
The Analytics tab visualizes year-wise trends in company registrations:
```python
master_copy['regdate'] = pd.to_datetime(master_copy['companyregistrationdate_date'], errors='coerce')
yearly = master_copy.groupby(master_copy['regdate'].dt.year).size()
```
Displays
The Analytics tab provides key registration metrics derived from year-wise aggregation:
- 📈 **Total Registrations** — Overall number of companies registered across all years  
- 🏆 **Peak Year** — The year with the highest number of new company incorporations
---

## 🔹 Step 7: Data Explorer
Interactive exploration tab:
- View unique column values
- Check top categories
- Inspect sample data
- Run quick consistency checks for state, class, and status fields.


