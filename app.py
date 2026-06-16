import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="India Air Pollution Dashboard", page_icon="🌫️",
                   layout="wide", initial_sidebar_state="expanded")

# ── PUT YOUR FREE WAQI TOKEN HERE ───────────────────────────────────────────
# Get one free at: https://aqicn.org/data-platform/token/
WAQI_TOKEN = st.secrets.get("WAQI_TOKEN", "YOUR_WAQI_TOKEN_HERE")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252940);
        border: 1px solid #3d4163; border-radius: 12px; padding: 16px;
    }
    [data-testid="metric-container"] label { color: #a0aec0 !important; font-size: 13px !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #e2e8f0 !important; font-size: 28px !important; font-weight: 700 !important;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 24px; font-weight: 700; margin-bottom: 8px;
    }
    .best-model-tag {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white; padding: 4px 10px; border-radius: 10px;
        font-size: 11px; font-weight: 700; margin-left: 6px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f35 0%, #0f1117 100%);
        border-right: 1px solid #2d3561;
    }
    .stTabs [data-baseweb="tab-list"] { background: #1e2130; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #a0aec0; border-radius: 8px; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important; color: white !important;
    }
    .info-box { background: #1e2130; border-left: 4px solid #667eea;
                border-radius: 8px; padding: 12px 16px; margin: 8px 0; color: #e2e8f0; }
    .good-box { border-left-color: #48bb78; }
    .moderate-box { border-left-color: #ecc94b; }
    .poor-box { border-left-color: #fc8181; }
    .severe-box { border-left-color: #9f7aea; }
</style>
""", unsafe_allow_html=True)

BUCKET_ORDER  = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BUCKET_COLORS = ["#48bb78", "#68d391", "#ecc94b", "#fc8181", "#f56565", "#9f7aea"]
POLLUTANTS    = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
PLOTLY_DARK   = dict(
    paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
    font=dict(color="#e2e8f0"),
    xaxis=dict(gridcolor="#2d3561", linecolor="#3d4163"),
    yaxis=dict(gridcolor="#2d3561", linecolor="#3d4163"),
)

# ── CPCB OFFICIAL AQI BREAKPOINT TABLES (India standard) ────────────────────
# Format per pollutant: (Concentration_low, Concentration_high, AQI_low, AQI_high)
AQI_BREAKPOINTS = {
    'PM2.5': [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),
              (121,250,301,400),(251,380,401,500)],
    'PM10':  [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),
              (351,430,301,400),(431,520,401,500)],
    'NO2':   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),
              (281,400,301,400),(401,520,401,500)],
    'SO2':   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),
              (801,1600,301,400),(1601,2100,401,500)],
    'CO':    [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),
              (17.1,34,301,400),(34.1,50,401,500)],
    'O3':    [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),
              (209,748,301,400),(749,1000,401,500)],
}

def calc_subindex(value, breakpoints):
    """Linear interpolation within the matching breakpoint band."""
    if pd.isna(value):
        return np.nan
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= value <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (value - c_lo) + i_lo
    last = breakpoints[-1]
    if value > last[1]:
        return 500.0  # cap at max AQI
    if value < 0:
        return 0.0
    return np.nan

def compute_cpcb_aqi(row):
    """Overall AQI = MAX of individual pollutant sub-indices (CPCB rule)."""
    sub_indices = []
    for pollutant, bps in AQI_BREAKPOINTS.items():
        if pollutant in row.index and pd.notna(row[pollutant]):
            si = calc_subindex(row[pollutant], bps)
            if pd.notna(si):
                sub_indices.append(si)
    if not sub_indices:
        return np.nan
    return max(sub_indices)

# ── HELPERS ───────────────────────────────────────────────────────────────────
CANDIDATE_DATA_FILES = ["air_quality.csv", "city_day.csv"]  # checks both names
DATA_PATH = next((f for f in CANDIDATE_DATA_FILES if os.path.exists(f)), CANDIDATE_DATA_FILES[0])

@st.cache_data(show_spinner="Loading dataset...")
def load_and_preprocess(source):
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()

    # AQI — CPCB official breakpoint formula: AQI = MAX of pollutant sub-indices
    avail_p = [p for p in POLLUTANTS if p in df.columns]
    df['AQI'] = df[avail_p].apply(compute_cpcb_aqi, axis=1)
    df = df.dropna(subset=['AQI'])

    def categorize_aqi(aqi):
        if pd.isna(aqi): return np.nan
        if aqi <= 50:    return "Good"
        if aqi <= 100:   return "Satisfactory"
        if aqi <= 200:   return "Moderate"
        if aqi <= 300:   return "Poor"
        if aqi <= 400:   return "Very Poor"
        return "Severe"
    df['AQI_Bucket'] = df['AQI'].apply(categorize_aqi)

    date_col = next((c for c in ['Date','Datetime','date','datetime'] if c in df.columns), None)
    if date_col:
        df['Date']   = pd.to_datetime(df[date_col], errors='coerce')
        df['Year']   = df['Date'].dt.year
        df['Month']  = df['Date'].dt.month
        df['Season'] = df['Month'].map({
            12:"Winter",1:"Winter",2:"Winter",
            3:"Spring",4:"Spring",5:"Spring",
            6:"Summer",7:"Summer",8:"Summer",
            9:"Autumn",10:"Autumn",11:"Autumn"
        })

    df = df.drop_duplicates().ffill().bfill()
    return df


@st.cache_resource(show_spinner="Training ML models...")
def train_all_models(_df_id, df):
    avail_p = [p for p in POLLUTANTS if p in df.columns]
    subset  = df[avail_p + ['AQI', 'AQI_Bucket']].dropna()
    subset  = subset[subset['AQI_Bucket'].isin(BUCKET_ORDER)]

    X     = subset[avail_p]
    y_reg = subset['AQI']

    le    = LabelEncoder()
    le.fit(subset['AQI_Bucket'])
    y_cls = le.transform(subset['AQI_Bucket'])

    X_train, X_test, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _,       _,      yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42)

    clf_models = {
        "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":      DecisionTreeClassifier(random_state=42),
    }
    clf_results = {}
    for name, mdl in clf_models.items():
        mdl.fit(X_train, yc_tr)
        preds = mdl.predict(X_test)
        clf_results[name] = {
            "model":    mdl,
            "accuracy": round(accuracy_score(yc_te, preds) * 100, 2),
            "preds":    preds,
            "y_test":   yc_te,
            "cm":       confusion_matrix(yc_te, preds),
        }

    reg_models = {
        "Random Forest Reg": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
    }
    reg_results = {}
    for name, mdl in reg_models.items():
        mdl.fit(X_train, yr_tr)
        preds = mdl.predict(X_test)
        reg_results[name] = {
            "model": mdl,
            "mae":   round(mean_absolute_error(yr_te, preds), 2),
            "r2":    round(r2_score(yr_te, preds) * 100, 2),
            "preds": preds, "y_test": yr_te,
        }

    best_name = max(clf_results, key=lambda k: clf_results[k]["accuracy"])

    # Compute 1st-99th percentile bounds per pollutant — used to clip live
    # API inputs so the model doesn't extrapolate on out-of-distribution data.
    clip_bounds = {p: (float(subset[p].quantile(0.01)), float(subset[p].quantile(0.99)))
                    for p in avail_p}

    return clf_results, reg_results, best_name, clf_results[best_name]["model"], le, avail_p, clip_bounds


def get_aqi_color(b):
    return {"Good":"#48bb78","Satisfactory":"#68d391","Moderate":"#ecc94b",
            "Poor":"#fc8181","Very Poor":"#f56565","Severe":"#9f7aea"}.get(b,"#a0aec0")

def get_bucket_label(aqi):
    if aqi<=50:  return "Good","🟢"
    if aqi<=100: return "Satisfactory","🟩"
    if aqi<=200: return "Moderate","🟡"
    if aqi<=300: return "Poor","🟠"
    if aqi<=400: return "Very Poor","🔴"
    return "Severe","🟣"

def clip_to_training_range(values_dict, clip_bounds):
    """Clip live/forecast pollutant values to the 1st-99th percentile range
    seen during training, so the regressor doesn't extrapolate wildly on
    out-of-distribution live API values."""
    clipped = {}
    for p, v in values_dict.items():
        if p in clip_bounds:
            lo, hi = clip_bounds[p]
            clipped[p] = float(np.clip(v, lo, hi))
        else:
            clipped[p] = v
    return clipped

def aqi_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"font":{"size":36,"color":"#e2e8f0"}},
        gauge={"axis":{"range":[0,500],"tickcolor":"#a0aec0"},
               "bar":{"color":"#667eea","thickness":0.3},
               "bgcolor":"#1e2130","bordercolor":"#3d4163",
               "steps":[{"range":[0,50],"color":"#48bb78"},{"range":[50,100],"color":"#68d391"},
                        {"range":[100,200],"color":"#ecc94b"},{"range":[200,300],"color":"#fc8181"},
                        {"range":[300,400],"color":"#f56565"},{"range":[400,500],"color":"#9f7aea"}],
               "threshold":{"line":{"color":"white","width":3},"thickness":0.85,"value":value}}
    ))
    fig.update_layout(paper_bgcolor="#0f1117",plot_bgcolor="#0f1117",
                      height=240,margin=dict(t=20,b=10,l=30,r=30))
    return fig


# ── LIVE DATA HELPERS (NEW) ─────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Fetching live air quality data...")
def fetch_live_data(city_name):
    """Fetch live AQI + forecast data for a city from the WAQI API."""
    try:
        url = f"https://api.waqi.info/feed/{city_name}/?token={WAQI_TOKEN}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("status") != "ok":
            return None
        return data["data"]
    except Exception:
        return None


def map_iaqi_to_pollutants(iaqi):
    """Map WAQI's iaqi keys to our column names.

    NOTE: CO is intentionally excluded here. WAQI's live 'co' value uses a
    different unit/scale than city_day.csv's CO column (mg/m3), causing the
    regressor to see an out-of-distribution CO and over-predict AQI. PM2.5,
    PM10, NO2, SO2, O3 are reliable and dominate the Indian AQI anyway, so CO
    is left out here and falls back to the dataset average automatically.
    """
    mapping = {"PM2.5":"pm25","PM10":"pm10","NO2":"no2","SO2":"so2","O3":"o3"}
    result = {}
    for our_name, waqi_key in mapping.items():
        if waqi_key in iaqi and "v" in iaqi[waqi_key]:
            result[our_name] = iaqi[waqi_key]["v"]
    return result


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ India Air Pollution\n### Dashboard")
    st.divider()
    st.markdown("**Tabs:** Overview | EDA | City | ML Models | Predictor | Live & Forecast | Data")

# Try to auto-load the bundled dataset; fall back to manual upload only if missing
df_source = None
if os.path.exists(DATA_PATH):
    df_source = DATA_PATH
else:
    with st.sidebar:
        st.warning(f"⚠️ '{DATA_PATH}' not found next to app.py")
        uploaded = st.file_uploader("📂 Upload city_day.csv", type=["csv"])
    if uploaded is not None:
        df_source = uploaded

if df_source is None:
    st.markdown("""
    <div style='text-align:center;padding:60px 0 20px 0'>
        <h1 style='font-size:52px'>🌫️</h1>
        <h1 style='background:linear-gradient(90deg,#667eea,#764ba2);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:40px'>
            India Air Pollution Dashboard</h1>
        <p style='color:#a0aec0;font-size:17px'>Place <code>city_day.csv</code> next to <code>app.py</code>,
        or upload it from the sidebar.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

try:
    df = load_and_preprocess(df_source)
except Exception as e:
    st.error(f"❌ Error: {e}"); st.stop()
if df.empty:
    st.warning("⚠️ Dataset is empty."); st.stop()

# Sidebar filters
with st.sidebar:
    st.markdown("### ⚙️ Filters")
    cities = sorted(df['City'].unique()) if 'City' in df.columns else []
    sel_cities = st.multiselect("🏙️ Cities", cities, default=cities)
    if 'Year' in df.columns:
        years = sorted(df['Year'].dropna().unique().astype(int))
        year_range = st.slider("📅 Year", int(min(years)), int(max(years)),
                               (int(min(years)),int(max(years)))) if len(years)>1 else (years[0],years[0])
    else: year_range = None
    bucket_filter = st.multiselect("🎨 Categories", BUCKET_ORDER, default=BUCKET_ORDER)

fdf = df.copy()
if sel_cities and 'City' in fdf.columns: fdf = fdf[fdf['City'].isin(sel_cities)]
if year_range and 'Year' in fdf.columns: fdf = fdf[(fdf['Year']>=year_range[0])&(fdf['Year']<=year_range[1])]
if bucket_filter: fdf = fdf[fdf['AQI_Bucket'].isin(bucket_filter)]
if fdf.empty: st.warning("⚠️ No data matches filters."); st.stop()

# ── Train models upfront so accuracy is available on Overview too ──────────────
with st.spinner("⚙️ Training ML models (cached after first run)..."):
    clf_results, reg_results, best_name, best_clf, le, avail_p, clip_bounds = train_all_models(id(df), df)
best_acc = clf_results[best_name]["accuracy"]

tabs = st.tabs(["🏠 Overview","📊 EDA","🏙️ City Analysis","🤖 ML Models","🔮 Predictor","🌍 Live & Forecast","📋 Raw Data"])

# ════ TAB 1 OVERVIEW ══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-header">📊 Dashboard Overview</p>', unsafe_allow_html=True)
    avg_aqi = float(fdf['AQI'].mean())
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.metric("📁 Records",       f"{len(fdf):,}")
    with c2: st.metric("🏙️ Cities",        fdf['City'].nunique() if 'City' in fdf.columns else "N/A")
    with c3: st.metric("📅 Years",         fdf['Year'].nunique() if 'Year' in fdf.columns else "N/A")
    with c4: st.metric("💨 Avg AQI",       f"{avg_aqi:.1f}")
    with c5:
        worst = fdf.groupby('City')['AQI'].mean().idxmax() if 'City' in fdf.columns else "N/A"
        st.metric("⚠️ Most Polluted", worst)
    with c6: st.metric("🤖 Model Accuracy", f"{best_acc:.2f}%", help=f"Best model: {best_name}")
    st.divider()
    cg,cb = st.columns([1,2])
    with cg:
        st.markdown("#### 🎯 Average AQI Gauge")
        bv = str(fdf['AQI_Bucket'].mode()[0]); color = get_aqi_color(bv)
        st.plotly_chart(aqi_gauge(avg_aqi), use_container_width=True)
        st.markdown(f"<div style='text-align:center'><span style='background:{color};color:black;"
                    f"padding:6px 18px;border-radius:20px;font-weight:700'>{bv}</span></div>",
                    unsafe_allow_html=True)
    with cb:
        st.markdown("#### 🎨 AQI Category Distribution")
        bc = fdf['AQI_Bucket'].value_counts().reindex(BUCKET_ORDER).dropna()
        fig = px.bar(x=bc.index, y=bc.values, color=bc.index,
                     color_discrete_sequence=BUCKET_COLORS,
                     labels={"x":"Category","y":"Days"}, text=bc.values)
        fig.update_traces(textposition='outside',textfont_color='white')
        fig.update_layout(**PLOTLY_DARK, showlegend=False, height=280, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.dataframe(pd.DataFrame({
        "Category":BUCKET_ORDER,
        "AQI Range":["0-50","51-100","101-200","201-300","301-400","401-500"],
        "Health Impact":["Minimal","Sensitive discomfort","Moderate discomfort",
                         "Most affected","Respiratory illness","Emergency"],
        "Level":["🟢","🟩","🟡","🟠","🔴","🟣"]
    }), use_container_width=True, hide_index=True)

# ════ TAB 2 EDA ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(fdf, x='AQI', nbins=50, color_discrete_sequence=['#667eea'],
                           marginal="box", title="AQI Distribution")
        fig.update_layout(**PLOTLY_DARK, height=320, margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if 'Month' in fdf.columns:
            monthly = fdf.groupby('Month')['AQI'].mean().reset_index()
            monthly['Month_Name'] = monthly['Month'].map(
                {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
            fig = px.line(monthly, x='Month_Name', y='AQI', markers=True,
                          color_discrete_sequence=['#4fc3f7'], title="Monthly AQI Trend")
            fig.update_layout(**PLOTLY_DARK, height=320, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        avail_poll = [p for p in POLLUTANTS+['AQI'] if p in fdf.columns]
        if len(avail_poll)>=2:
            fig = px.imshow(fdf[avail_poll].corr().round(2), text_auto=True,
                            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            title="Correlation Heatmap")
            fig.update_layout(**PLOTLY_DARK, height=380, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        avail_p2 = [p for p in POLLUTANTS if p in fdf.columns]
        if avail_p2:
            ps = st.selectbox("Pollutant", avail_p2, key="ps")
            fig = px.box(fdf, x='City' if 'City' in fdf.columns else None, y=ps,
                         color='City' if 'City' in fdf.columns else None,
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title=f"{ps} by City")
            fig.update_layout(**PLOTLY_DARK, height=320, margin=dict(t=40,b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        if 'Season' in fdf.columns:
            fig = px.violin(fdf, x='Season', y='AQI', color='Season', box=True,
                            category_orders={"Season":["Winter","Spring","Summer","Autumn"]},
                            color_discrete_sequence=['#4299e1','#48bb78','#f6ad55','#fc8181'],
                            title="Seasonal AQI Pattern")
            fig.update_layout(**PLOTLY_DARK, height=340, margin=dict(t=40,b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if 'Year' in fdf.columns and 'City' in fdf.columns:
            yearly = fdf.groupby(['Year','City'])['AQI'].mean().reset_index()
            fig = px.line(yearly, x='Year', y='AQI', color='City', markers=True,
                          color_discrete_sequence=px.colors.qualitative.Set1,
                          title="Year-wise AQI Trend")
            fig.update_layout(**PLOTLY_DARK, height=340, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

# ════ TAB 3 CITY ANALYSIS ═════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-header">🏙️ City-wise Analysis</p>', unsafe_allow_html=True)
    if 'City' not in fdf.columns:
        st.warning("No City column.")
    else:
        cs = fdf.groupby('City')['AQI'].agg(['mean','min','max','std']).round(1).reset_index()
        cs.columns = ['City','Avg AQI','Min','Max','Std Dev']
        cs = cs.sort_values('Avg AQI', ascending=False)
        c1,c2 = st.columns([1.2,1])
        with c1:
            fig = px.bar(cs, x='Avg AQI', y='City', orientation='h',
                         color='Avg AQI', color_continuous_scale='RdYlGn_r',
                         text='Avg AQI', title="City AQI Ranking")
            fig.update_traces(textposition='outside',textfont_color='white')
            fig.update_layout(**PLOTLY_DARK, height=400, margin=dict(t=40,b=20), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Statistics")
            st.dataframe(cs, use_container_width=True, hide_index=True)
        bc2 = fdf.groupby(['City','AQI_Bucket']).size().reset_index(name='Count')
        bc2['Pct'] = bc2.groupby('City')['Count'].transform(lambda x: x/x.sum()*100)
        fig = px.bar(bc2, x='City', y='Pct', color='AQI_Bucket',
                     color_discrete_map=dict(zip(BUCKET_ORDER,BUCKET_COLORS)),
                     category_orders={"AQI_Bucket":BUCKET_ORDER},
                     labels={"Pct":"(%)","AQI_Bucket":"Category"},
                     title="Category Distribution per City", text_auto='.1f')
        fig.update_layout(**PLOTLY_DARK, height=400, barmode='stack', margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
        avail_p3 = [p for p in POLLUTANTS if p in fdf.columns]
        if avail_p3 and 'Month' in fdf.columns and 'PM2.5' in fdf.columns:
            heat = fdf.pivot_table(values='PM2.5', index='City', columns='Month', aggfunc='mean').round(1)
            mnms = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            heat.columns = [mnms[int(m)-1] for m in heat.columns]
            fig = px.imshow(heat, text_auto=True, color_continuous_scale='YlOrRd',
                            labels=dict(color="PM2.5"), title="PM2.5 Monthly Heatmap")
            fig.update_layout(**PLOTLY_DARK, height=350, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

# ════ TAB 4 ML MODELS ═════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-header">🤖 Machine Learning Models</p>', unsafe_allow_html=True)
    best_acc = clf_results[best_name]["accuracy"]

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a2744,#1e2a1e);border:2px solid #48bb78;
                border-radius:20px;padding:28px 36px;text-align:center;margin-bottom:24px'>
        <div style='color:#a0aec0;font-size:14px;letter-spacing:2px;text-transform:uppercase'>
            Best Model — {best_name}
        </div>
        <div style='font-size:80px;font-weight:900;
                    background:linear-gradient(90deg,#48bb78,#667eea);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1'>
            {best_acc:.2f}%
        </div>
        <div style='color:#e2e8f0;font-size:18px;font-weight:600;margin-top:4px'>Classification Accuracy</div>
        <div style='color:#a0aec0;font-size:13px;margin-top:4px'>
            Predicting AQI Bucket: {" / ".join(le.classes_)}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 📊 Model Comparison")
    icons = {"Random Forest":"🌲","Logistic Regression":"📈","Decision Tree":"🌿"}
    cols = st.columns(3)
    for i,(name,res) in enumerate(clf_results.items()):
        is_best = (name==best_name)
        border  = "#48bb78" if is_best else "#3d4163"
        badge   = "<span class='best-model-tag'>⭐ BEST</span>" if is_best else ""
        with cols[i]:
            st.markdown(f"""
            <div style='background:#1e2130;border:2px solid {border};border-radius:16px;
                        padding:20px;text-align:center'>
                <div style='font-size:36px'>{icons.get(name,"🤖")}</div>
                <div style='color:#e2e8f0;font-weight:700;font-size:15px;margin:8px 0'>
                    {name}{badge}</div>
                <div style='font-size:42px;font-weight:900;
                            color:{"#48bb78" if is_best else "#667eea"}'>
                    {res['accuracy']:.2f}%</div>
                <div style='color:#a0aec0;font-size:12px'>Accuracy</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Accuracy Comparison")
        names = list(clf_results.keys()); accs = [clf_results[n]["accuracy"] for n in names]
        fig = go.Figure(go.Bar(
            x=names, y=accs,
            marker_color=["#48bb78" if n==best_name else "#667eea" for n in names],
            text=[f"{a:.2f}%" for a in accs], textposition="outside",
            textfont=dict(color="white",size=14)
        ))
        fig.update_layout(**PLOTLY_DARK, height=360, yaxis_range=[0,105],
                          margin=dict(t=20,b=20), yaxis_title="Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"#### Confusion Matrix — {best_name}")
        cm = clf_results[best_name]["cm"]
        labels = list(le.classes_)
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted",y="Actual"))
        fig.update_layout(**PLOTLY_DARK, height=380, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    if hasattr(best_clf,'feature_importances_'):
        st.markdown(f"#### 🔑 Feature Importance — {best_name}")
        fi = pd.DataFrame({"Feature":avail_p,"Importance":best_clf.feature_importances_})
        fi = fi.sort_values("Importance",ascending=True)
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='viridis',
                     text=fi['Importance'].apply(lambda x:f"{x*100:.1f}%"))
        fig.update_traces(textposition='outside',textfont_color='white')
        fig.update_layout(**PLOTLY_DARK, height=340, margin=dict(t=20,b=20),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### 📈 Regression — AQI Value Prediction")
    rcols = st.columns(2)
    for i,(name,res) in enumerate(reg_results.items()):
        with rcols[i]:
            st.markdown(f"""<div style='background:#1e2130;border:1px solid #3d4163;
                border-radius:14px;padding:18px;text-align:center'>
                <div style='color:#e2e8f0;font-weight:700;font-size:15px'>{name}</div>
                <div style='color:#4fc3f7;font-size:32px;font-weight:800'>R² {res['r2']:.1f}%</div>
                <div style='color:#a0aec0;font-size:12px'>MAE: {res['mae']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    best_reg = max(reg_results, key=lambda k: reg_results[k]["r2"])
    br = reg_results[best_reg]
    c1,c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=br["y_test"], y=br["preds"], opacity=0.5,
                         color_discrete_sequence=["#667eea"],
                         labels={"x":"Actual AQI","y":"Predicted AQI"},
                         title=f"Actual vs Predicted — {best_reg}")
        mn_v=float(min(br["y_test"].min(),br["preds"].min()))
        mx_v=float(max(br["y_test"].max(),br["preds"].max()))
        fig.add_shape(type="line",x0=mn_v,y0=mn_v,x1=mx_v,y1=mx_v,
                      line=dict(color="#48bb78",dash="dash",width=2))
        fig.update_layout(**PLOTLY_DARK, height=360, margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        residuals = np.array(br["y_test"]) - np.array(br["preds"])
        fig = px.histogram(x=residuals, nbins=50, color_discrete_sequence=["#f6ad55"],
                           labels={"x":"Residuals","y":"Count"},
                           title="Residual Distribution", marginal="rug")
        fig.add_vline(x=0, line_dash="dash", line_color="#48bb78", line_width=2)
        fig.update_layout(**PLOTLY_DARK, height=360, margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### 🔄 5-Fold Cross Validation — Random Forest")
    with st.spinner("Running CV..."):
        sub_cv = df[[p for p in POLLUTANTS if p in df.columns]+['AQI_Bucket']].dropna()
        sub_cv = sub_cv[sub_cv['AQI_Bucket'].isin(BUCKET_ORDER)]
        le_cv  = LabelEncoder()
        le_cv.fit(sub_cv['AQI_Bucket'])
        y_cv   = le_cv.transform(sub_cv['AQI_Bucket'])
        X_cv   = sub_cv[[p for p in POLLUTANTS if p in df.columns]]
        cv_s   = cross_val_score(RandomForestClassifier(n_estimators=50,random_state=42),
                                 X_cv, y_cv, cv=5, scoring='accuracy') * 100
    fig = go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)], y=cv_s,
        marker_color=['#48bb78' if s>=90 else '#667eea' for s in cv_s],
        text=[f"{s:.2f}%" for s in cv_s], textposition="outside", textfont=dict(color="white")
    ))
    fig.add_hline(y=cv_s.mean(), line_dash="dash", line_color="#f6ad55", line_width=2,
                  annotation_text=f"Mean: {cv_s.mean():.2f}%", annotation_font_color="#f6ad55")
    fig.update_layout(**PLOTLY_DARK, height=320, yaxis_range=[0,105],
                      margin=dict(t=20,b=20), yaxis_title="Accuracy (%)")
    st.plotly_chart(fig, use_container_width=True)
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Mean CV", f"{cv_s.mean():.2f}%")
    with c2: st.metric("Best Fold", f"{cv_s.max():.2f}%")
    with c3: st.metric("Std Dev", f"±{cv_s.std():.2f}%")

# ════ TAB 5 PREDICTOR ════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-header">🔮 Manual AQI Predictor</p>', unsafe_allow_html=True)

    subset_r = df[[p for p in POLLUTANTS if p in df.columns]+['AQI']].dropna()
    avail_p_r = [p for p in POLLUTANTS if p in df.columns]

    @st.cache_resource
    def train_rf_reg(_k):
        rf = RandomForestRegressor(n_estimators=100,random_state=42)
        rf.fit(subset_r[avail_p_r], subset_r['AQI'])
        return rf
    rf_reg = train_rf_reg(id(df))

    c1,c2 = st.columns([1.2,1])
    with c1:
        st.markdown("#### 🎛️ Input Pollutant Values")
        ranges = {'PM2.5':(0.0,500.0),'PM10':(0.0,600.0),'NO2':(0.0,400.0),
                  'CO':(0.0,50.0),'SO2':(0.0,100.0),'O3':(0.0,200.0)}
        icons2 = {'PM2.5':'🌫️','PM10':'💨','NO2':'🟤','CO':'⚫','SO2':'🟡','O3':'🔵'}
        inputs = {}
        sc = st.columns(2)
        for i,p in enumerate(avail_p_r):
            mn,mx = ranges.get(p,(0.0,500.0))
            dv = float(np.clip(df[p].mean(),mn,mx))
            with sc[i%2]:
                inputs[p] = st.slider(f"{icons2.get(p,'🔹')} {p}",float(mn),float(mx),round(dv,1),0.5)
    with c2:
        inp_df = pd.DataFrame([inputs])
        pred_aqi = float(np.clip(rf_reg.predict(inp_df)[0],0,500))
        bucket,emoji = get_bucket_label(pred_aqi)
        color = get_aqi_color(bucket)
        st.markdown("#### 🎯 Prediction")
        st.plotly_chart(aqi_gauge(pred_aqi), use_container_width=True)
        st.markdown(f"""<div style='background:#1e2130;border:2px solid {color};
                border-radius:16px;padding:20px;text-align:center'>
            <div style='font-size:36px'>{emoji}</div>
            <div style='color:{color};font-size:28px;font-weight:700'>{bucket}</div>
            <div style='color:#a0aec0;margin-top:8px'>AQI:
                <span style='color:#e2e8f0;font-weight:700'>{pred_aqi:.1f}</span></div>
        </div>""", unsafe_allow_html=True)
        advice = {"Good":"✅ Air quality is satisfactory! Enjoy outdoor activities.",
                  "Satisfactory":"👍 Acceptable. Sensitive people limit exertion.",
                  "Moderate":"⚠️ Sensitive groups reduce outdoor time.",
                  "Poor":"😷 Everyone may feel health effects.",
                  "Very Poor":"🚨 Avoid outdoor activity.",
                  "Severe":"☠️ Emergency! Stay indoors."}
        css = "good-box" if bucket in ["Good","Satisfactory"] else "moderate-box" if bucket=="Moderate" else "poor-box" if bucket=="Poor" else "severe-box"
        st.markdown(f"<div class='info-box {css}' style='margin-top:10px'>{advice.get(bucket,'')}</div>",
                    unsafe_allow_html=True)

# ════ TAB 6 LIVE & FORECAST (NEW) ═════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-header">🌍 Real-Time AQI & Future Forecast</p>', unsafe_allow_html=True)

    if WAQI_TOKEN == "YOUR_WAQI_TOKEN_HERE":
        st.warning("⚠️ Please set your free WAQI API token in the code "
                   "(get one at https://aqicn.org/data-platform/token/) to enable this tab.")
    else:
        col_in1, col_in2 = st.columns([3,1])
        with col_in1:
            city_input = st.text_input("🏙️ Enter city name (e.g. Delhi, Mumbai, Jalandhar)", "Delhi")
        with col_in2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh Live Data"):
                st.cache_data.clear()

        # Reuse the regression model trained earlier on raw pollutant concentrations
        subset_r = df[[p for p in POLLUTANTS if p in df.columns]+['AQI']].dropna()
        avail_p_r = [p for p in POLLUTANTS if p in df.columns]

        @st.cache_resource
        def train_rf_reg_live(_k):
            rf = RandomForestRegressor(n_estimators=100,random_state=42)
            rf.fit(subset_r[avail_p_r], subset_r['AQI'])
            return rf
        rf_reg_live = train_rf_reg_live(id(df))

        live = fetch_live_data(city_input)

        if live is None:
            st.error("❌ City not found, network issue, or API limit reached. "
                     "Try a major city name in English (e.g. Delhi, Mumbai, Bengaluru).")
        else:
            iaqi = live.get("iaqi", {})

            st.markdown(f"**📍 Location:** {live['city']['name']}  |  "
                        f"**🕒 Last updated:** {live['time']['s']}")

            # ── SECTION 1: CURRENT AQI — straight from WAQI (genuinely real-time) ──
            current_aqi = float(live.get("aqi", 0))
            bucket_now, emoji_now = get_bucket_label(current_aqi)
            color = get_aqi_color(bucket_now)

            c1, c2 = st.columns([1, 1.3])
            with c1:
                st.markdown("#### 🎯 Current Live AQI")
                st.plotly_chart(aqi_gauge(current_aqi), use_container_width=True)
                st.markdown(f"""<div style='background:#1e2130;border:2px solid {color};
                        border-radius:16px;padding:16px;text-align:center'>
                    <div style='font-size:32px'>{emoji_now}</div>
                    <div style='color:{color};font-size:24px;font-weight:700'>{bucket_now}</div>
                    <div style='color:#a0aec0'>Live AQI: <b style='color:#e2e8f0'>{current_aqi:.0f}</b>
                        <span style='font-size:11px'>(source: WAQI station)</span></div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown("#### 🌫️ Live AQI Sub-Indices (per pollutant)")
                pollutant_keys = {"PM2.5":"pm25","PM10":"pm10","NO2":"no2","SO2":"so2","CO":"co","O3":"o3"}
                rows = [(name, iaqi[key]["v"]) for name,key in pollutant_keys.items()
                        if key in iaqi and "v" in iaqi[key]]
                if rows:
                    pdf2 = pd.DataFrame(rows, columns=["Pollutant", "AQI sub-index (WAQI)"])
                    st.dataframe(pdf2, use_container_width=True, hide_index=True)
                    st.caption("These are WAQI's individual pollutant AQI sub-indices "
                               "(not raw concentrations). The overall AQI shown above is "
                               "the maximum of these — same convention as CPCB/EPA.")
                else:
                    st.info("No pollutant breakdown available for this station.")

            # ── SECTION 2: FUTURE FORECAST — this is where YOUR model is used ──
            st.divider()
            st.markdown("#### 🔮 Future AQI Forecast (your trained model)")
            st.caption("WAQI provides forecasted raw pollutant *concentrations* (µg/m³) for "
                       "upcoming days. These are fed into your Random Forest model "
                       "(trained on city_day.csv concentrations → CPCB AQI) to predict "
                       "future AQI — this is genuinely your model at work.")

            forecast = live.get("forecast", {}).get("daily", {})
            if not forecast:
                st.warning("Forecast data not available for this city/station.")
            else:
                days = forecast.get("pm25", forecast.get("pm10", []))
                future_rows = []
                for i in range(len(days)):
                    row = {}
                    for poll, key in [("PM2.5","pm25"),("PM10","pm10"),("O3","o3")]:
                        if key in forecast and i < len(forecast[key]):
                            row[poll] = forecast[key][i]["avg"]
                    # NO2 / SO2 / CO are not forecast by WAQI -> use dataset averages
                    for poll in ["NO2","SO2","CO"]:
                        if poll in df.columns:
                            row[poll] = df[poll].mean()
                    row["date"] = days[i]["day"] if i < len(days) else f"Day {i+1}"
                    future_rows.append(row)

                future_df = pd.DataFrame(future_rows)
                for p in avail_p_r:
                    if p not in future_df.columns:
                        future_df[p] = df[p].mean()

                # Clip to training distribution to avoid extrapolation
                future_X = future_df[avail_p_r].copy()
                for p in avail_p_r:
                    lo, hi = clip_bounds.get(p, (future_X[p].min(), future_X[p].max()))
                    future_X[p] = future_X[p].clip(lo, hi)

                future_preds = rf_reg_live.predict(future_X)
                future_df["Predicted AQI"] = np.clip(future_preds, 0, 500).round(1)
                future_df["Category"] = future_df["Predicted AQI"].apply(lambda x: get_bucket_label(x)[0])

                fig = px.line(future_df, x="date", y="Predicted AQI", markers=True,
                              color_discrete_sequence=["#4fc3f7"],
                              title="Predicted AQI — Upcoming Days")
                fig.update_layout(**PLOTLY_DARK, height=320, margin=dict(t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)

                display_cols = ["date","Predicted AQI","Category"] + \
                                [c for c in ["PM2.5","PM10","O3"] if c in future_df.columns]
                st.dataframe(future_df[display_cols], use_container_width=True, hide_index=True)

# ════ TAB 7 RAW DATA ══════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p class="section-header">📋 Dataset Explorer</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Rows",     f"{len(fdf):,}")
    with c2: st.metric("Columns",  fdf.shape[1])
    with c3: st.metric("Missing",  int(fdf.isnull().sum().sum()))
    with c4: st.metric("Duplicates", int(fdf.duplicated().sum()))
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Data Types")
        st.dataframe(pd.DataFrame({"Column":fdf.dtypes.index,"Type":fdf.dtypes.values.astype(str)}),
                     use_container_width=True,hide_index=True)
    with c2:
        st.markdown("#### Statistics")
        st.dataframe(fdf.describe().round(2),use_container_width=True)
    n = st.slider("Rows to show",5,100,20)
    st.dataframe(fdf.head(n),use_container_width=True)
    st.download_button("⬇️ Download CSV", fdf.to_csv(index=False).encode('utf-8'),
                       "india_pollution.csv","text/csv")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#4a5568;font-size:13px'>"
            "🌫️ India Air Pollution Dashboard | Streamlit + Plotly + Scikit-learn | "
            "Live data via WAQI API</div>",
            unsafe_allow_html=True)

