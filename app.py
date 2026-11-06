import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(layout="wide", page_title="Animal Bite Risk Dashboard", page_icon="🐾")

st.markdown("""
    <style>
    /* Main container styling */
    .main {background-color: #f8f9fa;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #0056b3 !important;
        color: white !important;
        font-weight: bold;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* Headers */
    h1, h2, h3 {color: #2c3e50;}
    </style>
""", unsafe_allow_html=True)

st.title("🐾 Animal Bite Risk Classification Dashboard")
st.markdown("**Agusan del Sur: A Survey-Driven Policy Analysis for Rabies Prevention**")
st.markdown("---")

# --- SIDEBAR: FILE UPLOADER ---
st.sidebar.header("📂 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Survey Dataset (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Assuming headers are on the second row based on the provided dataset structure
        df = pd.read_excel(uploaded_file, header=1)
        
        # Basic Data Cleaning
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', 'N/A', 'n/a', 'N/a', ''], np.nan)

        # Standardize Age
        if 'Age' in df.columns:
            df['Age'] = df['Age'].astype(str).str.replace(' ', '').str.replace('–', '-')
            age_dict = {'1825': '18-25', '2635': '26-35', '3645': '36-45', '3646': '36-45', '46andabove': '46 and above', '56andabove': '56 and above'}
            df['Age'] = df['Age'].replace(age_dict)
        
        # Standardize Distance
        dist_col = 'How far is the nearest animal bite center or hospital from your home?'
        if dist_col in df.columns:
            df[dist_col] = df[dist_col].astype(str).str.replace(' ', '')

    except Exception as e:
        st.error(f"Error loading file. Please ensure it is the correct Excel format. Details: {e}")
        st.stop()
else:
    st.info("👈 Please upload the survey dataset (e.g., FINALISM.xlsx) using the sidebar to begin.")
    st.stop()

# --- RISK SCORE CALCULATION ---
def calculate_risk_score(row):
    score = 0
    # Factor 1: Pet Vaccination
    if pd.notna(row.get('Are your pets vaccinated against rabies?')):
        if str(row['Are your pets vaccinated against rabies?']).strip().lower() in ['none vaccinated', 'not sure']:
            score += 2
    elif row.get('Do you have pets at home?') == 'Yes':
        score += 2
        
    # Factor 2: Stray Animal Interaction
    if pd.notna(row.get('Do you interact with stray animals in your community?')):
        if str(row['Do you interact with stray animals in your community?']).strip().lower() in ['yes', 'sometimes']:
            score += 2
            
    # Factor 3: First Aid Knowledge (Likert 1-5)
    if pd.notna(row.get('I know the proper first aid steps to take immediately after an animal bite.')):
        try:
            if int(float(row['I know the proper first aid steps to take immediately after an animal bite.'])) <= 3:
                score += 2
        except: pass
            
    # Factor 4: PEP Knowledge (Likert 1-5)
    if pd.notna(row.get('I know where to seek medical treatment or go to an animal bite center after an incident.')):
        try:
            if int(float(row['I know where to seek medical treatment or go to an animal bite center after an incident.'])) <= 3:
                score += 2
        except: pass
            
    # Factor 5: ABTC Distance
    if pd.notna(row.get(dist_col)):
        dist = str(row[dist_col]).strip().lower()
        if 'morethan30' in dist or '16-30' in dist:
            score += 2
            
    return score

df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
df['Risk_Level'] = pd.cut(df['Risk_Score'], bins=[-1, 3, 7, 10], labels=['Low Risk', 'Moderate Risk', 'High Risk'])

# --- KPI CARDS ---
total_respondents = len(df)
high_risk_count = len(df[df['Risk_Level'] == 'High Risk'])
high_risk_pct = (high_risk_count / total_respondents) * 100 if total_respondents > 0 else 0
bitten_count = len(df[df['Have you experienced an animal bite or scratch in the LAST 12 MONTHS?'] == 'Yes'])

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Respondents", f"{total_respondents}")
kpi2.metric("Bitten in Last 12 Mos.", f"{bitten_count}")
kpi3.metric("High Risk Respondents", f"{high_risk_count} ({high_risk_pct:.1f}%)")
kpi4.metric("Municipalities Covered", df['Municipality in Agusan del Sur'].nunique())

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Demographics & Geography", 
    "🚨 Risk Behaviors & Awareness", 
    "🏥 Healthcare Access", 
    "⚠️ Risk Score & Municipality", 
    "🤖 Classification Models",
    "💬 Qualitative Insights",
    "📜 Policy Recommendations"
])

with tab1:
    st.subheader("Demographics & Geographic Distribution")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Age Distribution**")
        age_counts = df['Age'].value_counts().reset_index()
        age_counts.columns = ['Age Group', 'Count']
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=age_counts, x='Age Group', y='Count', palette='viridis', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.markdown("**Sex Distribution**")
        sex_counts = df['Sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=sex_counts, x='Sex', y='Count', palette='magma', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.markdown("**Respondents per Municipality**")
        muni_counts = df['Municipality in Agusan del Sur'].value_counts().reset_index()
        muni_counts.columns = ['Municipality', 'Count']
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=muni_counts, y='Municipality', x='Count', palette='Blues_d', ax=ax)
        st.pyplot(fig)
        
        st.markdown("**Pet Ownership**")
        pet_counts = df['Do you have pets at home?'].value_counts().reset_index()
        pet_counts.columns = ['Owns Pet', 'Count']
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=pet_counts, x='Owns Pet', y='Count', palette='Set2', ax=ax)
        st.pyplot(fig)

with tab2:
    st.subheader("Awareness and Risk Behaviors")
    
    awareness_cols = [
        'I know that rabies can be transmitted through animal bites.',
        'I am aware that dogs, cats, pigs, and other animals should receive anti-rabies vaccination.',
        'I know the proper first aid steps to take immediately after an animal bite.',
        'I know where to seek medical treatment or go to an animal bite center after an incident.',
        'I am aware of the nearest animal bite center or hospital in Agusan del Sur.',
        'I know the symptoms of rabies in animals.'
    ]
    
    # Clean numeric columns
    for col in awareness_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    valid_awareness = [c for c in awareness_cols if c in df.columns]
    if valid_awareness:
        awareness_means = df[valid_awareness].mean().reset_index()
        awareness_means.columns = ['Awareness Statement', 'Mean Score (1-5)']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=awareness_means, y='Awareness Statement', x='Mean Score (1-5)', palette='Blues_d', ax=ax)
        ax.set_xlim(0, 5)
        for i in ax.containers:
            ax.bar_label(i, fmt='%.2f', padding=3)
        st.pyplot(fig)
    
    st.markdown("**Interaction with Stray Animals**")
    stray_col = 'Do you interact with stray animals in your community?'
    if stray_col in df.columns:
        stray_counts = df[stray_col].value_counts().reset_index()
        stray_counts.columns = ['Interaction', 'Count']
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=stray_counts, x='Interaction', y='Count', palette='Oranges', ax=ax)
        st.pyplot(fig)

with tab3:
    st.subheader("Healthcare Access and ABTC Availability")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distance to Nearest ABTC**")
        if dist_col in df.columns:
            dist_counts = df[dist_col].value_counts().reset_index()
            dist_counts.columns = ['Distance', 'Count']
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=dist_counts, x='Distance', y='Count', palette='Oranges', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    with col2:
        st.markdown("**Vaccine Availability**")
        vac_col = 'Were anti-rabies vaccines available when you sought treatment after a bite?'
        if vac_col in df.columns:
            vac_avail = df[vac_col].value_counts().reset_index()
            vac_avail.columns = ['Availability', 'Count']
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=vac_avail, x='Availability', y='Count', palette='Reds', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

with tab4:
    st.subheader("Risk Level Distribution by Municipality")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Overall Risk Levels**")
        risk_counts = df['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=risk_counts, x='Risk Level', y='Count', palette='coolwarm', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.markdown("**Risk Level by Municipality**")
        # Cross-tabulation for stacked bar chart
        muni_risk = pd.crosstab(df['Municipality in Agusan del Sur'], df['Risk_Level'])
        fig, ax = plt.subplots(figsize=(10, 5))
        muni_risk.plot(kind='bar', stacked=True, color=['#2ecc71', '#f1c40f', '#e74c3c'], ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Number of Respondents")
        plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

with tab5:
    st.subheader("🤖 Machine Learning Classification Models")
    st.markdown("Comparison of **Decision Tree, Random Forest, and Naïve Bayes** using 10-Fold Cross-Validation.")
    
    features = [
        'Age', 'Sex', 'Do you have pets at home?', 
        'Municipality in Agusan del Sur',
        'Do you interact with stray animals in your community?',
        'I know the proper first aid steps to take immediately after an animal bite.',
        'I know where to seek medical treatment or go to an animal bite center after an incident.',
        dist_col
    ]
    
    # Filter only columns that exist
    features = [f for f in features if f in df.columns]
    
    ml_df = df[features + ['Risk_Level']].dropna(subset=['Risk_Level']).copy()
    
    # Encode categorical variables
    for col in features:
        if ml_df[col].nunique() < 20:
            ml_df[col] = ml_df[col].astype(str).fillna('Unknown')
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col])
        else:
            ml_df[col] = pd.to_numeric(ml_df[col], errors='coerce')
            ml_df[col] = ml_df[col].fillna(ml_df[col].median())

    X = ml_df[features].to_numpy()
    y = ml_df['Risk_Level'].astype(str).to_numpy()
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Naïve Bayes': GaussianNB()
    }
    
    if len(np.unique(y)) > 1 and len(y) > 10:
        results = []
        min_class_count = pd.Series(y).value_counts().min()
        n_splits = min(10, min_class_count) if min_class_count > 1 else 2
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        
        for name, model in models.items():
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
            results.append({
                'Model': name,
                'Accuracy': np.mean(cv_results['test_accuracy']),
                'Precision': np.mean(cv_results['test_precision']),
                'Recall': np.mean(cv_results['test_recall']),
                'F1-Score': np.mean(cv_results['test_f1'])
            })
            
        results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
        st.dataframe(results_df.style.format("{:.3f}", subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']).highlight_max(color='lightgreen', axis=0), use_container_width=True)
        
        st.markdown("**Feature Importance (Random Forest)**")
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rf.fit(X, y)
        importances = rf.feature_importances_
        feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough data or classes to train models. Please ensure the uploaded dataset contains sufficient responses.")

with tab6:
    st.subheader("💬 Open-Ended Survey Responses")
    st.markdown("Top responses gathered from the qualitative survey.")
    
    q1 = 'What do you think are the most common causes of animal bites in your community?'
    q2 = 'What suggestions can you give to help reduce animal bite incidents in your barangay?'
    q3 = 'In your opinion, what additional facilities or resources does the Provincial Health Office need to improve animal bite prevention?'
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if q1 in df.columns:
            st.markdown("**Common Causes**")
            responses_q1 = df[q1].dropna().value_counts().reset_index()
            responses_q1.columns = ['Response', 'Frequency']
            st.dataframe(responses_q1.head(10), hide_index=True, use_container_width=True)
    with col2:
        if q2 in df.columns:
            st.markdown("**Suggestions to Reduce Incidents**")
            responses_q2 = df[q2].dropna().value_counts().reset_index()
            responses_q2.columns = ['Response', 'Frequency']
            st.dataframe(responses_q2.head(10), hide_index=True, use_container_width=True)
    with col3:
        if q3 in df.columns:
            st.markdown("**Facilities Needed**")
            responses_q3 = df[q3].dropna().value_counts().reset_index()
            responses_q3.columns = ['Response', 'Frequency']
            st.dataframe(responses_q3.head(10), hide_index=True, use_container_width=True)

with tab7:
    st.subheader("📜 Evidence-Based Policy Recommendations")
    st.markdown("""
    Based on the quantitative models and qualitative feedback, the following policy actions are recommended for the Provincial Health Office (PHO) and Local Government Units (LGUs):
    
    **1. Establishment of Local ABTCs and Mobile Vaccination**
    *   *Data Basis:* The Healthcare Access tab shows a large portion of respondents live 16-30 km or more than 30 km away from an ABTC. Feature Importance identifies distance as a top predictor of risk.
    *   *Action:* Establish Animal Bite Treatment Centers (ABTCs) in high-risk, remote municipalities and deploy mobile vaccination units.
    
    **2. Mass Dog Vaccination and Stray Control**
    *   *Data Basis:* A significant number of respondents report interacting with stray animals and having unvaccinated pets. The Qualitative Insights tab shows "Stray dogs roaming" as the most common cause of bites.
    *   *Action:* Intensify mass dog vaccination campaigns (targeting 70% coverage) and enforce stricter barangay ordinances on stray animal impounding and responsible pet ownership (leashing).
    
    **3. Intensified Information and Education Campaigns (IEC)**
    *   *Data Basis:* Awareness scores show that while respondents know rabies is transmitted by bites, knowledge of proper first aid and exact ABTC locations is moderate.
    *   *Action:* Conduct barangay-level health education focusing on immediate wound washing (soap and water) and the importance of completing the Post-Exposure Prophylaxis (PEP) schedule.
    
    **4. Municipal-Targeted Interventions**
    *   *Data Basis:* The Risk Score & Municipality tab highlights specific municipalities with high percentages of "High Risk" respondents.
    *   *Action:* Redirect resources and prioritize interventions in these specific municipalities.
    """)
