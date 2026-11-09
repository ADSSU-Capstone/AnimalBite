import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Animal Bite Risk Dashboard", page_icon="🐕")

st.markdown("""
    <style>
    .stApp { background-color: #f4f6f9; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        border: 1px solid #e0e0e0;
        border-bottom: none;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0056b3 !important;
        color: white !important;
        font-weight: bold;
        border: 1px solid #0056b3;
    }
    
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #0056b3;
    }
    div[data-testid="metric-container"] > div > div > div {
        font-size: 2rem;
        color: #2c3e50;
    }
    
    h1, h2, h3 { color: #1e3a8a; }
    
    .caption-text {
        font-size: 0.8rem; 
        color: #666666; 
        margin-top: -15px; 
        margin-bottom: 25px;
        font-style: italic;
    }
    
    .plot-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Animal Bite Risk Classification Dashboard")
st.markdown("**Agusan del Sur: A Survey-Driven Policy Analysis for Rabies Prevention**")
st.markdown("---")

# --- SIDEBAR DATA MANAGEMENT ---
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Survey Dataset (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Load data, skipping the first row (group headers) to get actual column names
        df_raw = pd.read_excel(uploaded_file, header=1)
        
        # Clean column names (strip whitespace)
        df_raw.columns = df_raw.columns.astype(str).str.strip()
        
        # Clean string data
        for col in df_raw.columns:
            if df_raw[col].dtype == 'object':
                df_raw[col] = df_raw[col].astype(str).str.strip()
                df_raw[col] = df_raw[col].replace(['nan', 'None', 'N/A', 'n/a', 'N/a', '', 'N/A '], np.nan)

        if 'Age' in df_raw.columns:
            df_raw['Age'] = df_raw['Age'].astype(str).str.replace(' ', '').str.replace('–', '-')
            age_dict = {'1825': '18-25', '2635': '26-35', '3645': '36-45', '3646': '36-45', '46andabove': '46 and above', '56andabove': '56 and above', '46-55': '46-55', '26-36': '26-36'}
            df_raw['Age'] = df_raw['Age'].replace(age_dict)
            
        dist_col = 'How far is the nearest animal bite center or hospital from your home?'
        if dist_col in df_raw.columns:
            df_raw[dist_col] = df_raw[dist_col].astype(str).str.replace(' ', '')

    except Exception as e:
        st.error(f"Error loading file. Please ensure it is the correct Excel format. Details: {e}")
        st.stop()
else:
    st.info("Please upload the survey dataset (e.g., FINALISM.xlsx) using the sidebar to begin.")
    st.stop()

# --- RISK SCORING FUNCTION ---
def calculate_risk_score(row):
    score = 0
    vax_col = 'Are your pets vaccinated against rabies?'
    
    # 1. Pet Vaccination Status
    if pd.notna(row.get(vax_col)):
        if str(row[vax_col]).strip().lower() in ['none vaccinated', 'not sure']:
            score += 2
    elif row.get('Do you have pets at home?') == 'Yes':
        score += 2
        
    # 2. Stray Animal Interaction
    if pd.notna(row.get('Do you interact with stray animals in your community?')):
        if str(row['Do you interact with stray animals in your community?']).strip().lower() in ['yes', 'sometimes']:
            score += 2
            
    # 3. First Aid Knowledge
    if pd.notna(row.get('I know the proper first aid steps to take immediately after an animal bite.')):
        try:
            if int(float(row['I know the proper first aid steps to take immediately after an animal bite.'])) <= 3:
                score += 2
        except: pass
            
    # 4. PEP Awareness
    if pd.notna(row.get('I know where to seek medical treatment or go to an animal bite center after an incident.')):
        try:
            if int(float(row['I know where to seek medical treatment or go to an animal bite center after an incident.'])) <= 3:
                score += 2
        except: pass
            
    # 5. Distance to ABTC
    if pd.notna(row.get(dist_col)):
        dist = str(row[dist_col]).strip().lower()
        if 'morethan30' in dist or '16-30' in dist:
            score += 2
            
    return score

df_raw['Risk_Score'] = df_raw.apply(calculate_risk_score, axis=1)
df_raw['Risk_Level'] = pd.cut(df_raw['Risk_Score'], bins=[-1, 3, 7, 10], labels=['Low Risk', 'Moderate Risk', 'High Risk'])

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("---")
st.sidebar.header("Interactive Filters")

muni_options = df_raw['Municipality in Agusan del Sur'].dropna().unique().tolist()
selected_munis = st.sidebar.multiselect("Filter by Municipality", options=muni_options, default=muni_options)

age_options = df_raw['Age'].dropna().unique().tolist()
selected_ages = st.sidebar.multiselect("Filter by Age Group", options=age_options, default=age_options)

sex_options = df_raw['Sex'].dropna().unique().tolist()
selected_sex = st.sidebar.multiselect("Filter by Sex", options=sex_options, default=sex_options)

df = df_raw[
    (df_raw['Municipality in Agusan del Sur'].isin(selected_munis)) &
    (df_raw['Age'].isin(selected_ages)) &
    (df_raw['Sex'].isin(selected_sex))
].copy()

# --- KPI CALCULATIONS ---
total_respondents = len(df)
high_risk_count = len(df[df['Risk_Level'] == 'High Risk'])
high_risk_pct = (high_risk_count / total_respondents) * 100 if total_respondents > 0 else 0
bitten_count = len(df[df['Have you experienced an animal bite or scratch in the LAST 12 MONTHS?'] == 'Yes'])

vax_col = 'Are your pets vaccinated against rabies?'
pet_owners_df = df[df['Do you have pets at home?'] == 'Yes']
total_pet_owners = len(pet_owners_df)

if vax_col in df.columns and total_pet_owners > 0:
    unvax_count = len(pet_owners_df[
        pet_owners_df[vax_col].astype(str).str.strip().str.lower().isin(['none vaccinated', 'not sure'])
    ])
    unvax_pct = (unvax_count / total_pet_owners) * 100
    unvax_label = f"{unvax_count} ({unvax_pct:.1f}%)"
else:
    unvax_label = "N/A"

# --- KPI ROW ---
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Filtered Respondents", f"{total_respondents}")
kpi2.metric("Bitten in Last 12 Mos.", f"{bitten_count}")
kpi3.metric("High Risk Respondents", f"{high_risk_count} ({high_risk_pct:.1f}%)")
kpi4.metric("Municipalities Selected", len(selected_munis))
kpi5.metric("Unvaccinated/Unsure Pets", unvax_label)

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Demographics & Geography", 
    "Risk Behaviors & Awareness", 
    "Healthcare Access", 
    "Risk Score & Municipality", 
    "Classification Models",
    "Qualitative Insights",
    "Policy Recommendations"
])

with tab1:
    st.subheader("Demographics & Geographic Distribution")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Age Distribution**")
            age_counts = df['Age'].value_counts().reset_index()
            age_counts.columns = ['Age Group', 'Count']
            fig_age = px.bar(age_counts, x='Age Group', y='Count', color='Age Group', 
                             color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_white')
            fig_age.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_age, use_container_width=True)
            st.caption("This figure displays the age distribution of the respondents, which is vital for understanding demographic vulnerabilities in the study.")
        
        with st.container(border=True):
            st.markdown("**Sex Distribution**")
            sex_counts = df['Sex'].value_counts().reset_index()
            sex_counts.columns = ['Sex', 'Count']
            fig_sex = px.pie(sex_counts, names='Sex', values='Count', hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig_sex.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_sex, use_container_width=True)
            st.caption("This chart illustrates the sex distribution of the survey participants, providing a baseline for analyzing risk behaviors.")
        
    with col2:
        with st.container(border=True):
            # NEW DROPDOWN FOR LOCATION LEVEL
            loc_level = st.selectbox("Select Location Level", ["Respondents per Municipality", "Respondents per Barangay"])
            
            if loc_level == "Respondents per Municipality":
                st.markdown("**Respondents per Municipality**")
                muni_counts = df['Municipality in Agusan del Sur'].value_counts().reset_index()
                muni_counts.columns = ['Location', 'Count']
                fig_loc = px.bar(muni_counts, x='Count', y='Location', orientation='h', 
                                 color='Count', color_continuous_scale='Blues', template='plotly_white')
                fig_loc.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_loc, use_container_width=True)
                st.caption("This graph shows the distribution of respondents across the selected municipalities.")
            else:
                st.markdown("**Respondents per Barangay**")
                barangay_col = 'Purok & Barangay' 
                muni_col = 'Municipality in Agusan del Sur'
                
                if barangay_col in df.columns and muni_col in df.columns:
                    # Clean up barangay names (remove "Purok X" and keep just the Barangay name)
                    df['Clean_Barangay'] = df[barangay_col].astype(str).str.replace(r'Purok\s*\w*\s*', '', regex=True).str.strip()
                    
                    # Combine Barangay and Municipality to avoid grouping all "Poblacion" together
                    df['Full_Location'] = df['Clean_Barangay'] + ", " + df[muni_col]
                    
                    # Count and get top 20
                    brgy_counts = df['Full_Location'].value_counts().reset_index().head(20) 
                    brgy_counts.columns = ['Location', 'Count']
                    
                    fig_loc = px.bar(brgy_counts, x='Count', y='Location', orientation='h', 
                                     color='Count', color_continuous_scale='Teal', template='plotly_white')
                    fig_loc.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_loc, use_container_width=True)
                    st.caption("This graph shows the distribution of respondents across the top 20 specific barangays (combined with their municipality).")
                else:
                    st.warning("Barangay or Municipality column not found in dataset.")

        with st.container(border=True):
            st.markdown("**Pet Ownership**")
            pet_counts = df['Do you have pets at home?'].value_counts().reset_index()
            pet_counts.columns = ['Owns Pet', 'Count']
            fig_pet = px.pie(pet_counts, names='Owns Pet', values='Count', color='Owns Pet',
                             color_discrete_map={'Yes': '#2ecc71', 'No': '#e74c3c'})
            fig_pet.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_pet, use_container_width=True)
            st.caption("This figure indicates the number of respondents who own pets at home, a major determinant of animal bite risk.")

        with st.container(border=True):
            st.markdown("**Pet Vaccination Status (Among Pet Owners)**")
            if vax_col in df.columns:
                vax_counts = pet_owners_df[vax_col].value_counts().reset_index()
                vax_counts.columns = ['Vaccination Status', 'Count']
                
                fig_vax = px.pie(vax_counts, names='Vaccination Status', values='Count', hole=0.4,
                                 color='Vaccination Status',
                                 color_discrete_map={
                                     'All vaccinated': '#2ecc71', 
                                     'Some vaccinated': '#f1c40f', 
                                     'None vaccinated': '#e74c3c',
                                     'Not sure': '#95a5a6'
                                 })
                fig_vax.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_vax, use_container_width=True)
                st.caption("This chart breaks down the rabies vaccination status of the pets owned by respondents, a critical factor in the risk scoring matrix.")
            else:
                st.warning("Vaccination column not found in dataset.")

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
    
    for col in awareness_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    valid_awareness = [c for c in awareness_cols if c in df.columns]
    if valid_awareness:
        with st.container(border=True):
            st.markdown("**Mean Awareness Scores (1-5 Likert Scale)**")
            awareness_means = df[valid_awareness].mean().reset_index()
            awareness_means.columns = ['Awareness Statement', 'Mean Score']
            awareness_means = awareness_means.sort_values(by='Mean Score', ascending=True)
            
            fig_aware = px.bar(awareness_means, x='Mean Score', y='Awareness Statement', orientation='h',
                               color='Mean Score', color_continuous_scale='Teal', text_auto='.2f', template='plotly_white')
            fig_aware.update_layout(xaxis_range=[0, 5], coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_aware, use_container_width=True)
            st.caption("This figure reflects the respondents' level of awareness regarding rabies transmission, first aid, and ABTC locations, which are critical inputs for the risk score.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    stray_col = 'Do you interact with stray animals in your community?'
    if stray_col in df.columns:
        with st.container(border=True):
            st.markdown("**Interaction with Stray Animals**")
            stray_counts = df[stray_col].value_counts().reset_index()
            stray_counts.columns = ['Interaction', 'Count']
            fig_stray = px.bar(stray_counts, x='Interaction', y='Count', color='Interaction',
                               color_discrete_sequence=px.colors.qualitative.Vivid, template='plotly_white')
            fig_stray.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_stray, use_container_width=True)
            st.caption("This chart shows how frequently respondents interact with stray animals, a behavior that directly adds to the risk score due to high exposure.")

with tab3:
    st.subheader("Healthcare Access and ABTC Availability")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Distance to Nearest ABTC**")
            if dist_col in df.columns:
                dist_counts = df[dist_col].value_counts().reset_index()
                dist_counts.columns = ['Distance', 'Count']
                fig_dist = px.pie(dist_counts, names='Distance', values='Count', hole=0.4,
                                  color_discrete_sequence=px.colors.sequential.Oranges_r)
                fig_dist.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_dist, use_container_width=True)
                st.caption("This figure highlights geographical accessibility. Distances over 10 km directly contribute to higher risk scores in the study's framework.")
            
    with col2:
        with st.container(border=True):
            st.markdown("**Vaccine Availability**")
            vac_col = 'Were anti-rabies vaccines available when you sought treatment after a bite?'
            if vac_col in df.columns:
                vac_avail = df[vac_col].value_counts().reset_index()
                vac_avail.columns = ['Availability', 'Count']
                fig_vac = px.bar(vac_avail, x='Availability', y='Count', color='Availability',
                                 color_discrete_sequence=px.colors.qualitative.Safe, template='plotly_white')
                fig_vac.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_vac, use_container_width=True)
                st.caption("This chart reveals whether anti-rabies vaccines were available, underscoring potential systemic gaps and stockouts in healthcare delivery.")

with tab4:
    st.subheader("Risk Level Distribution")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True):
            st.markdown("**Overall Risk Levels**")
            risk_counts = df['Risk_Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            fig_risk = px.pie(risk_counts, names='Risk Level', values='Count', hole=0.5,
                              color='Risk Level', color_discrete_map={'Low Risk': '#2ecc71', 'Moderate Risk': '#f1c40f', 'High Risk': '#e74c3c'})
            fig_risk.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_risk, use_container_width=True)
            st.caption("This chart categorizes respondents into Low, Moderate, and High-risk groups based on the study's scoring matrix.")
        
    with col2:
        with st.container(border=True):
            st.markdown("**Risk Level by Municipality (Stacked)**")
            muni_risk = pd.crosstab(df['Municipality in Agusan del Sur'], df['Risk_Level'])
            fig_muni_risk = px.bar(muni_risk, barmode='stack', 
                                   color_discrete_map={'Low Risk': '#2ecc71', 'Moderate Risk': '#f1c40f', 'High Risk': '#e74c3c'},
                                   template='plotly_white')
            fig_muni_risk.update_layout(xaxis_title="Municipality", yaxis_title="Number of Respondents", margin=dict(l=0, r=0, t=10, b=0), legend_title="Risk Level")
            st.plotly_chart(fig_muni_risk, use_container_width=True)
            st.caption("This cross-tabulation identifies which municipalities have the highest concentration of high-risk individuals for prioritized interventions.")
            
    st.markdown("---")
    st.subheader("Interactive What-If Risk Calculator")
    st.markdown("Adjust the variables below to see how they affect the overall risk score based on the 5-variable scoring matrix.")
    
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    with rc1:
        pet_vax = st.selectbox("Pet Vaccinated?", ["Yes", "No / Not Sure"], index=1)
    with rc2:
        stray_int = st.selectbox("Interacts with Strays?", ["Yes / Sometimes", "No"], index=0)
    with rc3:
        first_aid = st.slider("First Aid Knowledge (1-5)", 1, 5, 2)
    with rc4:
        pep_aware = st.slider("PEP Awareness (1-5)", 1, 5, 2)
    with rc5:
        abtc_dist = st.selectbox("Distance to ABTC", ["< 10 km", "> 10 km"], index=1)
        
    calc_score = 0
    if pet_vax == "No / Not Sure": calc_score += 2
    if stray_int == "Yes / Sometimes": calc_score += 2
    if first_aid <= 3: calc_score += 2
    if pep_aware <= 3: calc_score += 2
    if abtc_dist == "> 10 km": calc_score += 2
    
    if calc_score <= 3:
        calc_risk = "Low Risk"
        color = "green"
    elif calc_score <= 7:
        calc_risk = "Moderate Risk"
        color = "orange"
    else:
        calc_risk = "High Risk"
        color = "red"
        
    st.markdown(f"**Calculated Score:** {calc_score} / 10 &nbsp;&nbsp;|&nbsp;&nbsp; **Risk Level:** <span style='color:{color}; font-weight:bold; font-size:1.2rem;'>{calc_risk}</span>", unsafe_allow_html=True)
    st.caption("This interactive tool demonstrates how the 5 independent variables from the methodology combine to assign a risk category.")

with tab5:
    st.subheader("Machine Learning Classification Models")
    
    features = [
        'Age', 'Sex', 'Do you have pets at home?', 
        'Municipality in Agusan del Sur',
        'Do you interact with stray animals in your community?',
        'I know the proper first aid steps to take immediately after an animal bite.',
        'I know where to seek medical treatment or go to an animal bite center after an incident.',
        dist_col
    ]
    
    features = [f for f in features if f in df.columns]
    
    ml_df = df[features + ['Risk_Level']].dropna(subset=['Risk_Level']).copy()
    
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
        st.caption("This table displays the performance metrics of the three models using cross-validation to identify the most effective algorithm for predicting risk levels.")
        
        with st.container(border=True):
            st.markdown("**Feature Importance (Random Forest)**")
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            rf.fit(X, y)
            importances = rf.feature_importances_
            feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            fig_feat = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', 
                              color='Importance', color_continuous_scale='Viridis', template='plotly_white')
            fig_feat.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_feat, use_container_width=True)
            st.caption("This figure highlights the most influential variables in predicting animal bite risk, supporting evidence-based prioritization for health programs.")
    else:
        st.warning("Not enough data or classes to train models based on the current filters.")

with tab6:
    st.subheader("Open-Ended Survey Responses")
    
    q1 = 'What do you think are the most common causes of animal bites in your community?'
    q2 = 'What suggestions can you give to help reduce animal bite incidents in your barangay?'
    q3 = 'In your opinion, what additional facilities or resources does the Provincial Health Office need to improve animal bite prevention?'
    
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            if q1 in df.columns:
                st.markdown("**Common Causes**")
                responses_q1 = df[q1].dropna().value_counts().reset_index()
                responses_q1.columns = ['Response', 'Frequency']
                st.dataframe(responses_q1.head(10), hide_index=True, use_container_width=True)
                st.caption("Highlights community perceptions of bite causes, often pointing to stray animals and irresponsible ownership.")
    with col2:
        with st.container(border=True):
            if q2 in df.columns:
                st.markdown("**Suggestions to Reduce Incidents**")
                responses_q2 = df[q2].dropna().value_counts().reset_index()
                responses_q2.columns = ['Response', 'Frequency']
                st.dataframe(responses_q2.head(10), hide_index=True, use_container_width=True)
                st.caption("Captures local recommendations for prevention, such as vaccination drives and leash enforcement.")
    with col3:
        with st.container(border=True):
            if q3 in df.columns:
                st.markdown("**Facilities Needed**")
                responses_q3 = df[q3].dropna().value_counts().reset_index()
                responses_q3.columns = ['Response', 'Frequency']
                st.dataframe(responses_q3.head(10), hide_index=True, use_container_width=True)
                st.caption("Identifies immediate resource gaps felt by the community, primarily better access to vaccines and nearby ABTCs.")

with tab7:
    st.subheader("Evidence-Based Policy Recommendations")
    
    with st.expander("1. Establishment of Local ABTCs and Mobile Vaccination", expanded=True):
        st.markdown("""
        *   **Data Basis:** The Healthcare Access tab shows a large portion of respondents live 16-30 km or more than 30 km away from an ABTC. Feature Importance identifies distance as a top predictor of risk.
        *   **Action:** Establish Animal Bite Treatment Centers (ABTCs) in high-risk, remote municipalities and deploy mobile vaccination units.
        """)
        
    with st.expander("2. Mass Dog Vaccination and Stray Control", expanded=False):
        st.markdown("""
        *   **Data Basis:** A significant number of respondents report interacting with stray animals and having unvaccinated pets. The Qualitative Insights tab shows "Stray dogs roaming" as the most common cause of bites.
        *   **Action:** Intensify mass dog vaccination campaigns targeting 70% coverage and enforce stricter barangay ordinances on stray animal impounding.
        """)
        
    with st.expander("3. Intensified Information and Education Campaigns (IEC)", expanded=False):
        st.markdown("""
        *   **Data Basis:** Awareness scores show that while respondents know rabies is transmitted by bites, knowledge of proper first aid and exact ABTC locations is moderate.
        *   **Action:** Conduct barangay-level health education focusing on immediate wound washing and the importance of completing the Post-Exposure Prophylaxis schedule.
        """)
        
    with st.expander("4. Municipal-Targeted Interventions", expanded=False):
        st.markdown("""
        *   **Data Basis:** The Risk Score & Municipality tab highlights specific municipalities with high percentages of High Risk respondents.
        *   **Action:** Redirect resources and prioritize interventions in these specific municipalities based on the data-driven risk map.
        """)
