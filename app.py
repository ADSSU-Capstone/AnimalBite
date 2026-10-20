import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide", page_title="Animal Bite Risk Dashboard")

st.title("Animal Bite Risk Classification in Agusan del Sur")
st.markdown("A data-driven dashboard for rabies prevention and policy analysis.")
st.markdown("---")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

@st.cache_data
def load_data():
    df = pd.read_excel("AnimalBite-SurveyRespondents_AlmadinNalupa (2).xlsx", header=1)
    df = df.map(clean_text)
    df = df.replace(['n/a', 'N/A', 'na', 'N/a', 'None', ''], np.nan)
    
    # Deep clean the Age column to fix typos seen in the raw data
    df['Age'] = df['Age'].astype(str).str.replace(' ', '').str.replace('–', '-')
    age_map = {
        '1825': '18-25', '2635': '26-35', '3645': '36-45', '3646': '36-45',
        '46andabove': '46 and above', '56andabove': '56 and above'
    }
    df['Age'] = df['Age'].replace(age_map)
    
    # Deep clean Distance column
    df['How far is the nearest animal bite center or hospital from your home?'] = \
        df['How far is the nearest animal bite center or hospital from your home?'].astype(str).str.replace(' ', '')
    
    # Fix Pet Ownership anomaly
    df['Do you have pets at home?'] = df['Do you have pets at home?'].replace({'Female': np.nan})
    
    # Re-apply title case globally
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.title()
            
    return df

df = load_data()

def calculate_risk_score(row):
    score = 0
    
    if pd.notna(row['Are your pets vaccinated against rabies?']):
        if str(row['Are your pets vaccinated against rabies?']).strip().lower() in ['none vaccinated', 'not sure']:
            score += 2
    elif row['Do you have pets at home?'] == 'Yes':
        score += 2
        
    if pd.notna(row['Do you interact with stray animals in your community?']):
        if str(row['Do you interact with stray animals in your community?']).strip().lower() in ['yes', 'sometimes']:
            score += 2
            
    if pd.notna(row['I know the proper first aid steps to take immediately after an animal bite.']):
        try:
            val = int(float(row['I know the proper first aid steps to take immediately after an animal bite.']))
            if val <= 3:
                score += 2
        except:
            pass
            
    if pd.notna(row['I know where to seek medical treatment or go to an animal bite center after an incident.']):
        try:
            val = int(float(row['I know where to seek medical treatment or go to an animal bite center after an incident.']))
            if val <= 3:
                score += 2
        except:
            pass
            
    if pd.notna(row['How far is the nearest animal bite center or hospital from your home?']):
        dist = str(row['How far is the nearest animal bite center or hospital from your home?']).strip().lower()
        if 'morethan30' in dist or '16-30' in dist:
            score += 2
            
    return score

df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
df['Risk_Level'] = pd.cut(df['Risk_Score'], bins=[-1, 3, 7, 10], labels=['Low Risk', 'Moderate Risk', 'High Risk'])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Risk Behaviors & Awareness", "Healthcare Access", "Risk Score Distribution", "Classification Models"])

with tab1:
    st.subheader("Respondent Demographics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Age Distribution")
        age_counts = df['Age'].value_counts().reset_index()
        age_counts.columns = ['Age Group', 'Count']
        st.dataframe(age_counts, hide_index=True)
        fig, ax = plt.subplots()
        sns.barplot(data=age_counts, x='Age Group', y='Count', ax=ax, palette='viridis')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col2:
        st.write("Sex Distribution")
        sex_counts = df['Sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        st.dataframe(sex_counts, hide_index=True)
        fig, ax = plt.subplots()
        sns.barplot(data=sex_counts, x='Sex', y='Count', ax=ax, palette='magma')
        st.pyplot(fig)
        
    st.write("Pet Ownership")
    pet_counts = df['Do you have pets at home?'].value_counts().reset_index()
    pet_counts.columns = ['Pet Ownership', 'Count']
    st.dataframe(pet_counts, hide_index=True)

with tab2:
    st.subheader("Awareness and Practices")
    awareness_cols = [
        'I know that rabies can be transmitted through animal bites.',
        'I am aware that dogs, cats, pigs, and other animals should receive anti-rabies vaccination.',
        'I know the proper first aid steps to take immediately after an animal bite.',
        'I know where to seek medical treatment or go to an animal bite center after an incident.',
        'I am aware of the nearest animal bite center or hospital in Agusan del Sur.',
        'I know the symptoms of rabies in animals.'
    ]
    
    for col in awareness_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    awareness_means = df[awareness_cols].mean().reset_index()
    awareness_means.columns = ['Awareness Statement', 'Mean Score (1-5)']
    st.dataframe(awareness_means, hide_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=awareness_means, y='Awareness Statement', x='Mean Score (1-5)', ax=ax, palette='Blues_d')
    ax.set_xlim(0, 5)
    st.pyplot(fig)
    
    st.write("Animal Bite Incidence (Last 12 Months)")
    bite_inc = df['Have you experienced an animal bite or scratch in the LAST 12 MONTHS?'].value_counts().reset_index()
    bite_inc.columns = ['Bitten', 'Count']
    st.dataframe(bite_inc, hide_index=True)

with tab3:
    st.subheader("Healthcare Access and ABTC Availability")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distance to Nearest ABTC")
        dist_counts = df['How far is the nearest animal bite center or hospital from your home?'].value_counts().reset_index()
        dist_counts.columns = ['Distance', 'Count']
        st.dataframe(dist_counts, hide_index=True)
        fig, ax = plt.subplots()
        sns.barplot(data=dist_counts, x='Distance', y='Count', ax=ax, palette='Oranges')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col2:
        st.write("Vaccine Availability")
        vac_avail = df['Were anti-rabies vaccines available when you sought treatment after a bite?'].value_counts().reset_index()
        vac_avail.columns = ['Availability', 'Count']
        st.dataframe(vac_avail, hide_index=True)
        fig, ax = plt.subplots()
        sns.barplot(data=vac_avail, x='Availability', y='Count', ax=ax, palette='Reds')
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab4:
    st.subheader("Calculated Animal Bite Risk Scores")
    st.write("Distribution of Risk Levels")
    risk_counts = df['Risk_Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    st.dataframe(risk_counts, hide_index=True)
    fig, ax = plt.subplots()
    sns.barplot(data=risk_counts, x='Risk Level', y='Count', ax=ax, palette='coolwarm')
    st.pyplot(fig)

with tab5:
    st.subheader("Classification Model Performance (Cross-Validation)")
    
    features = [
        'Age', 'Sex', 'Do you have pets at home?', 
        'Do you interact with stray animals in your community?',
        'I know the proper first aid steps to take immediately after an animal bite.',
        'I know where to seek medical treatment or go to an animal bite center after an incident.',
        'How far is the nearest animal bite center or hospital from your home?'
    ]
    
    ml_df = df[features + ['Risk_Level']].dropna(subset=['Risk_Level']).copy()
    
    for col in features:
        if ml_df[col].dtype == 'object':
            counts = ml_df[col].value_counts()
            rare_cats = counts[counts < 3].index
            ml_df[col] = ml_df[col].replace(rare_cats, 'Other')
            ml_df[col] = ml_df[col].fillna('Unknown')

    le_dict = {}
    for col in features:
        if ml_df[col].dtype == 'object':
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col].astype(str))
            le_dict[col] = le
            
    X = ml_df[features]
    y = ml_df['Risk_Level'].astype(str)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Naïve Bayes': GaussianNB()
    }
    
    results = []
    min_class_count = y.value_counts().min()
    n_splits = min(10, min_class_count) if min_class_count > 1 else 2
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    
    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score='raise')
        results.append({
            'Model': name,
            'Accuracy': np.mean(cv_results['test_accuracy']),
            'Precision': np.mean(cv_results['test_precision']),
            'Recall': np.mean(cv_results['test_recall']),
            'F1-Score': np.mean(cv_results['test_f1'])
        })
        
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.format("{:.3f}", subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']), hide_index=True)
    
    st.write("Feature Importance (Random Forest)")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(X, y)
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    st.dataframe(feat_imp, hide_index=True)
    
    fig, ax = plt.subplots()
    sns.barplot(data=feat_imp, x='Importance', y='Feature', ax=ax, palette='viridis')
    st.pyplot(fig)
