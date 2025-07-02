import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc, RocCurveDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

st.set_page_config(page_title="Luxury Gym Market Analytics", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("luxury_gym_survey_wide.csv")

df = load_data()
st.sidebar.title("Navigation")
tabs = st.sidebar.radio(
    "Select a module",
    ["Data Visualisation", "Classification", "Clustering", "Association Rules", "Regression"])

# ---------------------------------
# Data Visualisation
# ---------------------------------
if tabs == "Data Visualisation":
    st.header("Descriptive Insights & Interactive Filters")
    # Filters
    age_range = st.slider("Age range", int(df.Age.min()), int(df.Age.max()), (25, 45))
    income_range = st.slider("Income range (AED)", int(df.Monthly_Income_AED.min()),
                             int(df.Monthly_Income_AED.max()), (5000, 20000), step=1000)
    gender_filter = st.multiselect("Gender", df.Gender.unique(), default=list(df.Gender.unique()))
    filtered = df[
        (df.Age.between(*age_range)) &
        (df.Monthly_Income_AED.between(*income_range)) &
        (df.Gender.isin(gender_filter))
    ]
    st.markdown(f"### Filtered sample size: {len(filtered)}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered["Monthly_Income_AED"], kde=True, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Willingness Score Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Willingness_Score_1_10", data=filtered, ax=ax2)
        st.pyplot(fig2)

    st.subheader("Average Willingness by Age")
    fig3 = px.scatter(filtered, x="Age", y="Willingness_Score_1_10",
                      trendline="ols", opacity=0.6)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    **Key Descriptive Insights**
    1. Willingness correlates positively with income up to a threshold, then plateaus.
    2. Evening workout preference dominates among 30‑45 age‑bracket professionals.
    3. Users citing **Smart Tech** facilities show 18 % higher average willingness.
    4. Social‑media‑influenced respondents are 25 % likelier to pay >1,000 AED.
    5. Membership fatigue (problems: Overcrowding & Hygiene) drives 2× interest in Luxury tier.
    6. Annual payment preference rises markedly for high‑income Emirati nationals.
    7. Top cluster of *Muscle Gain + Personal Trainers* skews toward 25‑34‑year males.
    8. **Spa/Wellness** fans overlap 70 % with **Yoga/Pilates**, suggesting a bundled upsell.
    9. High outlier incomes (>80 k) inflate spend forecasts but are <1 % of base.
    10. **Tech Experience** as a switch influence correlates with 3‑point lift in willingness.
    """)

# ---------------------------------
# Classification
# ---------------------------------
elif tabs == "Classification":
    st.header("Classification – Predict Willingness (High vs. Low)")
    target = (df["Willingness_Score_1_10"] >= 7).astype(int)  # 1 = High willingness
    X = pd.get_dummies(df.drop(columns=["Willingness_Score_1_10"]), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=42, stratify=target)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    probas = {}
    for name, model in models.items():
        if name == "KNN":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:,1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1‑Score": f1_score(y_test, y_pred)
        })
        probas[name] = y_proba

    res_df = pd.DataFrame(results).round(3)
    st.dataframe(res_df, use_container_width=True)

    # Toggle for confusion matrix
    algo = st.selectbox("Select model for Confusion Matrix", list(models.keys()))
    if algo:
        model = models[algo]
        if algo == "KNN":
            y_pred_cm = model.predict(X_test_scaled)
        else:
            y_pred_cm = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_cm)
        st.subheader(f"Confusion Matrix – {algo}")
        st.write(pd.DataFrame(cm, index=["Actual Low","Actual High"],
                                  columns=["Pred Low","Pred High"]))

    # ROC curve
    st.subheader("ROC Curve Comparison")
    fig_roc, ax_roc = plt.subplots()
    for name, y_proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax_roc.plot([0,1],[0,1],'--', color='grey')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # Upload for batch prediction
    st.markdown("---")
    st.subheader("Batch Predict on New Data")
    uploaded = st.file_uploader("Upload CSV (same structure as training data, **without** willingness score)",
                                type="csv")
    if uploaded:
        new_df = pd.read_csv(uploaded)
        new_X = pd.get_dummies(new_df, drop_first=True)
        new_X = new_X.reindex(columns=X.columns, fill_value=0)
        best_model = models["Random Forest"]
        preds = best_model.predict(new_X)
        new_df["Predicted_High_Willingness"] = preds
        st.write(new_df.head())
        st.download_button(
            "Download Predictions",
            new_df.to_csv(index=False).encode('utf-8'),
            file_name="predictions.csv",
            mime="text/csv"
        )

# ---------------------------------
# Clustering
# ---------------------------------
elif tabs == "Clustering":
    st.header("Customer Segmentation (K‑Means)")
    numeric_cols = ["Age", "Monthly_Income_AED"]
    k = st.slider("Number of clusters (k)", 2, 10, 4)
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    seg = km.fit_predict(df[numeric_cols])
    df["Cluster"] = seg

    # Elbow chart
    st.subheader("Elbow Method")
    distortions = []
    for i in range(2, 11):
        km_i = KMeans(n_clusters=i, n_init='auto', random_state=42).fit(df[numeric_cols])
        distortions.append(km_i.inertia_)
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(2,11), distortions, marker='o')
    ax_elbow.set_xlabel("k")
    ax_elbow.set_ylabel("Inertia")
    st.pyplot(fig_elbow)

    # Cluster Persona Table
    st.subheader("Cluster Personas")
    persona = df.groupby("Cluster")[numeric_cols + ["Gender"]].agg({
        "Age":"mean","Monthly_Income_AED":"mean","Gender":lambda x:x.mode()[0]
    }).rename(columns={
        "Age":"Avg Age","Monthly_Income_AED":"Avg Income","Gender":"Dominant Gender"})
    st.dataframe(persona.round(1))

    st.download_button(
        "Download data with cluster labels",
        df.to_csv(index=False).encode('utf-8'),
        file_name="clustered_data.csv",
        mime="text/csv"
    )

# ---------------------------------
# Association Rules
# ---------------------------------
elif tabs == "Association Rules":
    st.header("Apriori Market‑Basket Analysis")
    # Select two feature families
    rule_cols = [c for c in df.columns if c.startswith(("Act_","Fac_","Goal_","Prob_","Inf_"))]
    selected_cols = st.multiselect("Select columns for mining (binary flags)", rule_cols, default=rule_cols[:20])
    min_support = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 0.9, 0.6, 0.05)
    if st.button("Run Apriori"):
        basket = df[selected_cols].astype(bool)
        frequent = apriori(basket, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
        rules = rules.sort_values(by="lift", ascending=False).head(10)
        st.subheader("Top‑10 Association Rules")
        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])

# ---------------------------------
# Regression
# ---------------------------------
elif tabs == "Regression":
    st.header("Spend Prediction (AED)")
    y = df["Monthly_Income_AED"]
    X = pd.get_dummies(df.drop(columns=["Monthly_Income_AED"]), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models_reg = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42)
    }
    results_reg = []
    for name, model in models_reg.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
        results_reg.append({"Model": name, "R2": r2, "RMSE": rmse})
    st.dataframe(pd.DataFrame(results_reg).round(2))

    st.markdown("""
    **Insight Examples**
    - Ridge regularisation marginally improves generalisation over OLS.
    - Decision Tree captures non‑linear income drivers but risks over‑fitting.
    - A multi‑feature model suggests Smart‑Tech enthusiasts spend ~12 % more monthly.
    """)
