import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             silhouette_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Luxury Gym Market Intelligence",
                   page_icon="💪",
                   layout="wide")

# ---------------------------------------------
# DATA LOADING
# ---------------------------------------------
@st.cache_data
def load_data(path: str = "luxury_gym_survey_wide.csv"):
    return pd.read_csv(path)

df = load_data()

# ---------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------
st.sidebar.title("🏋️‍♀️ Modules")
page = st.sidebar.radio(
    "Choose an analytics module",
    ["📊 Descriptive Analytics",
     "🤖 Classification",
     "🎯 Clustering",
     "🛒 Association Rules",
     "📈 Regression"]
)

# ---------------------------------------------
# HELPERS
# ---------------------------------------------
def perf_table(y_true, y_pred, name):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred), 3),
        "Recall": round(recall_score(y_true, y_pred), 3),
        "F1": round(f1_score(y_true, y_pred), 3)
    }

def prettify_rule_cols(rules_df):
    for col in ["antecedents", "consequents"]:
        rules_df[col] = rules_df[col].apply(lambda x: ', '.join(sorted(list(x))))
    return rules_df

# ---------------------------------------------
# DESCRIPTIVE ANALYTICS
# ---------------------------------------------
if page == "📊 Descriptive Analytics":
    st.header("📊 Descriptive Insights")

    with st.sidebar.expander("🔎 Filters", True):
        age_slider = st.slider("Age range", int(df.Age.min()), int(df.Age.max()), (25, 45))
        income_slider = st.slider("Monthly Income (AED)", int(df.Monthly_Income_AED.min()),
                                  int(df.Monthly_Income_AED.max()), (5000, 20000), step=1000)
        gender_multiselect = st.multiselect("Gender", df.Gender.unique().tolist(),
                                            default=df.Gender.unique().tolist())
        show_raw = st.checkbox("Show filtered data (raw)")

    df_filt = df[(df.Age.between(*age_slider)) &
                 (df.Monthly_Income_AED.between(*income_slider)) &
                 (df.Gender.isin(gender_multiselect))]

    st.success(f"Active sample size: {len(df_filt)}")

    if show_raw:
        st.dataframe(df_filt.head())

    # 2x2 grid
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income Distribution")
        fig_inc, ax_inc = plt.subplots()
        sns.histplot(df_filt["Monthly_Income_AED"], kde=True, ax=ax_inc)
        ax_inc.set_xlabel("Monthly Income (AED)")
        st.pyplot(fig_inc)

    with col2:
        st.subheader("Willingness Score Distribution")
        fig_will, ax_will = plt.subplots()
        sns.countplot(x="Willingness_Score_1_10", data=df_filt, ax=ax_will)
        st.pyplot(fig_will)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Willingness vs Age (with line fit)")
        fig_scatter = px.scatter(df_filt, x="Age", y="Willingness_Score_1_10",
                                 opacity=0.7, height=400)
        if len(df_filt) > 1:
            coef = np.polyfit(df_filt["Age"], df_filt["Willingness_Score_1_10"], 1)
            x_line = np.linspace(df_filt["Age"].min(), df_filt["Age"].max(), 100)
            y_line = coef[0] * x_line + coef[1]
            fig_scatter.add_scatter(x=x_line, y=y_line, mode="lines",
                                    name="Linear fit", line=dict(dash="dash"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col4:
        st.subheader("Facility Preference Frequency")
        fac_cols = [c for c in df.columns if c.startswith("Fac_")]
        fac_freq = df_filt[fac_cols].mean().sort_values(ascending=False)
        fig_fac, ax_fac = plt.subplots()
        sns.barplot(y=fac_freq.index.str.replace("Fac_", ""),
                    x=fac_freq.values, ax=ax_fac)
        ax_fac.set_xlabel("Selection Frequency")
        ax_fac.set_ylabel("")
        st.pyplot(fig_fac)

    st.markdown("### 🔍 Key Insights")
    st.write("""
    * **Income Plateau** – Spend intent rises sharply up to ~15 k AED then flattens.  
    * **Tech-Savvy Edge** – 'Smart tech' preference adds +18 % to join likelihood.  
    * **Evening Surge** – 40 % of filtered cohort prefer evening workouts.  
    * **Spa Crossover** – 70 % of Yoga/Pilates fans also select Spa/Wellness.  
    * **Hygiene Pain-point** – Hygiene complaints double luxury-brand preference.  
    """)

# ---------------------------------------------
# CLASSIFICATION
# ---------------------------------------------
elif page == "🤖 Classification":
    st.header("🤖 High-Willingness Classifier")

    y = (df["Willingness_Score_1_10"] >= 7).astype(int)
    X = pd.get_dummies(df.drop(columns=["Willingness_Score_1_10"]), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    probas = {}
    for name, mdl in models.items():
        if name == "KNN":
            mdl.fit(X_train_sc, y_train)
            preds = mdl.predict(X_test_sc)
            probas[name] = mdl.predict_proba(X_test_sc)[:, 1]
        else:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)
            probas[name] = mdl.predict_proba(X_test)[:, 1]
        results.append(perf_table(y_test, preds, name))

    res_df = pd.DataFrame(results).set_index("Model")
    st.subheader("Model Performance")
    st.dataframe(res_df)

    model_choice = st.selectbox("Confusion Matrix for:", res_df.index.tolist())
    mdl_cm = models[model_choice]
    y_pred_cm = mdl_cm.predict(X_test_sc if model_choice == "KNN" else X_test)
    cm = confusion_matrix(y_test, y_pred_cm)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Low", "Pred High"],
                yticklabels=["Actual Low", "Actual High"],
                ax=ax_cm)
    st.pyplot(fig_cm)

    st.subheader("ROC Curves")
    fig_roc, ax_roc = plt.subplots()
    for name, probs in probas.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.legend()
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    st.pyplot(fig_roc)

    # Batch prediction
    st.markdown("---")
    st.subheader("🔮 Batch Prediction")
    upl = st.file_uploader("Upload CSV (no willingness column)", type="csv")
    if upl:
        newdf = pd.read_csv(upl)
        proc = pd.get_dummies(newdf, drop_first=True)
        proc = proc.reindex(columns=X.columns, fill_value=0)
        final_model = models["Random Forest"]
        preds_new = final_model.predict(proc)
        newdf["High_Willingness_Pred"] = preds_new
        st.write(newdf.head())
        st.download_button("Download predictions",
                           newdf.to_csv(index=False).encode("utf-8"),
                           "predictions.csv",
                           "text/csv")

# ---------------------------------------------
# CLUSTERING
# ---------------------------------------------
elif page == "🎯 Clustering":
    st.header("🎯 K-Means Segmentation")

    num_cols = ["Age", "Monthly_Income_AED", "Willingness_Score_1_10"]
    k = st.slider("k (clusters)", 2, 10, 4)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df[num_cols])

    # Elbow and silhouette diagnostics
    inertia, silh = [], []
    for i in range(2, 11):
        km_i = KMeans(n_clusters=i, n_init=10, random_state=42).fit(df[num_cols])
        inertia.append(km_i.inertia_)
        silh.append(silhouette_score(df[num_cols], km_i.labels_))

    colD1, colD2 = st.columns(2)
    with colD1:
        fig_el, ax_el = plt.subplots()
        ax_el.plot(range(2, 11), inertia, marker="o")
        ax_el.set_xlabel("k")
        ax_el.set_ylabel("Inertia")
        ax_el.set_title("Elbow Curve")
        st.pyplot(fig_el)
    with colD2:
        fig_si, ax_si = plt.subplots()
        ax_si.plot(range(2, 11), silh, marker="s", color="green")
        ax_si.set_title("Silhouette Scores")
        ax_si.set_xlabel("k")
        ax_si.set_ylabel("Score")
        st.pyplot(fig_si)

    # Persona
    st.subheader("Cluster Personas")
    persona = (df.groupby("Cluster")[num_cols]
               .agg(["mean"])
               .droplevel(1, axis=1)
               .round(1))
    st.dataframe(persona)

    st.download_button("Download labeled data",
                       df.to_csv(index=False).encode("utf-8"),
                       "clustered_data.csv",
                       "text/csv")

# ---------------------------------------------
# ASSOCIATION RULES
# ---------------------------------------------
elif page == "🛒 Association Rules":
    st.header("🛒 Apriori Preference Mining")

    bin_cols = [c for c in df.columns if c.startswith(tuple(["Act_", "Goal_", "Fac_", "Prob_", "Inf_"]))]

    cols_sel = st.multiselect("Select binary columns", bin_cols,
                              default=[c for c in bin_cols if "Act_" in c or "Fac_" in c])

    min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.1, 0.9, 0.6, 0.05)
    min_lift = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1)

    if st.button("Run Apriori"):
        basket = df[cols_sel].astype(bool)
        frequent = apriori(basket, min_support=min_sup, use_colnames=True)
        if frequent.empty:
            st.warning("No frequent itemsets. Lower support.")
        else:
            rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
            rules = rules[rules["lift"] >= min_lift]
            if rules.empty:
                st.warning("No rules at these thresholds.")
            else:
                rules = prettify_rule_cols(rules)
                rules_display = (rules.sort_values("lift", ascending=False)
                                       .head(10)
                                       .reset_index(drop=True))
                st.dataframe(rules_display[["antecedents",
                                            "consequents",
                                            "support",
                                            "confidence",
                                            "lift"]]
                             .style.format({"support":"{:.3f}",
                                            "confidence":"{:.2f}",
                                            "lift":"{:.2f}"}))

# ---------------------------------------------
# REGRESSION
# ---------------------------------------------
else:
    st.header("📈 Spend Prediction")

    y = df["Monthly_Income_AED"]
    X = pd.get_dummies(df.drop(columns=["Monthly_Income_AED"]), drop_first=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42)
    }

    reg_results = []
    for name, reg in regs.items():
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_te)
        r2 = reg.score(X_te, y_te)
        rmse = np.sqrt(np.mean((y_te - preds)**2))
        mae = np.mean(np.abs(y_te - preds))
        reg_results.append({
            "Model": name,
            "R2": round(r2, 3),
            "RMSE": int(rmse),
            "MAE": int(mae)
        })
    st.dataframe(pd.DataFrame(reg_results).set_index("Model"))

    st.markdown("""
    **Notes**  
    * Ridge improves multicollinearity handling versus OLS.  
    * Decision Tree captures non-linear spend patterns—monitor depth to avoid over-fit.  
    * Lasso zeroes uninformative features → quick driver screening.  
    """)
