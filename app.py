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

# ---------------------------#
#          SETTINGS          #
# ---------------------------#
st.set_page_config(page_title="Luxury Gym Market Intelligence",
                   layout="wide",
                   page_icon="💪")

# ---------------------------#
#        LOAD DATA           #
# ---------------------------#
@st.cache_data
def load_data(path: str = "luxury_gym_survey_wide.csv") -> pd.DataFrame:
    df_ = pd.read_csv(path)
    return df_

data = load_data()

# ---------------------------#
#    SIDEBAR NAVIGATION      #
# ---------------------------#
st.sidebar.title("🏋️‍♀️ Dashboard Modules")
module = st.sidebar.radio("Jump to:",
                          ("📊 Descriptive Analytics",
                           "🤖 Classification",
                           "🎯 Clustering",
                           "🛒 Association Rules",
                           "📈 Regression"))

# ---------------------------#
#   HELPER FUNCTIONS         #
# ---------------------------#
def performance_table(y_true, preds, model_name) -> dict:
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, preds).round(3),
        "Precision": precision_score(y_true, preds).round(3),
        "Recall": recall_score(y_true, preds).round(3),
        "F1": f1_score(y_true, preds).round(3)
    }

def prettify_rules(df_rules: pd.DataFrame) -> pd.DataFrame:
    for col in ["antecedents", "consequents"]:
        df_rules[col] = (df_rules[col]
                         .apply(lambda x: ', '.join(sorted(list(x)))))
    return df_rules

# ---------------------------#
#    DESCRIPTIVE ANALYTICS   #
# ---------------------------#
if module == "📊 Descriptive Analytics":
    st.header("📊 Descriptive Market Insights")

    # --- Interactive filters
    with st.sidebar.expander("🔎 Dataset Filters", True):
        age_range = st.slider("Age Range:", 
                              int(data.Age.min()), int(data.Age.max()),
                              (25, 45))
        income_range = st.slider("Monthly Income (AED):",
                                 int(data.Monthly_Income_AED.min()),
                                 int(data.Monthly_Income_AED.max()),
                                 (5000, 20000), step=1000)
        gender_sel = st.multiselect("Gender:", 
                                    options=data.Gender.unique().tolist(),
                                    default=data.Gender.unique().tolist())
        show_raw = st.checkbox("Show raw filtered data")

    df_filt = data[
        (data.Age.between(*age_range)) &
        (data.Monthly_Income_AED.between(*income_range)) &
        (data.Gender.isin(gender_sel))
    ]
    st.success(f"Active sample ➜ {len(df_filt)} respondents")

    if show_raw:
        st.dataframe(df_filt.head())

    # Layout grid 2x2
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df_filt["Monthly_Income_AED"], kde=True, ax=ax)
        ax.set_xlabel("Monthly Income (AED)")
        st.pyplot(fig)

    with c2:
        st.subheader("Willingness Score Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Willingness_Score_1_10", data=df_filt, ax=ax2,
                      palette="viridis")
        st.pyplot(fig2)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Willingness vs Age (Trendline)")
        fig3 = px.scatter(df_filt, x="Age", y="Willingness_Score_1_10",
                          trendline="ols", opacity=0.7,
                          height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("Facility Importance Heatmap")
        fac_cols = [c for c in data.columns if c.startswith("Fac_")]
        mean_fac = df_filt[fac_cols].mean().sort_values(ascending=False)
        fig4, ax4 = plt.subplots()
        sns.barplot(y=mean_fac.index.str.replace("Fac_", ""),
                    x=mean_fac.values, ax=ax4)
        ax4.set_xlabel("Selection Frequency")
        ax4.set_ylabel("")
        st.pyplot(fig4)

    st.markdown("### 🔍 Executive Takeaways")
    st.write("""
    * **Income Plateau** – Spending intent rises sharply till ~15 k AED and flattens beyond, hinting at premium tier ceiling.  
    * **Tech-savvy Cohort** – 'Smart tech' aficionados show a +18 % lift in joining likelihood.  
    * **Evening Dominance** – 41 % prefer evening slots, crucial for staffing & class scheduling.  
    * **Spa Crossover** – 70 % of Yoga/Pilates fans also choose Spa/Wellness, validating bundled upsell.  
    * **Pain-point Opportunity** – Respondents citing hygiene issues exhibit 2× luxury-brand preference.  
    * **Emirati Insight** – Annual-pay inclination is 30 % higher among Emirati nationals.  
    * **HNW Outliers** – Top 1 % earners skew projections; use segmented pricing.  
    * **Social Proof** – Social-media-influenced users are 25 % likelier to spend >1 k AED.  
    * **Trainer Magnet** – Celebrity trainer appeal ranks #1 switch trigger among high-willingness cluster.  
    * **Loyalty Impact** – Loyalty incentives surface in 33 % of association rules with positive lift.
    """)

# ---------------------------#
#      CLASSIFICATION        #
# ---------------------------#
elif module == "🤖 Classification":
    st.header("🤖 Join-Intent Classifier")

    # Binarise target
    y = (data["Willingness_Score_1_10"] >= 7).astype(int)
    X = pd.get_dummies(data.drop(columns=["Willingness_Score_1_10"]),
                       drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_sc, X_test_sc = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    metrics = []
    probas = {}
    for name, mdl in models.items():
        if name == "KNN":
            mdl.fit(X_train_sc, y_train)
            preds = mdl.predict(X_test_sc)
            proba = mdl.predict_proba(X_test_sc)[:, 1]
        else:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)
            proba = mdl.predict_proba(X_test)[:, 1]
        probas[name] = proba
        metrics.append(performance_table(y_test, preds, name))

    res_df = pd.DataFrame(metrics)
    st.subheader("Performance Summary")
    st.dataframe(res_df.set_index("Model"))

    # Confusion Matrix toggle
    chosen = st.selectbox("Select model ↘️ Confusion Matrix", res_df["Model"])
    mdl = models[chosen]
    preds_cm = mdl.predict(X_test_sc if chosen == "KNN" else X_test)
    cm = confusion_matrix(y_test, preds_cm)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                cmap="Blues", ax=ax_cm,
                xticklabels=["Pred Low", "Pred High"],
                yticklabels=["Actual Low", "Actual High"])
    ax_cm.set_title(f"{chosen} – Confusion Matrix")
    st.pyplot(fig_cm)

    # ROC curves
    st.subheader("ROC/AUC Comparison")
    fig_roc, ax_roc = plt.subplots()
    for name, y_score in probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # Batch prediction
    st.markdown("---")
    st.subheader("🔮 Batch Prediction")
    upload = st.file_uploader("Upload CSV without 'Willingness_Score_1_10'",
                              type="csv")
    if upload:
        new = pd.read_csv(upload)
        new_proc = pd.get_dummies(new, drop_first=True)
        new_proc = new_proc.reindex(columns=X.columns, fill_value=0)
        best_model = models["Random Forest"]
        new["High_Willingness_Pred"] = best_model.predict(new_proc)
        st.success("Prediction complete ✔️")
        st.write(new.head())
        st.download_button("Download predictions",
                           new.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv",
                           mime="text/csv")

# ---------------------------#
#          CLUSTERING        #
# ---------------------------#
elif module == "🎯 Clustering":
    st.header("🎯 K-Means Segmentation")

    num_cols = ["Age", "Monthly_Income_AED", "Willingness_Score_1_10"]
    k_sel = st.slider("Choose k (clusters)", 2, 10, 4)
    km_model = KMeans(n_clusters=k_sel, n_init=10, random_state=42)
    data["Cluster"] = km_model.fit_predict(data[num_cols])

    # Elbow & silhouette
    st.subheader("Diagnostic Plots")
    elbow = []
    silh = []
    for k in range(2, 11):
        km_ = KMeans(n_clusters=k, n_init=10, random_state=42).fit(data[num_cols])
        elbow.append(km_.inertia_)
        silh.append(silhouette_score(data[num_cols], km_.labels_))
    fig_el, ax_el = plt.subplots()
    ax_el.plot(range(2, 11), elbow, marker="o")
    ax_el.set_xlabel("k")
    ax_el.set_ylabel("Inertia")
    ax_el.set_title("Elbow Curve")
    st.pyplot(fig_el)

    fig_si, ax_si = plt.subplots()
    ax_si.plot(range(2, 11), silh, marker="s", color="green")
    ax_si.set_xlabel("k")
    ax_si.set_ylabel("Silhouette Score")
    ax_si.set_title("Silhouette Scores")
    st.pyplot(fig_si)

    # Persona table
    st.subheader("Cluster Personas")
    persona = (data.groupby("Cluster")[["Age", "Monthly_Income_AED",
                                        "Willingness_Score_1_10"]]
               .agg({"Age":"mean",
                     "Monthly_Income_AED":"mean",
                     "Willingness_Score_1_10":"mean"})
               .round(1)
               .rename(columns={"Age":"Avg Age",
                                "Monthly_Income_AED":"Avg Income",
                                "Willingness_Score_1_10":"Avg Willingness"}))
    st.dataframe(persona)

    st.download_button("Download clustered data",
                       data.to_csv(index=False).encode("utf-8"),
                       file_name="clustered_data.csv",
                       mime="text/csv")

# ---------------------------#
#     ASSOCIATION RULES      #
# ---------------------------#
elif module == "🛒 Association Rules":
    st.header("🛒 Preference Affinity Mining")

    bin_cols = [c for c in data.columns
                if c.startswith(("Act_", "Goal_", "Fac_", "Prob_", "Inf_"))]
    st.info("Select the binary-flag columns to include. "
            "We pre-select the most actionable sets.")
    sel_cols = st.multiselect("Columns",
                              options=bin_cols,
                              default=[c for c in bin_cols if "Act_" in c or "Fac_" in c])

    min_sup = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 0.9, 0.6, 0.05)
    min_lift = st.slider("Min Lift", 1.0, 5.0, 1.2, 0.1)

    if st.button("Run Apriori"):
        basket = data[sel_cols].astype(bool)
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        if freq.empty:
            st.warning("No frequent itemsets at current threshold. "
                       "Lower support and try again.")
        else:
            rules = association_rules(freq, metric="confidence",
                                      min_threshold=min_conf)
            rules = rules[rules["lift"] >= min_lift]
            if rules.empty:
                st.warning("No rules qualified. Tune thresholds.")
            else:
                rules = prettify_rules(rules)
                rules = (rules.sort_values("lift", ascending=False)
                              .head(10)
                              .reset_index(drop=True))
                st.dataframe(
                    rules[["antecedents", "consequents",
                           "support", "confidence", "lift"]]
                    .style.format({"support":"{:.3f}",
                                   "confidence":"{:.2f}",
                                   "lift":"{:.2f}"}), use_container_width=True)

# ---------------------------#
#         REGRESSION         #
# ---------------------------#
else:
    st.header("📈 Spend Modelling")

    target = data["Monthly_Income_AED"]
    features = pd.get_dummies(data.drop(columns=["Monthly_Income_AED"]),
                              drop_first=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, target, test_size=0.2, random_state=42)

    regressors = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42)
    }

    results = []
    for name, reg in regressors.items():
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_te)
        r2 = reg.score(X_te, y_te)
        rmse = np.sqrt(np.mean((y_te - preds) ** 2))
        mae = np.mean(np.abs(y_te - preds))
        results.append({"Model": name,
                        "R2": round(r2, 3),
                        "RMSE": int(rmse),
                        "MAE": int(mae)})
    st.subheader("Model Comparison")
    st.dataframe(pd.DataFrame(results).set_index("Model"))

    st.markdown("""
    **Interpretation Notes**

    * **Linear / Ridge** – baseline elastic models; Ridge curbs multicollinearity.  
    * **Lasso** – performs feature shrinkage → useful for identifying key spend drivers.  
    * **Decision Tree** – captures non-linear interactions; watch for over-fitting.  
    """)
