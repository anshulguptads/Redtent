import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from kmodes.kprototypes import KPrototypes
from mlxtend.frequent_patterns import apriori, association_rules

# ───────────────────────────────────────────────────────────────
# PAGE CONFIG & DATA
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Luxury Gym Intelligence", page_icon="💪", layout="wide")

@st.cache_data
def load_data(csv_path: str = "luxury_gym_survey_wide.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)

df = load_data()

# ───────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ───────────────────────────────────────────────────────────────
st.sidebar.title("🏋️‍♀️ Modules")
page = st.sidebar.radio(
    "Choose analytics module",
    ["📊 Descriptive Analytics",
     "🤖 Classification",
     "🎯 Clustering (K-Prototypes)",
     "🛒 Association Rules",
     "📈 Regression"]
)

# ───────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ───────────────────────────────────────────────────────────────
def score_row(y_true, y_pred, name):
    return {"Model": name,
            "Accuracy": round(accuracy_score(y_true, y_pred), 3),
            "Precision": round(precision_score(y_true, y_pred), 3),
            "Recall": round(recall_score(y_true, y_pred), 3),
            "F1": round(f1_score(y_true, y_pred), 3)}

def prettify_rules(rules_df):
    for c in ("antecedents", "consequents"):
        rules_df[c] = rules_df[c].apply(lambda x: ", ".join(sorted(list(x))))
    return rules_df

# ───────────────────────────────────────────────────────────────
# 1️⃣  DESCRIPTIVE
# ───────────────────────────────────────────────────────────────
if page == "📊 Descriptive Analytics":
    st.header("📊 Descriptive Insights")
    with st.sidebar.expander("Filters", True):
        age_rng = st.slider("Age", int(df.Age.min()), int(df.Age.max()), (25, 45))
        inc_rng = st.slider("Income (AED)", int(df.Monthly_Income_AED.min()),
                            int(df.Monthly_Income_AED.max()), (5_000, 20_000), step=1_000)
        g_opt   = st.multiselect("Gender", df.Gender.unique(), default=list(df.Gender.unique()))
        show_raw = st.checkbox("Show raw rows")
    view = df[(df.Age.between(*age_rng)) &
              (df.Monthly_Income_AED.between(*inc_rng)) &
              (df.Gender.isin(g_opt))]
    st.success(f"Filtered respondents: {len(view)}")
    if show_raw:
        st.dataframe(view.head())

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Income distribution")
        fig, ax = plt.subplots()
        sns.histplot(view["Monthly_Income_AED"], kde=True, ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Willingness distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Willingness_Score_1_10", data=view, ax=ax2)
        st.pyplot(fig2)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Willingness vs Age")
        fig3 = px.scatter(view, x="Age", y="Willingness_Score_1_10", opacity=0.6)
        if len(view) > 1:
            coef = np.polyfit(view["Age"], view["Willingness_Score_1_10"], 1)
            lin_x = np.linspace(view["Age"].min(), view["Age"].max(), 100)
            lin_y = coef[0] * lin_x + coef[1]
            fig3.add_scatter(x=lin_x, y=lin_y, mode="lines",
                             name="Linear fit", line=dict(dash="dash"))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("Facility preference frequency")
        fac_cols = [c for c in df.columns if c.startswith("Fac_")]
        fac_freq = view[fac_cols].mean().sort_values(ascending=False)
        fig4, ax4 = plt.subplots()
        sns.barplot(y=fac_freq.index.str.replace("Fac_", ""),
                    x=fac_freq.values, ax=ax4)
        ax4.set_xlabel("Selection rate");  ax4.set_ylabel("")
        st.pyplot(fig4)

# ───────────────────────────────────────────────────────────────
# 2️⃣  CLASSIFICATION
# ───────────────────────────────────────────────────────────────
elif page == "🤖 Classification":
    st.header("🤖 High-Willingness Classifier")
    y = (df["Willingness_Score_1_10"] >= 7).astype(int)
    X = pd.get_dummies(df.drop(columns=["Willingness_Score_1_10"]), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=0.25, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_sc, X_test_sc = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "KNN":               KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":     DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    scores, probas = [], {}
    for name, mdl in models.items():
        if name == "KNN":
            mdl.fit(X_train_sc, y_train)
            preds = mdl.predict(X_test_sc)
            probas[name] = mdl.predict_proba(X_test_sc)[:, 1]
        else:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)
            probas[name] = mdl.predict_proba(X_test)[:, 1]
        scores.append(score_row(y_test, preds, name))

    st.subheader("Metrics")
    st.dataframe(pd.DataFrame(scores).set_index("Model"))

    choice = st.selectbox("Confusion matrix for:", [s["Model"] for s in scores])
    sel_model = models[choice]
    y_pred = sel_model.predict(X_test_sc if choice == "KNN" else X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Low", "Pred High"],
                yticklabels=["Actual Low", "Actual High"], ax=ax_cm)
    st.pyplot(fig_cm)

    st.subheader("ROC curves")
    fig_roc, ax_roc = plt.subplots()
    for name, pr in probas.items():
        fpr, tpr, _ = roc_curve(y_test, pr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray");  ax_roc.legend()
    st.pyplot(fig_roc)

# ───────────────────────────────────────────────────────────────
# 3️⃣  HYBRID CLUSTERING (K-PROTOTYPES)
# ───────────────────────────────────────────────────────────────
elif page == "🎯 Clustering (K-Prototypes)":
    st.header("🎯 K-Prototypes Segmentation")

    num_cols = ["Age", "Monthly_Income_AED", "Willingness_Score_1_10"]
    # everything else (except pre-existing 'Cluster') is categorical
    cat_cols = [c for c in df.columns if c not in num_cols + ["Cluster"]]

    # ── Robust NaN handling ────────────────────────────────────
    df_clean = df.copy()

    # numeric → mean imputation
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())

    # categorical:
    binary_cols = [c for c in cat_cols                # 0/1 flags
                   if pd.api.types.is_numeric_dtype(df[c])]
    multi_cols  = [c for c in cat_cols if c not in binary_cols]

    df_clean[binary_cols] = df_clean[binary_cols].fillna(0).astype(int)
    df_clean[multi_cols]  = df_clean[multi_cols].fillna("Missing").astype(str)

    # ── Build mixed matrix for K-Prototypes ────────────────────
    scaler   = StandardScaler()
    X_num    = scaler.fit_transform(df_clean[num_cols])
    X_cat    = df_clean[cat_cols].to_numpy()           # ints & strings OK
    X_mix    = np.hstack([X_num, X_cat])

    cat_idx  = list(range(X_num.shape[1], X_mix.shape[1]))  # positions of categorical cols

    # UI controls
    k  = st.slider("k (clusters)", 2, 10, 4)
    g  = st.number_input("γ (numeric-vs-categorical weight – 0 = auto)", 0.0, 10.0, 0.0, 0.1)
    γ  = None if g == 0 else g

    kp = KPrototypes(n_clusters=k, init="Huang", n_init=10,
                     gamma=γ, random_state=42, verbose=0)

    clusters = kp.fit_predict(X_mix, categorical=cat_idx)
    df["Cluster"] = clusters
    st.success(f"Clustering complete → {k} segments")

    # ── Cost curve (diagnostic) ───────────────────────────────
    costs = []
    for ki in range(2, 11):
        km = KPrototypes(n_clusters=ki, n_init=5, random_state=42)
        km.fit_predict(X_mix, categorical=cat_idx)
        costs.append(km.cost_)
    fig_cost, ax_cost = plt.subplots()
    ax_cost.plot(range(2, 11), costs, marker="o")
    ax_cost.set(xlabel="k", ylabel="Cost", title="Cost curve")
    st.pyplot(fig_cost)

    # ── Persona table ─────────────────────────────────────────
    persona_num = df.groupby("Cluster")[num_cols].mean().round(1)
    persona_cat = df.groupby("Cluster")[multi_cols].agg(lambda s: s.mode().iloc[0])
    persona_bin = df.groupby("Cluster")[binary_cols].mean().round(2)

    persona = pd.concat([persona_num, persona_cat, persona_bin], axis=1)
    st.subheader("Cluster personas")
    st.dataframe(persona)

    st.download_button("Download labelled data",
                       df.to_csv(index=False).encode("utf-8"),
                       "clustered_data.csv",
                       "text/csv")

# ───────────────────────────────────────────────────────────────
# 4️⃣  ASSOCIATION RULES
# ───────────────────────────────────────────────────────────────
elif page == "🛒 Association Rules":
    st.header("🛒 Preference associations (Apriori)")

    bin_cols = [c for c in df.columns if c.startswith(("Act_", "Goal_", "Fac_", "Prob_", "Inf_"))]
    sel = st.multiselect("Binary columns", bin_cols,
                         default=[c for c in bin_cols if "Act_" in c or "Fac_" in c])

    min_sup  = st.slider("Min support",     0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence",  0.1,  0.9, 0.6,  0.05)
    min_lift = st.slider("Min lift",        1.0,  5.0, 1.2,  0.1)

    if st.button("Run Apriori"):
        basket = df[sel].astype(bool)
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        if freq.empty:
            st.warning("No itemsets — lower support.")
        else:
            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
            rules = rules[rules["lift"] >= min_lift]
            if rules.empty:
                st.warning("No rules at these thresholds.")
            else:
                rules = prettify_rules(rules).sort_values("lift", ascending=False).head(10)
                st.dataframe(rules[["antecedents", "consequents",
                                    "support", "confidence", "lift"]]
                             .style.format({"support":"{:.3f}",
                                            "confidence":"{:.2f}",
                                            "lift":"{:.2f}"}))

# ───────────────────────────────────────────────────────────────
# 5️⃣  REGRESSION
# ───────────────────────────────────────────────────────────────
else:
    st.header("📈 Regression – Income prediction")
    y = df["Monthly_Income_AED"]
    X = pd.get_dummies(df.drop(columns=["Monthly_Income_AED"]), drop_first=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    regs = {"Linear": LinearRegression(),
            "Ridge":  Ridge(alpha=1.0),
            "Lasso":  Lasso(alpha=0.001),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42)}
    out = []
    for name, r in regs.items():
        r.fit(X_tr, y_tr);  preds = r.predict(X_te)
        out.append({"Model": name,
                    "R2":   round(r.score(X_te, y_te), 3),
                    "RMSE": int(np.sqrt(((y_te - preds) ** 2).mean())),
                    "MAE":  int(np.abs(y_te - preds).mean())})
    st.dataframe(pd.DataFrame(out).set_index("Model"))
