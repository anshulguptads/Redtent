
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from kmodes.kprototypes import KPrototypes
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Luxury Gym Intelligence", page_icon="💪", layout="wide")

@st.cache_data
def load_data(path="luxury_gym_survey_wide.csv"):
    return pd.read_csv(path)

df = load_data()

st.sidebar.title("🏋️‍♀️ Modules")
page = st.sidebar.radio(
    "Choose analytics module",
    ["📊 Descriptive Analytics",
     "🤖 Classification",
     "🎯 Clustering (K-Prototypes)",
     "🛒 Association Rules",
     "📈 Regression"]
)

def perf_row(y_true, y_pred, model):
    return {"Model": model,
            "Accuracy": round(accuracy_score(y_true, y_pred),3),
            "Precision": round(precision_score(y_true, y_pred),3),
            "Recall": round(recall_score(y_true, y_pred),3),
            "F1": round(f1_score(y_true, y_pred),3)}

def pretty_rules(r):
    for c in ["antecedents","consequents"]:
        r[c] = r[c].apply(lambda x: ", ".join(sorted(list(x))))
    return r

# ---------- DESCRIPTIVE ----------
if page == "📊 Descriptive Analytics":
    st.header("📊 Descriptive Insights")
    with st.sidebar.expander("Filters", True):
        age_rng = st.slider("Age", int(df.Age.min()), int(df.Age.max()), (25,45))
        inc_rng = st.slider("Income (AED)", int(df.Monthly_Income_AED.min()),
                            int(df.Monthly_Income_AED.max()), (5000,20000), step=1000)
        gender_opt = st.multiselect("Gender", df.Gender.unique(), default=list(df.Gender.unique()))
        show_raw = st.checkbox("Show raw")
    dff = df[(df.Age.between(*age_rng)) &
             (df.Monthly_Income_AED.between(*inc_rng)) &
             (df.Gender.isin(gender_opt))]
    st.success(f"Filtered sample: {len(dff)}")
    if show_raw: st.dataframe(dff.head())

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(dff["Monthly_Income_AED"], kde=True, ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Willingness Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Willingness_Score_1_10", data=dff, ax=ax2)
        st.pyplot(fig2)
    c3,c4 = st.columns(2)
    with c3:
        st.subheader("Willingness vs Age")
        fig3 = px.scatter(dff, x="Age", y="Willingness_Score_1_10", opacity=0.6)
        if len(dff)>1:
            coef = np.polyfit(dff["Age"], dff["Willingness_Score_1_10"],1)
            x_line = np.linspace(dff["Age"].min(), dff["Age"].max(),100)
            y_line = coef[0]*x_line + coef[1]
            fig3.add_scatter(x=x_line, y=y_line, mode="lines", name="Fit", line=dict(dash="dash"))
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        st.subheader("Facility Frequency")
        fac_cols=[c for c in df.columns if c.startswith("Fac_")]
        fac_freq = dff[fac_cols].mean().sort_values(ascending=False)
        fig4, ax4 = plt.subplots()
        sns.barplot(y=fac_freq.index.str.replace("Fac_",""), x=fac_freq.values, ax=ax4)
        ax4.set_xlabel("Frequency")
        ax4.set_ylabel("")
        st.pyplot(fig4)

# ---------- CLASSIFICATION ----------
elif page == "🤖 Classification":
    st.header("🤖 High-Willingness Classifier")
    y = (df["Willingness_Score_1_10"] >= 7).astype(int)
    X = pd.get_dummies(df.drop(columns=["Willingness_Score_1_10"]), drop_first=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr_sc, X_te_sc = scaler.transform(X_tr), scaler.transform(X_te)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    rows, probas = [], {}
    for n,m in models.items():
        if n=="KNN":
            m.fit(X_tr_sc,y_tr); preds=m.predict(X_te_sc); probas[n]=m.predict_proba(X_te_sc)[:,1]
        else:
            m.fit(X_tr,y_tr); preds=m.predict(X_te); probas[n]=m.predict_proba(X_te)[:,1]
        rows.append(perf_row(y_te,preds,n))
    st.dataframe(pd.DataFrame(rows).set_index("Model"))
    sel = st.selectbox("Confusion Matrix", [r["Model"] for r in rows])
    mdl = models[sel]
    p_cm = mdl.predict(X_te_sc if sel=="KNN" else X_te)
    cm = confusion_matrix(y_te,p_cm)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d", cmap="Blues",
                xticklabels=["Pred Low","Pred High"],
                yticklabels=["Actual Low","Actual High"], ax=ax_cm)
    st.pyplot(fig_cm)
    st.subheader("ROC Curves")
    fig_roc, ax_roc = plt.subplots()
    for n,score in probas.items():
        fpr,tpr,_ = roc_curve(y_te,score); ax_roc.plot(fpr,tpr,label=f"{n} (AUC={auc(fpr,tpr):.2f})")
    ax_roc.plot([0,1],[0,1],'--',color='gray'); ax_roc.legend(); st.pyplot(fig_roc)
    st.markdown("---")
    upl = st.file_uploader("Upload CSV (no willingness column)",type="csv")
    if upl:
        new = pd.read_csv(upl)
        proc=pd.get_dummies(new,drop_first=True)
        proc=proc.reindex(columns=X.columns, fill_value=0)
        best=models["Random Forest"]; new["High_Willingness_Pred"]=best.predict(proc)
        st.write(new.head())
        st.download_button("Download predictions", new.to_csv(index=False).encode("utf-8"),
                           "predictions.csv","text/csv")

# ---------- K-PROTOTYPES ----------
elif page == "🎯 Clustering (K-Prototypes)":
    st.header("🎯 Segmentation with K-Prototypes")
    num_cols=["Age","Monthly_Income_AED","Willingness_Score_1_10"]
    cat_cols=[c for c in df.columns if c not in num_cols and c!="Cluster"]
    scaler=StandardScaler()
    num_scaled=scaler.fit_transform(df[num_cols])
    cat_data=df[cat_cols].copy()
    cat_data=cat_data.apply(lambda s: s.astype(int) if s.dtype!=object else s)
    X_mix=np.concatenate([num_scaled,cat_data.values],axis=1)
    cat_idx=list(range(num_scaled.shape[1], X_mix.shape[1]))
    k = st.slider("k (clusters)",2,10,4)
    gamma = st.number_input("gamma (0 = auto)", value=0.0, step=0.1)
    gamma_val=None if gamma==0 else gamma
    kp=KPrototypes(n_clusters=k, init='Huang', n_init=10, verbose=0, random_state=42, gamma=gamma_val)
    clusters=kp.fit_predict(X_mix, categorical=cat_idx)
    df["Cluster"]=clusters
    st.success(f"Clustering complete. k={k}")
    # cost plot
    costs=[]
    for i in range(2,11):
        kp_i=KPrototypes(n_clusters=i, init='Huang', n_init=5, random_state=42)
        kp_i.fit_predict(X_mix, categorical=cat_idx)
        costs.append(kp_i.cost_)
    fig_cost, ax_cost = plt.subplots()
    ax_cost.plot(range(2,11),costs,marker="o"); ax_cost.set_xlabel("k"); ax_cost.set_ylabel("Cost"); ax_cost.set_title("Cost vs k")
    st.pyplot(fig_cost)
    # persona table
    st.subheader("Cluster Personas")
    persona_num=df.groupby("Cluster")[num_cols].mean().round(1)
    persona_cat=df.groupby("Cluster")[cat_cols].agg(lambda x: x.mode().iloc[0])
    persona=pd.concat([persona_num, persona_cat],axis=1)
    st.dataframe(persona)
    st.download_button("Download labeled data", df.to_csv(index=False).encode("utf-8"),
                       "clustered_data.csv","text/csv")

# ---------- ASSOCIATION RULES ----------
elif page == "🛒 Association Rules":
    st.header("🛒 Association Rule Mining")
    bin_cols=[c for c in df.columns if c.startswith(("Act_","Goal_","Fac_","Prob_","Inf_"))]
    cols_sel=st.multiselect("Binary columns", bin_cols, default=[c for c in bin_cols if "Act_" in c or "Fac_" in c])
    sup=st.slider("Min support",0.01,0.5,0.05,0.01)
    conf=st.slider("Min confidence",0.1,0.9,0.6,0.05)
    lift=st.slider("Min lift",1.0,5.0,1.2,0.1)
    if st.button("Run"):
        basket=df[cols_sel].astype(bool)
        freq=apriori(basket,min_support=sup,use_colnames=True)
        if freq.empty:
            st.warning("No itemsets at this support."); st.stop()
        rules=association_rules(freq,metric="confidence",min_threshold=conf)
        rules=rules[rules["lift"]>=lift]
        if rules.empty:
            st.warning("No rules at these thresholds."); st.stop()
        rules=pretty_rules(rules).sort_values("lift",ascending=False).head(10)
        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]]
                     .style.format({"support":"{:.3f}","confidence":"{:.2f}","lift":"{:.2f}"}))

# ---------- REGRESSION ----------
else:
    st.header("📈 Regression – Income Prediction")
    y=df["Monthly_Income_AED"]
    X=pd.get_dummies(df.drop(columns=["Monthly_Income_AED"]), drop_first=True)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    regs={
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42)
    }
    res=[]
    for n,r in regs.items():
        r.fit(X_tr,y_tr); p=r.predict(X_te)
        res.append({"Model":n,
                    "R2":round(r.score(X_te,y_te),3),
                    "RMSE":int(np.sqrt(((y_te-p)**2).mean())),
                    "MAE":int(np.mean(np.abs(y_te-p)))})
    st.dataframe(pd.DataFrame(res).set_index("Model"))
