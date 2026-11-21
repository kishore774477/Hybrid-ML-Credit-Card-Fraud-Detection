"""Streamlit dashboard for the hybrid credit card fraud detector.

Run with:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import DATA_PATH, PCA_COLS, RAW_COLS
from src.hybrid_predict import HybridFraudDetector

st.set_page_config(page_title="Hybrid Credit Card Fraud Detection", page_icon="💳", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data(show_spinner="Loading dataset ...")
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource(show_spinner="Loading models ...")
def load_detector():
    try:
        return HybridFraudDetector(), None
    except Exception as exc:  # noqa: BLE001 - surface any load error in the UI
        return None, str(exc)


def show_result(result: dict) -> None:
    score, thr = result["hybrid_score"], result["threshold"]
    if result["is_fraud"]:
        st.error(f"🚨 **FRAUD DETECTED** — hybrid score {score:.3f} (threshold {thr:.2f})")
    elif result["risk_level"] == "Medium":
        st.warning(f"⚠️ **Suspicious, below threshold** — hybrid score {score:.3f} (threshold {thr:.2f})")
    else:
        st.success(f"✅ **Normal transaction** — hybrid score {score:.3f} (threshold {thr:.2f})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk level", result["risk_level"])
    c2.metric("Random Forest fraud prob.", f"{result['fraud_probability']:.3f}")
    c3.metric("Isolation Forest anomaly", f"{result['anomaly_score']:.3f}")
    c4.metric("Isolation Forest verdict", result["isolation_forest_prediction"])
    st.progress(min(score, 1.0), text="Hybrid fraud score")


st.title("💳 Hybrid Credit Card Fraud Detection")
st.caption(
    "Isolation Forest (unsupervised anomaly detection) + Random Forest (supervised classification), "
    "combined into one weighted hybrid score."
)

detector, load_error = load_detector()
page = st.sidebar.radio(
    "Navigation", ["Dashboard", "Model Performance", "Single Prediction", "Batch Prediction"]
)
if detector:
    meta = detector.metadata
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Hybrid weight (RF):** {detector.weight:.2f}  \n"
        f"**Decision threshold:** {detector.threshold:.2f}  \n"
        f"**Trained:** {meta['trained_at'][:10]}"
    )

# --------------------------------------------------------------------------- Dashboard
if page == "Dashboard":
    st.header("📊 Dataset Overview")
    df = load_data()
    if df is None:
        st.warning(
            "Dataset not found at `data/creditcard.csv`. Download it with "
            "`python -m src.download_data`. Prediction pages still work with the saved models."
        )
    else:
        n_fraud = int(df["Class"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{len(df):,}")
        c2.metric("Fraudulent", f"{n_fraud:,}")
        c3.metric("Fraud rate", f"{n_fraud / len(df):.3%}")
        c4.metric("Total amount", f"${df['Amount'].sum() / 1e6:,.2f}M")

        st.subheader("Sample rows")
        st.dataframe(df.head(10), width="stretch")

        left, right = st.columns(2)
        with left:
            st.subheader("Class distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            counts = df["Class"].value_counts().sort_index()
            sns.barplot(x=["Normal", "Fraud"], y=counts.values, hue=["Normal", "Fraud"], legend=False,
                        ax=ax, palette=["#4c72b0", "#c44e52"])
            ax.set_yscale("log")
            ax.set_ylabel("Count (log scale)")
            st.pyplot(fig)
        with right:
            st.subheader("Amount by class")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            plot_df = df[df["Amount"] > 0].assign(Label=lambda d: d["Class"].map({0: "Normal", 1: "Fraud"}))
            sns.boxplot(data=plot_df, x="Label", y="Amount", hue="Label", legend=False, ax=ax,
                        order=["Normal", "Fraud"], palette={"Normal": "#4c72b0", "Fraud": "#c44e52"})
            ax.set_yscale("log")
            ax.set_xlabel("")
            st.pyplot(fig)

        st.subheader("Fraud by hour of day")
        hours = (df["Time"] // 3600 % 24).astype(int)
        hourly = df.groupby(hours)["Class"].agg(["sum", "count"]).rename(columns={"sum": "Fraud", "count": "All"})
        hourly["Fraud rate (%)"] = hourly["Fraud"] / hourly["All"] * 100
        st.bar_chart(hourly["Fraud rate (%)"])

# --------------------------------------------------------------------------- Model performance
elif page == "Model Performance":
    st.header("📈 Model Performance (held-out test set)")
    if not detector:
        st.error(f"Models not loaded: {load_error}. Train them with `python -m src.train_models`.")
    else:
        metrics = detector.metadata["test_metrics"]
        names = {"isolation_forest": "Isolation Forest", "random_forest": "Random Forest", "hybrid": "Hybrid"}
        table = pd.DataFrame(
            {
                names[k]: {
                    "Precision": v["precision"],
                    "Recall": v["recall"],
                    "F1": v["f1"],
                    "ROC-AUC": v["roc_auc"],
                    "PR-AUC": v["pr_auc"],
                }
                for k, v in metrics.items()
            }
        ).T
        best = "background-color: #2e7d32; color: white; font-weight: bold"
        st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, props=best), width="stretch")
        st.caption(
            f"Test set: {detector.metadata['split']['test']:,} transactions. "
            "PR-AUC is the most informative metric for such an imbalanced dataset."
        )

        st.subheader("Confusion matrices")
        cols = st.columns(3)
        for col, (key, label) in zip(cols, names.items()):
            cm = metrics[key]["confusion_matrix"]
            grid = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
            fig, ax = plt.subplots(figsize=(3.6, 3))
            sns.heatmap(grid, annot=True, fmt=",", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(label)
            col.pyplot(fig)

        st.subheader("Top Random Forest feature importances")
        imp = pd.Series(detector.metadata["feature_importances"]).head(15)
        st.bar_chart(imp)

# --------------------------------------------------------------------------- Single prediction
elif page == "Single Prediction":
    st.header("🔍 Check a Transaction")
    if not detector:
        st.error(f"Models not loaded: {load_error}. Train them with `python -m src.train_models`.")
    else:
        df = load_data()
        defaults = {"Time": 0.0, "Amount": 100.0, **{c: 0.0 for c in PCA_COLS}}
        for c, v in defaults.items():
            st.session_state.setdefault(f"in_{c}", v)

        if df is not None:
            st.write("Load a real transaction from the dataset to try the model:")
            b1, b2, _ = st.columns([1, 1, 3])
            pick = None
            if b1.button("🎲 Random normal"):
                pick = df[df["Class"] == 0].sample(1).iloc[0]
            if b2.button("🎯 Random fraud"):
                pick = df[df["Class"] == 1].sample(1).iloc[0]
            if pick is not None:
                st.session_state.picked = {c: float(pick[c]) for c in RAW_COLS}
                st.session_state.actual = int(pick["Class"])
                for c in RAW_COLS:
                    st.session_state[f"in_{c}"] = float(pick[c])

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            time_val = c1.number_input("Time (seconds since first transaction)", min_value=0.0, key="in_Time")
            amount_val = c2.number_input("Amount ($)", min_value=0.0, key="in_Amount")
            with st.expander("PCA features V1–V28", expanded=False):
                grid = st.columns(4)
                v_values = {
                    c: grid[i % 4].number_input(c, format="%.6f", key=f"in_{c}")
                    for i, c in enumerate(PCA_COLS)
                }
            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            tx = {"Time": time_val, "Amount": amount_val, **v_values}
            show_result(detector.predict(tx))
            if "actual" in st.session_state and tx == st.session_state.get("picked"):
                st.info(f"Actual label in dataset: **{'Fraud' if st.session_state.actual else 'Normal'}**")

# --------------------------------------------------------------------------- Batch prediction
elif page == "Batch Prediction":
    st.header("📁 Batch Prediction")
    if not detector:
        st.error(f"Models not loaded: {load_error}. Train them with `python -m src.train_models`.")
    else:
        st.write(
            "Upload a CSV with the columns `Time, V1 … V28, Amount` "
            "(an optional `Class` column is used to report accuracy)."
        )
        upload = st.file_uploader("CSV file", type="csv")
        if upload is not None:
            batch = pd.read_csv(upload)
            missing = [c for c in RAW_COLS if c not in batch.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                with st.spinner(f"Scoring {len(batch):,} transactions ..."):
                    preds = detector.predict_batch(batch)
                out = pd.concat([batch, preds], axis=1)
                c1, c2, c3 = st.columns(3)
                c1.metric("Transactions", f"{len(out):,}")
                c2.metric("Flagged as fraud", f"{int(out['is_fraud'].sum()):,}")
                c3.metric("High / Medium risk", f"{(out['risk_level'] == 'High').sum():,} / "
                                                f"{(out['risk_level'] == 'Medium').sum():,}")
                if "Class" in out.columns:
                    tp = int(((out["Class"] == 1) & out["is_fraud"]).sum())
                    st.info(
                        f"Caught {tp} of {int(out['Class'].sum())} labelled frauds; "
                        f"{int(((out['Class'] == 0) & out['is_fraud']).sum())} false alarms."
                    )
                st.dataframe(
                    out.sort_values("hybrid_score", ascending=False).head(200), width="stretch"
                )
                st.download_button(
                    "⬇️ Download results CSV",
                    out.to_csv(index=False).encode(),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )
