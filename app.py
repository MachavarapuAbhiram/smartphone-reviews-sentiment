from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

from train_day1 import clean_review_text, map_rating_to_sentiment

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
MODEL_PATH = Path("models/best_sentiment_pipeline.joblib")
LABELS_PATH = Path("models/sentiment_labels.joblib")
DATA_PATH = Path("data/smartphone_reviews.csv")
METRICS_PATH = Path("outputs/day1_model_metrics.csv")
CM_PATH = Path("outputs/confusion_matrix.png")
ERRORS_PATH = Path("outputs/sample_misclassifications.csv")

ASPECT_KEYWORDS = {
    "Battery": [
        r"\bbattery\b",
        r"\bcharg(?:e|ing|er)\b",
        r"\bmah\b",
        r"\bbackup\b",
        r"\bdrain(?:s|age|ing)?\b",
        r"\badapter\b",
        r"\btop[- ]?up\b",
    ],
    "Camera": [
        r"\bcamera\b",
        r"\bphoto(?:s|graphy)?\b",
        r"\bselfie(?:s)?\b",
        r"\bportrait(?:s)?\b",
        r"\bdaylight\b",
        r"\blow[- ]?light\b",
        r"\bnight\b",
        r"\bsnap(?:s)?\b",
        r"\bvideo(?:s)?\b",
        r"\bois\b",
        r"\bfocus(?:es)?\b",
    ],
    "Display": [
        r"\bdisplay\b",
        r"\bscreen\b",
        r"\bamoled\b",
        r"\boled\b",
        r"\blcd\b",
        r"\bbrightness\b",
        r"\bsunlight\b",
        r"\boutdoors\b",
        r"\brefresh rate\b",
        r"\b(?:90|120|144)\s*hz\b",
        r"\btouch\b",
        r"\bbezels?\b",
        r"\bflicker(?:s|ing)?\b",
    ],
    "Heating/Performance": [
        r"\bheat(?:s|ing)?\b",
        r"\bwarm\b",
        r"\bcool\b",
        r"\bperformance\b",
        r"\bprocessor\b",
        r"\blag(?:s|ging)?\b",
        r"\bstutter(?:s|ing)?\b",
        r"\bgaming\b",
        r"\bbgmi\b",
        r"\bmultitasking\b",
        r"\bspeed\b",
        r"\bslow(?:s)?\b",
        r"\bsluggish\b",
        r"\bframe drops?\b",
        r"\bsmooth(?:ly)?\b",
        r"\bsnappy\b",
        r"\bresponsive\b",
    ],
}

SAMPLE_REVIEWS = {
    "Positive (All-rounder)": (
        "Bought the Vivo T1 44W recently. Amazing 120Hz AMOLED display stays "
        "bright outdoors, fast charging tops up the battery in under an hour, "
        "and daylight camera clicks crisp photos with zero heating issues."
    ),
    "Negative (Heating & Battery)": (
        "Phone heats up quickly during BGMI gaming and battery drains fast on "
        "5G. UI has too much bloatware and indoor photos look blurry. Not worth "
        "the money at all."
    ),
    "Mixed (Aspect Contrast)": (
        "Daylight camera is sharp and display colors look vibrant, but battery "
        "drains fast on mobile data and barely lasts till evening. Performance "
        "heats up quickly during BGMI gaming and lags badly."
    ),
    "Neutral (Average)": (
        "Decent phone overall, nothing extraordinary. Display is fine indoors "
        "while low light photos are average, and battery gets through a "
        "workday with light use."
    ),
}

CLAUSE_SPLIT_PATTERN = re.compile(
    r"[.!?;,]+|\b(?:but|however|although|while|though|yet|and)\b",
    flags=re.IGNORECASE,
)

SENTIMENT_COLORS = {
    "Positive": ("#15803d", "#dcfce7"),
    "Neutral": ("#b45309", "#fef3c7"),
    "Negative": ("#b91c1c", "#fee2e2"),
}


# ==========================================
# CACHED LOADERS
# ==========================================
@st.cache_resource
def load_model_artifacts():
    pipeline = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    labels = joblib.load(LABELS_PATH) if LABELS_PATH.exists() else None
    return pipeline, labels


@st.cache_data
def load_reviews_data() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    if "rating" in df.columns and "sentiment" not in df.columns:
        df["sentiment"] = pd.to_numeric(df["rating"], errors="coerce").apply(
            lambda r: map_rating_to_sentiment(r) if pd.notna(r) else None
        )
    return df


@st.cache_data
def load_metrics_data() -> pd.DataFrame | None:
    return pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None


@st.cache_data
def load_misclassifications_data() -> pd.DataFrame | None:
    return pd.read_csv(ERRORS_PATH) if ERRORS_PATH.exists() else None


@st.cache_data
def compute_aspect_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mention counts and negative-sentiment rates across aspects."""
    text_series = df["review_text"].fillna("").astype(str)
    rows = []
    total = len(df)
    for aspect, patterns in ASPECT_KEYWORDS.items():
        combined_re = re.compile("|".join(patterns), flags=re.IGNORECASE)
        mask = text_series.str.contains(combined_re, regex=True)
        mentioned = df[mask]
        count = int(mask.sum())
        neg_count = int((mentioned["sentiment"] == "Negative").sum()) if count else 0
        pos_count = int((mentioned["sentiment"] == "Positive").sum()) if count else 0
        neu_count = int((mentioned["sentiment"] == "Neutral").sum()) if count else 0
        neg_rate = round((neg_count / count) * 100, 1) if count else 0.0
        rows.append(
            {
                "Aspect": aspect,
                "Mentions": count,
                "Share (%)": round((count / total) * 100, 1) if total else 0.0,
                "Positive": pos_count,
                "Neutral": neu_count,
                "Negative": neg_count,
                "Negative Rate (%)": neg_rate,
            }
        )
    return pd.DataFrame(rows)


# ==========================================
# NLP & PREDICTION HELPERS
# ==========================================
def inject_responsive_css() -> None:
    st.markdown(
        """
        <style>
        .sentiment-badge {
            display: inline-block;
            padding: 10px 18px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 1.15rem;
            margin: 4px 0 8px 0;
        }
        .aspect-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.85rem;
        }
        .aspect-card {
            padding: 12px 14px;
            border-radius: 10px;
            box-sizing: border-box;
        }
        .aspect-card-title {
            font-size: 0.85rem;
            font-weight: 600;
        }
        .aspect-card-sentiment {
            font-size: 1.08rem;
            font-weight: 700;
            margin-top: 2px;
        }
        .aspect-card-conf {
            font-size: 0.8rem;
            margin-top: 2px;
            opacity: 0.9;
        }
        @media (max-width: 768px) {
            .block-container {
                padding: 1.1rem 0.9rem 2rem 0.9rem !important;
            }
            h1 {
                font-size: 1.55rem !important;
            }
            [data-baseweb="tab-list"] {
                overflow-x: auto !important;
                flex-wrap: nowrap !important;
                gap: 0.35rem !important;
            }
            [data-baseweb="tab"] {
                white-space: nowrap !important;
                padding-left: 0.7rem !important;
                padding-right: 0.7rem !important;
            }
            .sentiment-badge {
                display: block;
                width: 100%;
                box-sizing: border-box;
                text-align: center;
                font-size: 1.05rem;
            }
            .aspect-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.55rem;
            }
            .stButton > button {
                min-height: 2.65rem;
                font-size: 0.9rem;
            }
            [data-testid="stMetric"] {
                padding: 0.45rem 0.6rem;
                border-radius: 8px;
                background: rgba(128, 128, 128, 0.06);
            }
        }
        @media (max-width: 420px) {
            .aspect-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sentiment_badge(label: str, subtitle: str = "") -> str:
    fg, bg = SENTIMENT_COLORS.get(label, ("#374151", "#f3f4f6"))
    extra = (
        f"<div style='font-size:0.85rem;margin-top:4px;opacity:0.9;'>{subtitle}</div>"
        if subtitle
        else ""
    )
    return (
        f"<div class='sentiment-badge' "
        f"style='background:{bg};color:{fg};border:1px solid {fg}33;'>"
        f"Predicted Sentiment: {label}{extra}</div>"
    )


def get_prediction_scores(pipeline, cleaned_text: str) -> tuple[str, pd.DataFrame | None, str]:
    pred = str(pipeline.predict([cleaned_text])[0])
    classes = list(getattr(pipeline, "classes_", []))

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([cleaned_text])[0]
        score_df = pd.DataFrame({"Sentiment": classes, "Score": probs}).sort_values(
            "Score", ascending=False
        )
        return pred, score_df, "Probability"

    if hasattr(pipeline, "decision_function"):
        raw = pipeline.decision_function([cleaned_text])[0]
        if hasattr(raw, "__len__"):
            scores = list(raw)
        else:
            scores = [-float(raw), float(raw)]
        score_df = pd.DataFrame({"Sentiment": classes, "Score": scores}).sort_values(
            "Score", ascending=False
        )
        return pred, score_df, "Decision Score"

    return pred, None, ""


def get_top_influential_features(
    pipeline, cleaned_text: str, target_label: str, top_n: int = 10
) -> pd.DataFrame | None:
    """Return top influential TF-IDF n-grams for LogisticRegression predictions."""
    if not hasattr(pipeline, "named_steps"):
        return None
    tfidf = pipeline.named_steps.get("tfidf")
    clf = pipeline.named_steps.get("classifier")
    if tfidf is None or not isinstance(clf, LogisticRegression):
        return None

    vec = tfidf.transform([cleaned_text])
    nz_cols = vec.nonzero()[1]
    if len(nz_cols) == 0:
        return None

    classes = list(clf.classes_)
    if target_label not in classes:
        return None
    class_idx = classes.index(target_label)

    # Binary LogisticRegression stores a single row in coef_ for classes_[1]
    if clf.coef_.shape[0] == 1:
        sign = 1.0 if class_idx == 1 else -1.0
        coefs = sign * clf.coef_[0]
    else:
        coefs = clf.coef_[class_idx]

    feature_names = tfidf.get_feature_names_out()
    rows = []
    for col_idx in nz_cols:
        tfidf_val = float(vec[0, col_idx])
        coef_val = float(coefs[col_idx])
        contribution = tfidf_val * coef_val
        rows.append(
            {
                "Term / N-gram": feature_names[col_idx],
                "TF-IDF Weight": round(tfidf_val, 4),
                "Model Weight": round(coef_val, 4),
                "Contribution": round(contribution, 4),
            }
        )

    feat_df = pd.DataFrame(rows).sort_values("Contribution", ascending=False)
    return feat_df.head(top_n).reset_index(drop=True)


def split_clauses(text: str) -> list[str]:
    parts = [p.strip() for p in CLAUSE_SPLIT_PATTERN.split(text)]
    return [p for p in parts if p]


def analyze_aspects(pipeline, raw_text: str) -> list[dict]:
    clauses = split_clauses(raw_text)
    results = []

    for aspect, patterns in ASPECT_KEYWORDS.items():
        aspect_re = re.compile("|".join(patterns), flags=re.IGNORECASE)
        matched = [c for c in clauses if aspect_re.search(c)]
        if not matched:
            results.append(
                {
                    "Aspect": aspect,
                    "Sentiment": "Not mentioned",
                    "Confidence": None,
                    "Matched Clauses": "—",
                }
            )
            continue

        combined_cleaned = clean_review_text(" ".join(matched))
        if not combined_cleaned:
            results.append(
                {
                    "Aspect": aspect,
                    "Sentiment": "Not mentioned",
                    "Confidence": None,
                    "Matched Clauses": "—",
                }
            )
            continue

        pred, score_df, score_kind = get_prediction_scores(pipeline, combined_cleaned)
        confidence = None
        if score_df is not None and score_kind == "Probability":
            top_row = score_df[score_df["Sentiment"] == pred]
            if not top_row.empty:
                confidence = f"{float(top_row.iloc[0]['Score']) * 100:.1f}%"

        results.append(
            {
                "Aspect": aspect,
                "Sentiment": pred,
                "Confidence": confidence or "—",
                "Matched Clauses": " | ".join(matched),
            }
        )

    return results


# ==========================================
# UI RENDERING
# ==========================================
def set_sample_review(sample_text: str) -> None:
    st.session_state["review_input"] = sample_text


def main():
    st.set_page_config(
        page_title="Smartphone Sentiment Analysis",
        page_icon="📱",
        layout="wide",
    )
    inject_responsive_css()

    st.title("📱 Smartphone Sentiment Analysis")
    st.caption(
        "Capstone dashboard powered by the TF-IDF + scikit-learn "
        "sentiment pipeline with negation-preserving preprocessing."
    )

    pipeline, labels = load_model_artifacts()
    reviews_df = load_reviews_data()
    metrics_df = load_metrics_data()
    errors_df = load_misclassifications_data()

    if pipeline is None:
        st.error(
            f"Trained model not found at `{MODEL_PATH}`. Run `python train_day1.py` first."
        )
        st.stop()

    tab_live, tab_data, tab_eval = st.tabs(
        ["Live Review & Aspects", "Dataset Overview", "Model Evaluation"]
    )

    # ------------------------------------------
    # TAB 1: LIVE REVIEW & ASPECTS
    # ------------------------------------------
    with tab_live:
        st.subheader("Analyze a Smartphone Review")

        if "review_input" not in st.session_state:
            st.session_state["review_input"] = SAMPLE_REVIEWS["Mixed (Aspect Contrast)"]

        sample_items = list(SAMPLE_REVIEWS.items())
        for row_start in range(0, len(sample_items), 2):
            btn_cols = st.columns(2)
            for col, (btn_label, sample_txt) in zip(
                btn_cols, sample_items[row_start : row_start + 2]
            ):
                col.button(
                    btn_label,
                    width="stretch",
                    on_click=set_sample_review,
                    args=(sample_txt,),
                )

        review_text = st.text_area(
            "Enter or edit a smartphone customer review:",
            key="review_input",
            height=130,
            placeholder="Type a smartphone review mentioning battery, camera, display, or gaming...",
        )

        cleaned = clean_review_text(review_text)
        if not cleaned:
            st.info("Enter a review above to view sentiment predictions and aspect analysis.")
        else:
            pred, score_df, score_kind = get_prediction_scores(pipeline, cleaned)

            top_subtitle = ""
            if score_df is not None and score_kind == "Probability":
                match_row = score_df[score_df["Sentiment"] == pred]
                if not match_row.empty:
                    top_subtitle = f"Confidence: {float(match_row.iloc[0]['Score']) * 100:.1f}%"

            st.markdown(render_sentiment_badge(pred, top_subtitle), unsafe_allow_html=True)

            with st.expander("Preprocessed text (negation-preserving)", expanded=False):
                st.code(cleaned, language="text")

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown(f"#### Class {score_kind or 'Scores'}")
                if score_df is not None:
                    display_scores = score_df.copy()
                    if score_kind == "Probability":
                        display_scores["Probability (%)"] = (
                            display_scores["Score"] * 100
                        ).round(2)
                        st.dataframe(
                            display_scores[["Sentiment", "Probability (%)"]],
                            width="stretch",
                            hide_index=True,
                        )
                        st.bar_chart(
                            display_scores.set_index("Sentiment")["Probability (%)"]
                        )
                    else:
                        display_scores["Decision Score"] = display_scores["Score"].round(4)
                        st.dataframe(
                            display_scores[["Sentiment", "Decision Score"]],
                            width="stretch",
                            hide_index=True,
                        )
                else:
                    st.info("Score breakdown is unavailable for this classifier.")

            with col_right:
                st.markdown("#### Top Influential Words / N-grams")
                feat_df = get_top_influential_features(pipeline, cleaned, pred, top_n=10)
                if feat_df is not None and not feat_df.empty:
                    st.dataframe(feat_df, width="stretch", hide_index=True)
                elif not isinstance(
                    getattr(pipeline, "named_steps", {}).get("classifier"),
                    LogisticRegression,
                ):
                    st.info(
                        "Token contribution breakdown is enabled for Logistic Regression "
                        "pipelines."
                    )
                else:
                    st.info("No vocabulary terms from this review matched the TF-IDF features.")

            st.divider()
            st.markdown("#### Aspect-Level Sentiment Breakdown")
            st.caption(
                "Splits the review on punctuation and conjunctions (`but`, `however`, "
                "`although`, `while`, `and`) and scores clauses matching each aspect."
            )

            aspect_rows = analyze_aspects(pipeline, review_text)
            cards_html = []
            for item in aspect_rows:
                sent = item["Sentiment"]
                fg, bg = SENTIMENT_COLORS.get(sent, ("#4b5563", "#f3f4f6"))
                conf_line = (
                    f"<div class='aspect-card-conf'>Confidence: {item['Confidence']}</div>"
                    if item["Confidence"] not in (None, "—")
                    else ""
                )
                cards_html.append(
                    f"<div class='aspect-card' style='background:{bg};"
                    f"border:1px solid {fg}33;color:{fg};'>"
                    f"<div class='aspect-card-title'>{item['Aspect']}</div>"
                    f"<div class='aspect-card-sentiment'>{sent}</div>"
                    f"{conf_line}</div>"
                )
            st.markdown(
                f"<div class='aspect-grid'>{''.join(cards_html)}</div>",
                unsafe_allow_html=True,
            )

            st.dataframe(
                pd.DataFrame(aspect_rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Matched Clauses": st.column_config.TextColumn(
                        "Matched Clauses", width="large"
                    )
                },
            )

    # ------------------------------------------
    # TAB 2: DATASET OVERVIEW
    # ------------------------------------------
    with tab_data:
        st.subheader("Dataset Overview")
        if reviews_df is None:
            st.warning(f"Dataset not found at `{DATA_PATH}`.")
        else:
            total_reviews = len(reviews_df)
            avg_rating = (
                pd.to_numeric(reviews_df["rating"], errors="coerce").mean()
                if "rating" in reviews_df.columns
                else None
            )
            neg_share = (
                (reviews_df["sentiment"] == "Negative").mean() * 100
                if "sentiment" in reviews_df.columns
                else 0.0
            )
            pos_share = (
                (reviews_df["sentiment"] == "Positive").mean() * 100
                if "sentiment" in reviews_df.columns
                else 0.0
            )
            unique_models = (
                reviews_df["model"].nunique() if "model" in reviews_df.columns else None
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Reviews", f"{total_reviews:,}")
            m2.metric("Average Rating", f"{avg_rating:.2f} ★" if avg_rating is not None else "N/A")
            m3.metric("Positive Share", f"{pos_share:.1f}%")
            m4.metric(
                "Unique Models" if unique_models is not None else "Negative Share",
                f"{unique_models:,}" if unique_models is not None else f"{neg_share:.1f}%",
            )

            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("#### Sentiment Distribution")
                order = labels if labels else ["Positive", "Neutral", "Negative"]
                sent_counts = (
                    reviews_df["sentiment"]
                    .value_counts()
                    .reindex(order, fill_value=0)
                    .rename_axis("Sentiment")
                    .reset_index(name="Reviews")
                )
                st.bar_chart(sent_counts.set_index("Sentiment")["Reviews"])
                st.dataframe(sent_counts, width="stretch", hide_index=True)

            with c_right:
                st.markdown("#### Aspect Mentions & Negative-Sentiment Rates")
                aspect_df = compute_aspect_overview(reviews_df)
                st.dataframe(aspect_df, width="stretch", hide_index=True)

                fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
                bars = ax1.bar(
                    aspect_df["Aspect"],
                    aspect_df["Mentions"],
                    color="#3b82f6",
                    alpha=0.85,
                    label="Mentions",
                )
                ax1.set_ylabel("Mention Count", color="#1d4ed8")
                ax1.tick_params(axis="y", labelcolor="#1d4ed8")
                ax1.tick_params(axis="x", rotation=12)

                ax2 = ax1.twinx()
                ax2.plot(
                    aspect_df["Aspect"],
                    aspect_df["Negative Rate (%)"],
                    color="#dc2626",
                    marker="o",
                    linewidth=2,
                    label="Negative Rate (%)",
                )
                ax2.set_ylabel("Negative Rate (%)", color="#dc2626")
                ax2.tick_params(axis="y", labelcolor="#dc2626")
                ax2.set_ylim(0, max(50, float(aspect_df["Negative Rate (%)"].max()) * 1.25))

                for bar in bars:
                    h = bar.get_height()
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        h,
                        f"{int(h)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                plt.title("Aspect Mentions vs. Negative Rate (%)")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ------------------------------------------
    # TAB 3: MODEL EVALUATION
    # ------------------------------------------
    with tab_eval:
        st.subheader("Benchmark Evaluation")

        if metrics_df is not None and not metrics_df.empty:
            best_row = metrics_df.iloc[0]
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Best Model", str(best_row["Model"]))
            e2.metric("Accuracy", f"{float(best_row['Accuracy']):.4f}")
            e3.metric("Weighted F1", f"{float(best_row['Weighted_F1']):.4f}")
            e4.metric("Macro F1", f"{float(best_row['Macro_F1']):.4f}")

            st.markdown("#### Model Comparison Leaderboard (`outputs/day1_model_metrics.csv`)")
            st.dataframe(metrics_df, width="stretch", hide_index=True)
        else:
            st.warning(f"Metrics file not found at `{METRICS_PATH}`.")

        col_cm, col_err = st.columns([1, 1.3])
        with col_cm:
            st.markdown("#### Confusion Matrix (`outputs/confusion_matrix.png`)")
            if CM_PATH.exists():
                st.image(str(CM_PATH), width="stretch")
            else:
                st.warning(f"Confusion matrix image not found at `{CM_PATH}`.")

        with col_err:
            st.markdown("#### Sample Misclassifications (`outputs/sample_misclassifications.csv`)")
            if errors_df is not None and not errors_df.empty:
                st.dataframe(
                    errors_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "review_text": st.column_config.TextColumn(
                            "Review Text", width="large"
                        )
                    },
                )
            elif errors_df is not None:
                st.info("No misclassifications recorded in the sample file.")
            else:
                st.warning(f"Misclassifications file not found at `{ERRORS_PATH}`.")


if __name__ == "__main__":
    main()
