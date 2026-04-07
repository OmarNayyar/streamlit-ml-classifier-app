"""Professional Streamlit upgrade of the original Coursera mushroom classifier app.

The original guided-project implementation is preserved in `base_app.py`.
This version keeps the same educational workflow while improving the UI,
state handling, plotting, and overall code organization for portfolio use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


DATA_PATH = Path(__file__).with_name("mushrooms.csv")
RANDOM_STATE = 42
TARGET_COLUMN = "type"
CLASS_LABELS = {0: "Edible", 1: "Poisonous"}

PLOT_HELP = {
    "Confusion Matrix": "Shows how many predictions were correct or confused across the two classes.",
    "ROC Curve": "Shows how well the model separates edible and poisonous mushrooms across thresholds.",
    "Precision-Recall Curve": "Useful for checking prediction quality when you care about confident positive predictions.",
}

METRIC_HELP = {
    "Accuracy": "Overall share of correct predictions.",
    "Precision": "When the model predicts poisonous, how often that prediction is correct.",
    "Recall": "How often the model successfully catches poisonous mushrooms.",
}

CLASSIFIER_HELP = {
    "Support Vector Machine (SVM)": "A strong classifier that tries to draw the best boundary between edible and poisonous samples.",
    "Logistic Regression": "A fast linear model that estimates class probabilities and is easy to interpret.",
    "Random Forest": "An ensemble of decision trees that usually handles tabular categorical data very well.",
}


@dataclass(frozen=True)
class RunResult:
    classifier_name: str
    model: object
    accuracy: float
    precision: float
    recall: float
    selected_plots: list[str]
    predictions: pd.Series


def inject_styles() -> None:
    """Apply a restrained dark theme polish using lightweight custom CSS."""
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.10), transparent 28%),
                    radial-gradient(circle at top right, rgba(148, 163, 184, 0.10), transparent 26%),
                    linear-gradient(180deg, #0b1120 0%, #111827 50%, #0f172a 100%);
                color: #e5e7eb;
            }
            [data-testid="stSidebar"] {
                background: rgba(15, 23, 42, 0.92);
                border-right: 1px solid rgba(148, 163, 184, 0.14);
            }
            .hero-card,
            .section-card {
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 18px;
                padding: 1.2rem 1.2rem 1rem 1.2rem;
                box-shadow: 0 14px 32px rgba(2, 6, 23, 0.22);
                backdrop-filter: blur(8px);
            }
            .hero-title {
                font-size: 2.25rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin-bottom: 0.35rem;
            }
            .hero-subtitle {
                color: #cbd5e1;
                font-size: 1rem;
                line-height: 1.65;
                margin-bottom: 0;
            }
            .section-label {
                color: #93c5fd;
                font-size: 0.82rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }
            .metric-caption {
                color: #94a3b8;
                font-size: 0.88rem;
                margin-top: -0.55rem;
                margin-bottom: 0;
            }
            .stMetric {
                background: rgba(15, 23, 42, 0.65);
                border: 1px solid rgba(148, 163, 184, 0.15);
                padding: 0.85rem;
                border-radius: 16px;
            }
            div[data-testid="stExpander"] {
                background: rgba(15, 23, 42, 0.55);
                border: 1px solid rgba(148, 163, 184, 0.12);
                border-radius: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the original categorical mushroom dataset."""
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def encode_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Encode each categorical column to preserve the original model workflow."""
    encoded_df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for column in encoded_df.columns:
        encoder = LabelEncoder()
        encoded_df[column] = encoder.fit_transform(encoded_df[column])
        encoders[column] = encoder

    return encoded_df, encoders


@st.cache_data(show_spinner=False)
def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stable train/test split for consistent comparison across models."""
    features = df.drop(columns=TARGET_COLUMN)
    target = df[TARGET_COLUMN]

    return train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">Portfolio Upgrade</div>
            <div class="hero-title">Mushroom Classifier Studio</div>
            <p class="hero-subtitle">
                A polished Streamlit version of the original guided project that compares
                three classification models on the UCI mushroom dataset. The learning flow
                stays intact, while the interface, structure, and presentation are upgraded
                for a portfolio-ready finish.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, dict[str, object], list[str]]:
    st.sidebar.title("Model Controls")
    st.sidebar.caption("Choose a classifier, tune a few parameters, and run an evaluation.")

    classifier_name = st.sidebar.selectbox(
        "Classifier",
        options=list(CLASSIFIER_HELP.keys()),
        help="Each option uses the same encoded mushroom dataset but learns patterns in a different way.",
    )

    st.sidebar.info(CLASSIFIER_HELP[classifier_name])
    st.sidebar.markdown("### Hyperparameters")

    params: dict[str, object] = {}

    if classifier_name == "Support Vector Machine (SVM)":
        params["C"] = st.sidebar.slider(
            "C",
            min_value=0.10,
            max_value=5.00,
            value=1.00,
            step=0.10,
            help="Controls how strictly the model tries to avoid training mistakes. Higher values fit the training data more aggressively.",
        )
        params["kernel"] = st.sidebar.selectbox(
            "Kernel",
            options=["rbf", "linear"],
            help="Changes the shape of the decision boundary. `rbf` is more flexible, while `linear` is simpler.",
        )
        params["gamma"] = st.sidebar.selectbox(
            "Gamma",
            options=["scale", "auto"],
            help="Controls how far the influence of each training point reaches when using an RBF kernel.",
        )

    elif classifier_name == "Logistic Regression":
        params["C"] = st.sidebar.slider(
            "C",
            min_value=0.10,
            max_value=5.00,
            value=1.00,
            step=0.10,
            help="Controls regularization strength. Lower values keep the model simpler, higher values allow a tighter fit.",
        )
        params["max_iter"] = st.sidebar.slider(
            "Max iterations",
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help="Sets how many optimization steps the solver can take before stopping.",
        )

    else:
        params["n_estimators"] = st.sidebar.slider(
            "Number of trees",
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help="More trees can improve stability, but they also increase training time.",
        )
        params["max_depth"] = st.sidebar.slider(
            "Max depth",
            min_value=2,
            max_value=20,
            value=8,
            step=1,
            help="Limits how deep each tree can grow. Smaller values reduce overfitting.",
        )
        params["bootstrap"] = st.sidebar.checkbox(
            "Bootstrap samples",
            value=True,
            help="When enabled, each tree trains on a random sample of the data instead of the full dataset.",
        )

    selected_plots = st.sidebar.multiselect(
        "Evaluation plots",
        options=list(PLOT_HELP.keys()),
        default=["Confusion Matrix", "ROC Curve"],
        help="Choose which diagnostic plots to include in the results section.",
    )

    st.sidebar.markdown("### Quick Help")
    with st.sidebar.expander("Metric glossary", expanded=False):
        for metric_name, description in METRIC_HELP.items():
            st.markdown(f"**{metric_name}**: {description}")
    with st.sidebar.expander("Plot glossary", expanded=False):
        for plot_name, description in PLOT_HELP.items():
            st.markdown(f"**{plot_name}**: {description}")

    if st.sidebar.button("Train and evaluate", type="primary", use_container_width=True):
        st.session_state["run_requested"] = True

    return classifier_name, params, selected_plots


def build_model(classifier_name: str, params: dict[str, object]):
    """Instantiate the selected model using the current sidebar settings."""
    if classifier_name == "Support Vector Machine (SVM)":
        return SVC(
            C=float(params["C"]),
            kernel=str(params["kernel"]),
            gamma=str(params["gamma"]),
            probability=True,
            random_state=RANDOM_STATE,
        )

    if classifier_name == "Logistic Regression":
        return LogisticRegression(
            C=float(params["C"]),
            penalty="l2",
            max_iter=int(params["max_iter"]),
            random_state=RANDOM_STATE,
        )

    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        bootstrap=bool(params["bootstrap"]),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_and_evaluate(
    classifier_name: str,
    params: dict[str, object],
    selected_plots: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> RunResult:
    """Fit the chosen model and calculate the core evaluation metrics."""
    model = build_model(classifier_name, params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return RunResult(
        classifier_name=classifier_name,
        model=model,
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, pos_label=1),
        recall=recall_score(y_test, predictions, pos_label=1),
        selected_plots=selected_plots,
        predictions=pd.Series(predictions, name="prediction"),
    )


def render_summary_cards(raw_df: pd.DataFrame, encoded_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Dataset Snapshot</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(raw_df):,}")
    col1.caption("Mushroom records")
    col2.metric("Features", encoded_df.shape[1] - 1)
    col2.caption("Predictor columns")
    col3.metric("Classes", raw_df[TARGET_COLUMN].nunique())
    col3.caption("Edible vs poisonous")
    poisonous_share = (raw_df[TARGET_COLUMN] == "p").mean()
    col4.metric("Poisonous share", f"{poisonous_share:.1%}")
    col4.caption("Target balance")


def render_model_summary(classifier_name: str, params: dict[str, object]) -> None:
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Model summary")
        st.write(
            "This section mirrors the original guided-project goal: select a classifier, adjust a few key hyperparameters, then compare performance on the same test split."
        )
        st.markdown(f"**Selected classifier:** {classifier_name}")
        st.json(params, expanded=False)
        st.markdown("</div>", unsafe_allow_html=True)


def render_metric_cards(result: RunResult) -> None:
    st.markdown('<div class="section-label">Evaluation Metrics</div>', unsafe_allow_html=True)
    metric_cols = st.columns(3)

    for column, (name, value) in zip(
        metric_cols,
        [
            ("Accuracy", result.accuracy),
            ("Precision", result.precision),
            ("Recall", result.recall),
        ],
    ):
        column.metric(name, f"{value:.3f}")
        column.caption(METRIC_HELP[name])


def draw_plot(title: str, help_text: str, plot_callable) -> None:
    st.markdown(f"#### {title}")
    st.caption(help_text)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_callable(ax)
    fig.patch.set_alpha(0)
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#e5e7eb")
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    ax.title.set_color("#f8fafc")

    for spine in ax.spines.values():
        spine.set_color("#94a3b8")

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("#0f172a")
        legend.get_frame().set_edgecolor("#475569")
        for text in legend.get_texts():
            text.set_color("#e5e7eb")

    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def render_plots(result: RunResult, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    if not result.selected_plots:
        st.info("Choose one or more plots from the sidebar to see visual diagnostics here.")
        return

    st.markdown('<div class="section-label">Visual Diagnostics</div>', unsafe_allow_html=True)

    for plot_name in result.selected_plots:
        if plot_name == "Confusion Matrix":
            draw_plot(
                title="Confusion Matrix",
                help_text=PLOT_HELP[plot_name],
                plot_callable=lambda ax: ConfusionMatrixDisplay.from_estimator(
                    result.model,
                    x_test,
                    y_test,
                    display_labels=[CLASS_LABELS[0], CLASS_LABELS[1]],
                    cmap="Blues",
                    colorbar=False,
                    ax=ax,
                ),
            )
        elif plot_name == "ROC Curve":
            draw_plot(
                title="ROC Curve",
                help_text=PLOT_HELP[plot_name],
                plot_callable=lambda ax: RocCurveDisplay.from_estimator(
                    result.model,
                    x_test,
                    y_test,
                    ax=ax,
                ),
            )
        else:
            draw_plot(
                title="Precision-Recall Curve",
                help_text=PLOT_HELP[plot_name],
                plot_callable=lambda ax: PrecisionRecallDisplay.from_estimator(
                    result.model,
                    x_test,
                    y_test,
                    ax=ax,
                ),
            )


def render_dataset_views(raw_df: pd.DataFrame) -> None:
    overview_tab, preview_tab, features_tab = st.tabs(
        ["About", "Dataset Preview", "Feature Overview"]
    )

    with overview_tab:
        st.markdown("#### About this app")
        st.write(
            "This project started as a Coursera guided exercise and was then refined into a stronger portfolio piece. "
            "The upgraded version keeps the original classification workflow while improving the layout, readability, and UX."
        )
        with st.expander("How to use", expanded=False):
            st.markdown(
                """
                1. Pick a classifier from the sidebar.
                2. Adjust the model hyperparameters.
                3. Choose the evaluation plots you want to inspect.
                4. Click **Train and evaluate** to compare results.
                """
            )

    with preview_tab:
        st.markdown("#### Dataset preview")
        st.caption(
            "The raw dataset is shown below so viewers can see that the source data is categorical before encoding."
        )
        st.dataframe(raw_df.head(12), use_container_width=True)

    with features_tab:
        st.markdown("#### Feature overview")
        feature_summary = pd.DataFrame(
            {
                "feature": raw_df.drop(columns=TARGET_COLUMN).columns,
                "unique_values": [raw_df[col].nunique() for col in raw_df.drop(columns=TARGET_COLUMN).columns],
            }
        ).sort_values("unique_values", ascending=False)
        st.dataframe(feature_summary, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Mushroom Classifier Studio",
        page_icon=":mushroom:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    render_header()

    raw_df = load_raw_data()
    encoded_df, _ = encode_dataset(raw_df)
    x_train, x_test, y_train, y_test = split_dataset(encoded_df)

    classifier_name, params, selected_plots = render_sidebar()
    render_summary_cards(raw_df, encoded_df)

    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        render_model_summary(classifier_name, params)

        if st.session_state.get("run_requested"):
            with st.spinner("Training the model and preparing evaluation visuals..."):
                st.session_state["last_result"] = train_and_evaluate(
                    classifier_name=classifier_name,
                    params=params,
                    selected_plots=selected_plots,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                )
            st.session_state["run_requested"] = False

        result: RunResult | None = st.session_state.get("last_result")
        if result is None:
            st.info("Use the sidebar to configure a model, then click **Train and evaluate**.")
        else:
            st.markdown(f"### Results: {result.classifier_name}")
            render_metric_cards(result)
            render_plots(result, x_test, y_test)

    with right_col:
        render_dataset_views(raw_df)


if __name__ == "__main__":
    main()
