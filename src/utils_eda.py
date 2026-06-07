import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# --- NLTK ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.corpus import wordnet
from nltk import pos_tag

# --- Sklearn ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# =============================================================================
# 0. NLTK Downloads
# =============================================================================

def download_nltk_resources():
    """Download all required NLTK resources."""
    resources = [
        "stopwords", "punkt", "wordnet",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "omw-1.4",
    ]
    for r in resources:
        nltk.download(r, quiet=True)

download_nltk_resources()

# =============================================================================
# 1. CONSTANTS
# =============================================================================

LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_COLORS = {0: "#E05C5C", 1: "#5CB85C", 2: "#5B9BD5"}
RANDOM_STATE = 67

# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs. Returns (train_df, test_df)."""
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    print(f"Train shape : {train_df.shape}")
    print(f"Test  shape : {test_df.shape}")
    return train_df, test_df


def basic_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print basic info: shape, dtypes, nulls, class distribution."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(df.dtypes)
    print(f"\nNull values:\n{df.isnull().sum()}")
    if "label" in df.columns:
        counts = df["label"].value_counts().sort_index()
        print(f"\nClass distribution:")
        for k, v in counts.items():
            print(f"  {k} ({LABEL_MAP[k]}): {v}  ({v/len(df)*100:.1f}%)")

# =============================================================================
# 3. CORPUS SPLIT
# =============================================================================

def make_split(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.2,
    save_path: str | None = None,
) -> dict:
    """
    Stratified train/validation split.
    Returns dict with keys: X_train, X_val, y_train, y_val.
    Optionally saves to pickle at save_path.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        df[text_col],
        df[label_col],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[label_col],
    )
    split = {
        "X_train": X_train.reset_index(drop=True),
        "X_val":   X_val.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_val":   y_val.reset_index(drop=True),
    }
    print(f"Train size      : {len(X_train)} samples")
    print(f"Validation size : {len(X_val)} samples")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(split, f)
        print(f"Split saved to  : {save_path}")
    return split


def load_split(path: str) -> dict:
    """Load a previously saved train/val split from pickle."""
    with open(path, "rb") as f:
        split = pickle.load(f)
    print(f"Split loaded from: {path}")
    return split

# =============================================================================
# 4. PREPROCESSING FUNCTIONS
# =============================================================================

_stop_words  = set(stopwords.words("english"))
_lemmatizer  = WordNetLemmatizer()
_porter      = PorterStemmer()
_snowball    = SnowballStemmer("english")
_tweet_tok   = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# --- 4.1  Regex / Noise Removal ---

def remove_noise(text: str) -> str:
    """
    Remove URLs, mentions, RT markers, cashtags, HTML entities,
    non-ASCII characters, and excessive whitespace.
    """
    text = re.sub(r"http\S+|www\.\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                       # mentions
    text = re.sub(r"\bRT\b", "", text)                     # retweet marker
    text = re.sub(r"\$[A-Za-z]+", "", text)                # cashtags  e.g. $AAPL
    text = re.sub(r"&[a-z]+;", "", text)                   # HTML entities
    text = re.sub(r"[^\x00-\x7F]+", " ", text)             # non-ASCII
    text = re.sub(r"[^a-zA-Z\s#]", " ", text)             # keep letters + hashtag symbol
    text = re.sub(r"#(\w+)", r"\1", text)                  # strip # but keep word
    text = re.sub(r"\s+", " ", text).strip()               # whitespace
    return text


# --- 4.2  Lowercasing ---

def to_lowercase(text: str) -> str:
    return text.lower()


# --- 4.3  Tokenization ---

def tokenize(text: str, mode: str = "tweet") -> list[str]:
    """
    Tokenize text.
    mode='tweet' uses NLTK TweetTokenizer (recommended for tweets).
    mode='word'  uses NLTK word_tokenize.
    """
    if mode == "tweet":
        return _tweet_tok.tokenize(text)
    return word_tokenize(text)


# --- 4.4  Stop Word Removal ---

def remove_stopwords(tokens: list[str], extra_keep: set | None = None) -> list[str]:
    """
    Remove NLTK English stop words from a token list.
    extra_keep: set of words to keep even if they are stop words (e.g. {'not', 'no'}).
    """
    keep = extra_keep or set()
    return [t for t in tokens if t not in _stop_words or t in keep]


# --- 4.5  Lemmatization ---

def _get_wordnet_pos(treebank_tag: str) -> str:
    """Map POS treebank tag to WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize tokens using POS-aware WordNetLemmatizer."""
    tagged = pos_tag(tokens)
    return [_lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in tagged]


# --- 4.6  Stemming ---

def stem_porter(tokens: list[str]) -> list[str]:
    """Stem tokens using Porter Stemmer."""
    return [_porter.stem(t) for t in tokens]


def stem_snowball(tokens: list[str]) -> list[str]:
    """Stem tokens using Snowball Stemmer."""
    return [_snowball.stem(t) for t in tokens]


# =============================================================================
# 5. FULL PREPROCESSING PIPELINES
# =============================================================================

def preprocess_lemma(text: str, keep_stopwords: set | None = None) -> str:
    """
    Pipeline A — Lemmatization (recommended for traditional ML + Transformers input).
    Steps: noise removal → lowercase → tokenize → stopwords → lemmatize → rejoin
    """
    text   = remove_noise(text)
    text   = to_lowercase(text)
    tokens = tokenize(text, mode="tweet")
    tokens = remove_stopwords(tokens, extra_keep=keep_stopwords)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_stem(text: str, stemmer: str = "porter", keep_stopwords: set | None = None) -> str:
    """
    Pipeline B — Stemming.
    Steps: noise removal → lowercase → tokenize → stopwords → stem → rejoin
    stemmer: 'porter' | 'snowball'
    """
    text   = remove_noise(text)
    text   = to_lowercase(text)
    tokens = tokenize(text, mode="tweet")
    tokens = remove_stopwords(tokens, extra_keep=keep_stopwords)
    tokens = stem_porter(tokens) if stemmer == "porter" else stem_snowball(tokens)
    return " ".join(tokens)


def preprocess_raw_clean(text: str) -> str:
    """
    Pipeline C — Light cleaning only (for Transformer models that do their own tokenization).
    Steps: noise removal → lowercase
    """
    text = remove_noise(text)
    text = to_lowercase(text)
    return text


def apply_preprocessing(series: pd.Series, pipeline: str = "lemma", **kwargs) -> pd.Series:
    """
    Apply a named preprocessing pipeline to a pandas Series of texts.
    pipeline: 'lemma' | 'stem_porter' | 'stem_snowball' | 'raw_clean'
    """
    pipeline_map = {
        "lemma":         preprocess_lemma,
        "stem_porter":   lambda t: preprocess_stem(t, stemmer="porter",   **kwargs),
        "stem_snowball": lambda t: preprocess_stem(t, stemmer="snowball",  **kwargs),
        "raw_clean":     preprocess_raw_clean,
    }
    if pipeline not in pipeline_map:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Choose from: {list(pipeline_map.keys())}")
    fn = pipeline_map[pipeline]
    return series.apply(fn)

# =============================================================================
# 6. EVALUATION
# =============================================================================

def evaluate_model(
    y_true,
    y_pred,
    model_name: str = "Model",
    print_report: bool = True,
) -> dict:
    """
    Compute Accuracy, Precision, Recall and F1 (all macro-averaged).
    Returns a dict suitable for building a metrics DataFrame.
    """
    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "f1_macro":  round(f1_score(y_true, y_pred, average="macro",        zero_division=0), 4),
    }
    if print_report:
        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")
        print(classification_report(
            y_true, y_pred,
            target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
            zero_division=0,
        ))
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model") -> None:
    """Plot a labeled confusion matrix."""
    labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    cm     = confusion_matrix(y_true, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()


def build_metrics_df(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Turn a list of evaluate_model() dicts into a sorted comparison DataFrame.
    """
    df = pd.DataFrame(metrics_list).sort_values("f1_macro", ascending=False)
    return df.reset_index(drop=True)

# =============================================================================
# 7. EDA PLOTTING HELPERS
# =============================================================================

def plot_class_distribution(df: pd.DataFrame, label_col: str = "label") -> None:
    """Bar chart of class distribution with percentages."""
    counts = df[label_col].value_counts().sort_index()
    labels = [LABEL_MAP[i] for i in counts.index]
    colors = [LABEL_COLORS[i] for i in counts.index]
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val}\n({val/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Class Distribution", fontsize=14, pad=10)
    ax.set_ylabel("Number of Tweets")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_tweet_length_distribution(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> None:
    """Overlapping KDE of tweet character length per class."""
    df = df.copy()
    df["_len"] = df[text_col].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Character length
    for label, group in df.groupby(label_col):
        group["_len"].plot.kde(ax=axes[0], label=LABEL_MAP[label], color=LABEL_COLORS[label], linewidth=2)
    axes[0].set_title("Tweet Character Length by Class")
    axes[0].set_xlabel("Characters")
    axes[0].legend()

    # Word count
    df["_words"] = df[text_col].apply(lambda t: len(str(t).split()))
    for label, group in df.groupby(label_col):
        group["_words"].plot.kde(ax=axes[1], label=LABEL_MAP[label], color=LABEL_COLORS[label], linewidth=2)
    axes[1].set_title("Tweet Word Count by Class")
    axes[1].set_xlabel("Word Count")
    axes[1].legend()

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_wordclouds(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> None:
    """One word cloud per class."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (label, group) in zip(axes, df.groupby(label_col)):
        corpus = " ".join(group[text_col].astype(str).tolist())
        wc = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap="RdYlGn" if label != 2 else "Blues",
            max_words=150,
        ).generate(corpus)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{LABEL_MAP[label]} tweets", fontsize=13)
        ax.axis("off")
    plt.suptitle("Word Clouds by Sentiment Class", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_top_tokens(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    n: int = 20,
) -> None:
    """Horizontal bar charts of top-N tokens per class (after basic cleaning)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, group) in zip(axes, df.groupby(label_col)):
        tokens = " ".join(group[text_col].astype(str)).split()
        top    = Counter(tokens).most_common(n)
        words, freqs = zip(*top)
        ax.barh(list(words)[::-1], list(freqs)[::-1], color=LABEL_COLORS[label])
        ax.set_title(f"Top {n} tokens — {LABEL_MAP[label]}", fontsize=12)
        ax.set_xlabel("Frequency")
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()


def show_preprocessing_examples(
    df: pd.DataFrame,
    text_col: str = "text",
    n: int = 5,
    pipeline_fn=None,
) -> pd.DataFrame:
    """
    Show side-by-side original vs. preprocessed tweet examples.
    pipeline_fn: a callable that takes raw text and returns cleaned text.
                 Defaults to preprocess_lemma if None.
    """
    if pipeline_fn is None:
        pipeline_fn = preprocess_lemma
    sample = df[text_col].sample(n, random_state=RANDOM_STATE).reset_index(drop=True)
    result = pd.DataFrame({
        "original":     sample,
        "preprocessed": sample.apply(pipeline_fn),
    })
    return result