import os
import re
import pickle
import warnings
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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# --- Sklearn ---
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# 0. NLTK Downloads
# =============================================================================

def download_nltk_resources():
    """Download all required NLTK resources."""
    resources = [
        "stopwords", "punkt", "wordnet",
        "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "omw-1.4",
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

# =============================================================================
# 4. PREPROCESSING FUNCTIONS
# =============================================================================

_stop_words  = set(stopwords.words("english"))
_lemmatizer  = WordNetLemmatizer()
_tweet_tok_lower = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
_tweet_tok_keepcase = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)

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


def normalize_for_transformer(text: str) -> str:
    """
    Lightweight normalization for transformer models.
    Replaces noisy entities with common placeholder tokens and keeps most text intact.
    """
    text = re.sub(r"http\S+|www\.\S+", " [URL] ", text)    # URLs
    text = re.sub(r"@\w+", " [USER] ", text)                 # mentions
    text = re.sub(r"\$[A-Za-z]+", " [CASHTAG] ", text)      # cashtags
    text = re.sub(r"#(\w+)", r" [HASHTAG] \1 ", text)       # hashtag marker + token
    text = re.sub(r"\bRT\b", " [RT] ", text)                # retweet marker
    text = re.sub(r"&[a-z]+;", " ", text)                    # basic HTML entities
    text = re.sub(r"\s+", " ", text).strip()                # whitespace
    return text


# --- 4.2  Tokenization ---

def tokenize(text: str, mode: str = "tweet", preserve_case: bool = False) -> list[str]:
    """
    Tokenize text.
    mode='tweet' uses NLTK TweetTokenizer (recommended for tweets).
    mode='word'  uses NLTK word_tokenize.
    preserve_case=False lowercases during tokenization.
    """
    if mode == "tweet":
        tokenizer = _tweet_tok_keepcase if preserve_case else _tweet_tok_lower
        return tokenizer.tokenize(text)
    return word_tokenize(text if preserve_case else text.lower())


# --- 4.3  Stop Word Removal ---

def remove_stopwords(tokens: list[str], extra_keep: set | None = None) -> list[str]:
    """
    Remove NLTK English stop words from a token list.
    extra_keep: set of words to keep even if they are stop words (e.g. {'not', 'no'}).
    """
    keep = extra_keep or set()
    return [t for t in tokens if t not in _stop_words or t in keep]


# --- 4.4  Lemmatization ---

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


# =============================================================================
# 5. FULL PREPROCESSING PIPELINE
# =============================================================================

def preprocess_pipeline(
    series: pd.Series,
    mode: str = "classical",
    keep_stopwords: set | None = None,
) -> pd.Series:
    """
    Apply project preprocessing to a pandas Series of texts.

    mode='classical' uses NLTK-style preprocessing for classic ML pipelines.
    mode='transformer' uses lightweight normalization with placeholder tokens.
    """

    if mode not in {"classical", "transformer"}:
        raise ValueError("Invalid mode. Expected 'classical' or 'transformer'.")

    def _preprocess_text(text: str) -> str:
        current = str(text)

        if mode == "transformer":
            return normalize_for_transformer(current)

        current = remove_noise(current)
        tokens = tokenize(current, mode="tweet", preserve_case=False)
        tokens = remove_stopwords(tokens, extra_keep=keep_stopwords)
        tokens = lemmatize(tokens)
        return " ".join(tokens)

    return series.fillna("").apply(_preprocess_text)

# =============================================================================
# 6. EDA PLOTTING HELPERS
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

    _, axes = plt.subplots(1, 2, figsize=(13, 4))

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
    **pipeline_kwargs,
) -> pd.DataFrame:
    """
    Show side-by-side original vs. preprocessed tweet examples.
    pipeline_fn: a callable that takes a pandas Series and returns a pandas Series.
                 Defaults to preprocess_pipeline if None.
    pipeline_kwargs: optional keyword args passed to pipeline_fn.
    """
    if pipeline_fn is None:
        pipeline_fn = preprocess_pipeline

    sample = df[text_col].sample(n, random_state=RANDOM_STATE).reset_index(drop=True)
    processed = pipeline_fn(sample, **pipeline_kwargs)

    result = pd.DataFrame({
        "original":     sample,
        "preprocessed": processed,
    })
    return result