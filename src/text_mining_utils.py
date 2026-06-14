"""
Text Mining Utilities
Provides preprocessing, vectorization, and model evaluation functions.
"""
import os
import re
import pickle
import tempfile
import warnings
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from tqdm.auto import tqdm

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support, accuracy_score,
)
from sklearn.model_selection import cross_validate

# Optional dependencies
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        pipeline,
    )
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

Word2Vec = None
try:
    _gensim_models = importlib.import_module("gensim.models")
    Word2Vec = _gensim_models.Word2Vec
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS
# ============================================================================

LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_COLORS = {0: "#E05C5C", 1: "#5CB85C", 2: "#5B9BD5"}
RANDOM_STATE = 67
EMBEDDING_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "transformer_cache"
)

# ============================================================================
# NLTK SETUP
# ============================================================================

def download_nltk_resources():
    """Download required NLTK resources."""
    resources = [
        "stopwords", "punkt", "wordnet", "punkt_tab",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "omw-1.4",
    ]
    for resource in resources:
        nltk.download(resource, quiet=True)

download_nltk_resources()

# Precompile regex and NLTK objects
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()
_tweet_tok = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

def compute_cv_summary(estimator, X, y, cv, n_jobs: int = -1) -> dict:
    """Run cross-validation and return mean/std metrics."""
    scoring = {
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
    }
    cv_res = cross_validate(clone(estimator), X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)

    summary = {}
    for metric_name in scoring:
        key = f"test_{metric_name}"
        if key in cv_res:
            scores = cv_res[key]
            summary[f"cv_{metric_name}_mean"] = round(float(np.mean(scores)), 4)
            summary[f"cv_{metric_name}_std"] = round(float(np.std(scores)), 4)
    return summary

# ============================================================================
# PREPROCESSING
# ============================================================================

def _remove_noise(text: str) -> str:
    """Remove noise: URLs, mentions, cashtags, HTML entities, non-ASCII."""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\bRT\b", "", text)
    text = re.sub(r"\$[A-Za-z]+", "", text)
    text = re.sub(r"&[a-z]+;", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s#]", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_for_transformer(text: str) -> str:
    """Lightweight normalization with placeholder tokens."""
    text = re.sub(r"http\S+|www\.\S+", " [URL] ", text)
    text = re.sub(r"@\w+", " [USER] ", text)
    text = re.sub(r"\$[A-Za-z]+", " [CASHTAG] ", text)
    text = re.sub(r"#(\w+)", r" [HASHTAG] \1 ", text)
    text = re.sub(r"\bRT\b", " [RT] ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_wordnet_pos(treebank_tag: str) -> str:
    """Map POS treebank tag to WordNet POS."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def _lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize with POS awareness."""
    tagged = pos_tag(tokens)
    return [
        _lemmatizer.lemmatize(word, _get_wordnet_pos(tag))
        for word, tag in tagged
    ]


def preprocess_pipeline(series: pd.Series, mode: str = "classical") -> pd.Series:
    """
    Preprocess text series.
    
    Args:
        series: Text column
        mode: "classical" (for TF-IDF/Word2Vec) or "transformer" (for BERT)
    """
    if mode not in {"classical", "transformer"}:
        raise ValueError(f"Invalid mode: {mode}. Use 'classical' or 'transformer'.")

    def _preprocess(text: str) -> str:
        text = str(text)
        if mode == "transformer":
            return _normalize_for_transformer(text)

        text = _remove_noise(text)
        tokens = _tweet_tok.tokenize(text)
        tokens = [t for t in tokens if t not in _stop_words]
        tokens = _lemmatize(tokens)
        return " ".join(tokens)

    return series.fillna("").apply(_preprocess)

# ============================================================================
# VECTORIZATION
# ============================================================================

class Word2VecVectorizer:
    """Word2Vec vectorizer with sklearn-compatible interface."""

    def __init__(self, size: int = 100, window: int = 5, min_count: int = 2):
        if not HAS_GENSIM:
            raise ImportError("gensim required. Install: pip install gensim")
        self.size = size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, texts):
        """Train Word2Vec."""
        tokenized = [str(text).split() for text in texts]
        self.model = Word2Vec(
            tokenized,
            vector_size=self.size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
        )
        return self

    def transform(self, texts) -> np.ndarray:
        """Convert texts to mean word vectors."""
        vectors = []
        for text in texts:
            tokens = str(text).split()
            token_vecs = [
                self.model.wv[token] for token in tokens
                if token in self.model.wv
            ]
            mean_vec = np.mean(token_vecs, axis=0) if token_vecs else np.zeros(self.size)
            vectors.append(mean_vec)
        return np.array(vectors)

    def fit_transform(self, texts) -> np.ndarray:
        """Fit and transform."""
        return self.fit(texts).transform(texts)

# ============================================================================
# PLOTTING
# ============================================================================

def plot_class_distribution(df: pd.DataFrame, label_col: str = "label") -> None:
    """Bar chart of class distribution."""
    counts = df[label_col].value_counts().sort_index()
    labels = [LABEL_MAP[i] for i in counts.index]
    colors = [LABEL_COLORS[i] for i in counts.index]
    total = counts.sum()

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


def plot_tweet_length_distribution(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "label"
) -> None:
    """KDE plot of tweet length per class."""
    df = df.copy()
    df["_len"] = df[text_col].str.len()

    _, axes = plt.subplots(1, 2, figsize=(13, 4))

    for label, group in df.groupby(label_col):
        group["_len"].plot.kde(
            ax=axes[0], label=LABEL_MAP[label], color=LABEL_COLORS[label], linewidth=2
        )
    axes[0].set_title("Tweet Character Length by Class")
    axes[0].set_xlabel("Characters")
    axes[0].legend()

    df["_words"] = df[text_col].apply(lambda t: len(str(t).split()))
    for label, group in df.groupby(label_col):
        group["_words"].plot.kde(
            ax=axes[1], label=LABEL_MAP[label], color=LABEL_COLORS[label], linewidth=2
        )
    axes[1].set_title("Tweet Word Count by Class")
    axes[1].set_xlabel("Word Count")
    axes[1].legend()

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_wordclouds(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "label"
) -> None:
    """Word cloud per class."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (label, group) in zip(axes, df.groupby(label_col)):
        corpus = " ".join(group[text_col].astype(str).tolist())
        colormap = "RdYlGn" if label != 2 else "Blues"
        wc = WordCloud(
            width=600, height=400, background_color="white",
            colormap=colormap, max_words=150,
        ).generate(corpus)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{LABEL_MAP[label]} tweets", fontsize=13)
        ax.axis("off")
    plt.suptitle("Word Clouds by Sentiment Class", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_top_tokens(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "label", n: int = 20
) -> None:
    """Top N tokens per class."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, group) in zip(axes, df.groupby(label_col)):
        tokens = " ".join(group[text_col].astype(str)).split()
        top = Counter(tokens).most_common(n)
        words, freqs = zip(*top)
        ax.barh(list(words)[::-1], list(freqs)[::-1], color=LABEL_COLORS[label])
        ax.set_title(f"Top {n} tokens — {LABEL_MAP[label]}", fontsize=12)
        ax.set_xlabel("Frequency")
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()


def show_preprocessing_examples(
    df: pd.DataFrame, text_col: str = "text", n: int = 5, pipeline_fn=None, **kwargs
) -> pd.DataFrame:
    """Show before/after preprocessing examples."""
    if pipeline_fn is None:
        pipeline_fn = preprocess_pipeline

    sample = df[text_col].sample(n, random_state=RANDOM_STATE).reset_index(drop=True)
    processed = pipeline_fn(sample, **kwargs)

    return pd.DataFrame({"original": sample, "preprocessed": processed})

def plot_model_confusion_matrix(y_true, y_pred, model_name: str = "Model") -> None:
    """Plot confusion matrix."""
    labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    cm = confusion_matrix(y_true, y_pred, labels=sorted(LABEL_MAP))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

# ============================================================================
# MODEL TRAINING
# ============================================================================

def run_classical_text_experiment(
    experiment_name: str,
    vectorizer,
    classifier,
    X_train_text,
    y_train_labels,
    cv,
    verbose: bool = True,
) -> dict:
    """Train classical ML model with CV evaluation."""
    x_train_vec = vectorizer.fit_transform(X_train_text)
    estimator = clone(classifier)

    cv_summary = compute_cv_summary(
        estimator=estimator,
        X=x_train_vec,
        y=np.asarray(y_train_labels),
        cv=cv,
        n_jobs=-1,
    )

    # Train on full data
    estimator.fit(x_train_vec, np.asarray(y_train_labels))
    y_pred = estimator.predict(x_train_vec)

    metrics = {"model": experiment_name}
    metrics.update(cv_summary)
    # Keep flat keys for summary tables that expect these names.
    metrics.update(
        {
            "accuracy": cv_summary.get("cv_accuracy_mean"),
            "precision": cv_summary.get("cv_precision_macro_mean"),
            "recall": cv_summary.get("cv_recall_macro_mean"),
            "f1_macro": cv_summary.get("cv_f1_macro_mean"),
        }
    )

    if verbose:
        precision = cv_summary.get("cv_precision_macro_mean", "N/A")
        recall = cv_summary.get("cv_recall_macro_mean", "N/A")
        f1 = cv_summary.get("cv_f1_macro_mean", "N/A")
        accuracy = cv_summary.get("cv_accuracy_mean", "N/A")
        print(
            f"{experiment_name} | "
            f"Precision: {precision} | Recall: {recall} | F1: {f1} | Accuracy: {accuracy}"
        )

    return {
        "vectorizer": vectorizer,
        "classifier": estimator,
        "metrics": metrics,
        "predictions": y_pred,
    }


def run_transformer_encoder_experiment(
    experiment_name: str,
    model_checkpoint: str,
    X_train_text,
    y_train_labels,
    cv,
    classifier=None,
    cache_dir: str = "embeddings_cache",
    batch_size: int = 16,
    max_length: int = 96,
    device: int | None = None,
) -> dict:
    """Train transformer encoder model with CV evaluation."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required. Install: pip install transformers torch")

    if device is None:
        device, _ = detect_transformer_device()

    os.makedirs(cache_dir, exist_ok=True)

    cache_path = get_transformer_embedding_cache_path(
        model_checkpoint, "train", len(X_train_text), max_length, cache_dir
    )

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            x_train_emb = pickle.load(f)
    else:
        torch_device = "cpu" if device == -1 else f"cuda:{device}"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        hf_model = AutoModel.from_pretrained(model_checkpoint)

        x_train_emb = generate_cls_embeddings(
            texts=list(X_train_text),
            tokenizer=tokenizer,
            model=hf_model,
            batch_size=batch_size,
            max_length=max_length,
            desc=experiment_name,
            device=torch_device,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(x_train_emb, f)

    if classifier is None:
        classifier = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )

    estimator = clone(classifier)

    cv_summary = compute_cv_summary(
        estimator=estimator,
        X=x_train_emb,
        y=np.asarray(y_train_labels),
        cv=cv,
        n_jobs=-1,
    )

    # Train on full data
    estimator.fit(x_train_emb, np.asarray(y_train_labels))
    y_pred = estimator.predict(x_train_emb)

    metrics = {"model": experiment_name, "checkpoint": model_checkpoint}
    metrics.update(cv_summary)
    # Keep flat keys for summary tables that expect these names.
    metrics.update(
        {
            "accuracy": cv_summary.get("cv_accuracy_mean"),
            "precision": cv_summary.get("cv_precision_macro_mean"),
            "recall": cv_summary.get("cv_recall_macro_mean"),
            "f1_macro": cv_summary.get("cv_f1_macro_mean"),
        }
    )

    precision = cv_summary.get("cv_precision_macro_mean", "N/A")
    recall = cv_summary.get("cv_recall_macro_mean", "N/A")
    f1 = cv_summary.get("cv_f1_macro_mean", "N/A")
    accuracy = cv_summary.get("cv_accuracy_mean", "N/A")
    print(
        f"{experiment_name} | "
        f"Precision: {precision} | Recall: {recall} | F1: {f1} | Accuracy: {accuracy}"
    )

    return {
        "checkpoint": model_checkpoint,
        "classifier": estimator,
        "metrics": metrics,
        "predictions": y_pred,
        "x_train_emb": x_train_emb,
    }


def run_finetune_encoder_cv_experiment(
    model_name: str,
    checkpoint: str,
    X_text,
    y,
    cv,
    classifiers: dict | None = None,
    max_length: int = 96,
    num_train_epochs: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
) -> dict:
    """Fine-tune encoder per fold, then evaluate classifier(s) on top of embeddings."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required. Install: pip install transformers torch")

    import inspect

    class _HFTextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    def _compute_macro(y_true, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_macro": float(f1),
            "accuracy": float(acc),
        }

    def _build_training_kwargs(output_dir: str):
        kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": 8 if torch.cuda.is_available() else 4,
            "per_device_eval_batch_size": 16,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "save_strategy": "no",
            "logging_strategy": "no",
            "disable_tqdm": True,
            "report_to": [],
            "fp16": torch.cuda.is_available(),
        }

        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in ta_params:
            kwargs["evaluation_strategy"] = "epoch"
        else:
            kwargs["eval_strategy"] = "epoch"
        return kwargs

    def _build_trainer(model, tokenizer, train_ds, eval_ds, training_args):
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        }

        trainer_params = inspect.signature(Trainer.__init__).parameters
        if "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer

        return Trainer(**trainer_kwargs)

    if classifiers is None:
        classifiers = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            )
        }

    device, _ = detect_transformer_device()
    torch_device = "cpu" if device == -1 else f"cuda:{device}"
    emb_batch_size = 16

    X_arr = np.asarray(list(X_text))
    y_arr = np.asarray(y)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr), start=1):
        print(f"  Fold {fold_idx}/{cv.get_n_splits()}...")

        X_train = X_arr[train_idx]
        X_val = X_arr[val_idx]
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        train_enc = tokenizer(
            list(X_train), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        val_enc = tokenizer(
            list(X_val), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )

        train_ds = _HFTextDataset(train_enc, y_train)
        val_ds = _HFTextDataset(val_enc, y_val)

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=len(LABEL_MAP),
        )
        if torch.cuda.is_available():
            model.gradient_checkpointing_enable()

        safe_name = checkpoint.replace("/", "__")
        fold_out_dir = tempfile.mkdtemp(prefix=f"ft_{safe_name}_fold{fold_idx}_")
        training_args = TrainingArguments(**_build_training_kwargs(fold_out_dir))
        trainer = _build_trainer(model, tokenizer, train_ds, val_ds, training_args)
        trainer.train()

        # Use the fine-tuned encoder in memory; no checkpoint files are written.
        encoder_model = model.base_model
        x_train_emb = generate_cls_embeddings(
            texts=list(X_train),
            tokenizer=tokenizer,
            model=encoder_model,
            batch_size=emb_batch_size,
            max_length=max_length,
            desc=f"{model_name} FT fold {fold_idx} train emb",
            device=torch_device,
        )
        x_val_emb = generate_cls_embeddings(
            texts=list(X_val),
            tokenizer=tokenizer,
            model=encoder_model,
            batch_size=emb_batch_size,
            max_length=max_length,
            desc=f"{model_name} FT fold {fold_idx} val emb",
            device=torch_device,
        )

        for clf_name, clf in classifiers.items():
            estimator = clone(clf)
            estimator.fit(x_train_emb, y_train)
            y_pred = estimator.predict(x_val_emb)
            metrics = _compute_macro(y_val, y_pred)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "classifier": clf_name,
                    **metrics,
                }
            )
            print(
                f"    {clf_name} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1_macro']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f}"
            )

        del trainer
        del model
        del encoder_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    folds_df = pd.DataFrame(fold_metrics)
    if folds_df.empty:
        raise RuntimeError("No fold metrics were computed.")

    classifier_summary_df = (
        folds_df.groupby("classifier", as_index=False)
        .agg(
            cv_precision_mean=("precision", "mean"),
            cv_precision_std=("precision", lambda x: float(np.std(x, ddof=0))),
            cv_recall_mean=("recall", "mean"),
            cv_recall_std=("recall", lambda x: float(np.std(x, ddof=0))),
            cv_f1_macro_mean=("f1_macro", "mean"),
            cv_f1_macro_std=("f1_macro", lambda x: float(np.std(x, ddof=0))),
            cv_accuracy_mean=("accuracy", "mean"),
            cv_accuracy_std=("accuracy", lambda x: float(np.std(x, ddof=0))),
        )
        .sort_values("cv_f1_macro_mean", ascending=False)
        .reset_index(drop=True)
    )
    classifier_summary_df["model"] = classifier_summary_df["classifier"].apply(
        lambda c: f"{model_name} FT + {c}"
    )
    classifier_summary_df["encoder"] = model_name
    classifier_summary_df["checkpoint"] = checkpoint
    classifier_summary_df["num_train_epochs"] = num_train_epochs

    return {
        "encoder": model_name,
        "checkpoint": checkpoint,
        "num_train_epochs": num_train_epochs,
        "classifier_results": classifier_summary_df,
        "fold_metrics": folds_df,
    }

# ============================================================================
# TRANSFORMER UTILITIES
# ============================================================================

def detect_transformer_device() -> tuple[int, str]:
    """Detect device (GPU or CPU)."""
    if not HAS_TRANSFORMERS:
        return -1, "cpu"

    if torch.cuda.is_available():
        return 0, "cuda"
    return -1, "cpu"


def get_transformer_embedding_cache_path(
    model_checkpoint: str, split_name: str, n_examples: int, max_length: int, cache_dir: str
) -> str:
    """Build cache path for embeddings."""
    safe_name = model_checkpoint.replace("/", "__").replace("-", "_")
    return os.path.join(cache_dir, f"{split_name}_{safe_name}_n{n_examples}_l{max_length}.pkl")


def generate_cls_embeddings(
    texts, tokenizer, model, batch_size: int = 16, max_length: int = 96,
    desc: str = "Embeddings", device: str = "cpu",
) -> np.ndarray:
    """Extract [CLS] token embeddings."""
    model.eval()
    model.to(device)
    vectors = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = list(texts[i : i + batch_size])
            encoded = tokenizer(
                batch, truncation=True, padding=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            outputs = model(**encoded)
            cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            vectors.append(cls_vecs)

    return np.vstack(vectors)


def parse_decoder_label_from_text(text: str, label_map: dict | None = None) -> int | None:
    """Parse sentiment label id from decoder output text."""
    if label_map is None:
        label_map = LABEL_MAP
    label_to_id = {name.lower(): idx for idx, name in label_map.items()}
    low = str(text).lower()
    for label_name, label_id in label_to_id.items():
        if label_name in low:
            return int(label_id)
    return None


def build_decoder_prompt(text: str) -> str:
    """Build single-label sentiment prompt for decoder-only models."""
    return (
        "Classify the investor tweet sentiment into exactly one label: Bearish, Bullish, or Neutral. "
        "Reply with only one word from this set.\n\n"
        f"Tweet: {text}\nLabel:"
    )


def classify_with_decoder_batch(gen_pipe, texts, label_map: dict | None = None, batch_size: int = 16):
    """Run batched decoder inference and parse labels."""
    if label_map is None:
        label_map = LABEL_MAP
    default_label = 2 if 2 in label_map else next(iter(label_map.keys()))

    prompts = [build_decoder_prompt(t) for t in texts]
    outputs = gen_pipe(
        prompts,
        return_full_text=False,
        batch_size=batch_size,
    )

    preds = []
    for item in outputs:
        generated = item[0]["generated_text"] if isinstance(item, list) else item["generated_text"]
        pred = parse_decoder_label_from_text(generated, label_map=label_map)
        preds.append(default_label if pred is None else pred)
    return preds


def run_decoder_cv_experiment(
    model_name: str,
    checkpoint: str,
    X_text,
    y,
    cv,
    label_map: dict | None = None,
    batch_size: int = 16,
) -> dict:
    """Evaluate one decoder-only model with CV and macro metrics."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required. Install: pip install transformers torch")
    if label_map is None:
        label_map = LABEL_MAP

    X_arr = np.array(list(X_text))
    y_arr = np.array(y)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.generation_config.max_new_tokens = 6
    model.generation_config.do_sample = False
    model.generation_config.max_length = None

    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    fold_metrics = []
    for fold_idx, (_, val_idx) in enumerate(cv.split(X_arr, y_arr), start=1):
        X_val_fold = X_arr[val_idx]
        y_val_fold = y_arr[val_idx]
        y_pred = classify_with_decoder_batch(
            gen_pipe,
            X_val_fold,
            label_map=label_map,
            batch_size=batch_size,
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_fold, y_pred, average="macro", zero_division=0
        )
        acc = accuracy_score(y_val_fold, y_pred)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "precision": float(precision),
                "recall": float(recall),
                "f1_macro": float(f1),
                "accuracy": float(acc),
            }
        )
        print(
            f"  Fold {fold_idx}/{cv.get_n_splits()} | Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}"
        )

    folds_df = pd.DataFrame(fold_metrics)

    del gen_pipe
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "checkpoint": checkpoint,
        "cv_precision_mean": float(folds_df["precision"].mean()),
        "cv_precision_std": float(folds_df["precision"].std(ddof=0)),
        "cv_recall_mean": float(folds_df["recall"].mean()),
        "cv_recall_std": float(folds_df["recall"].std(ddof=0)),
        "cv_f1_macro_mean": float(folds_df["f1_macro"].mean()),
        "cv_f1_macro_std": float(folds_df["f1_macro"].std(ddof=0)),
        "cv_accuracy_mean": float(folds_df["accuracy"].mean()),
        "cv_accuracy_std": float(folds_df["accuracy"].std(ddof=0)),
        "fold_metrics": folds_df,
    }

# ============================================================================
# MODEL COMPARISON
# ============================================================================

def build_model_comparison_df(model_runs: dict, family: str) -> pd.DataFrame:
    """Build comparison dataframe from experiment runs."""
    records = []
    for exp_name, run in model_runs.items():
        metrics = run.get("metrics", {})
        records.append({
            "family": family,
            "experiment": exp_name,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "val_f1": metrics.get("f1_macro"),
            "cv_f1_mean": metrics.get("cv_f1_macro_mean"),
            "cv_f1_std": metrics.get("cv_f1_macro_std"),
            "cv_accuracy_mean": metrics.get("cv_accuracy_mean"),
            "cv_accuracy_std": metrics.get("cv_accuracy_std"),
        })
    return pd.DataFrame(records).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)


def plot_family_cv_f1(comparison_df: pd.DataFrame) -> None:
    """Plot CV F1 scores with error bars."""
    df = comparison_df.dropna(subset=["cv_f1_mean"]).copy()
    if df.empty:
        return

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x,
        df["cv_f1_mean"].values,
        yerr=df["cv_f1_std"].fillna(0).values,
        capsize=4,
        color="#4C78A8",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["experiment"].tolist(), rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("CV F1-macro (mean ± std)")
    ax.set_title("5-Fold CV F1-macro by Experiment")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_family_validation_f1(comparison_df: pd.DataFrame) -> None:
    """Plot validation F1 scores."""
    if comparison_df.empty:
        return

    plt.figure(figsize=(12, 5))
    sns.barplot(data=comparison_df, x="experiment", y="val_f1", hue="family")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Validation F1-macro")
    plt.title("Validation F1-macro by Experiment")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def style_model_comparison_table(comparison_df: pd.DataFrame):
    """Style comparison table."""
    if comparison_df.empty:
        return comparison_df

    rename_map = {
        "family": "Family",
        "experiment": "Experiment",
        "cv_f1_mean": "CV F1 Mean",
        "cv_f1_std": "CV F1 Std",
        "cv_accuracy_mean": "CV Accuracy Mean",
        "cv_accuracy_std": "CV Accuracy Std",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "val_f1": "Val F1",
    }
    table_df = comparison_df.rename(columns=rename_map)

    def _highlight_best(series):
        numeric = pd.to_numeric(series, errors="coerce")
        max_value = numeric.max()
        return [
            "font-weight: bold; color: #2c7bb6"
            if (pd.notna(v) and v == max_value) else ""
            for v in numeric
        ]

    subset = ["CV F1 Mean", "CV Accuracy Mean", "Accuracy", "Precision", "Recall", "Val F1"]

    return (
        table_df.style
        .apply(_highlight_best, subset=subset)
        .format({
            "CV F1 Mean": "{:.4f}",
            "CV F1 Std": "{:.4f}",
            "CV Accuracy Mean": "{:.4f}",
            "CV Accuracy Std": "{:.4f}",
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "Val F1": "{:.4f}",
        })
        .set_caption("Model Comparison - TF-IDF, Word2Vec, and Transformer Encoders")
    )
