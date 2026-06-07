# 📘 Text Mining Project – Step-by-Step Guide
**Market Sentiment Classification on Tweets (Bearish / Bullish / Neutral)**  
Spring Semester 2025/2026 | NOVA IMS

---

## 📁 Folder & File Structure

```
group_xx/
├── data/
│   ├── train.csv
│   └── test.csv
├── project/
│   ├── utils.py
│   ├── train_val_split.pkl
│   ├── train_val_split_no_preproc.pkl
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_feature_engineering_baseline_models.ipynb
│   ├── 03_transformer_classifiers.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── tm_final_xx.ipynb
│   └── pred_xx.csv
└── report_xx.pdf
```

> **Delivery:** zip as `group_xx.zip` and submit on Moodle by **23:59 of June 15th, 2026**.  
> ⚠️ 1.0-point penalty per half-day late. Max 15 pages in PDF report (0.5pt penalty per extra page).

---

## 🔧 Setup: `utils.py`

Before any notebook, create a shared `utils.py` with reusable imports and helper functions. This avoids repeating boilerplate across notebooks.

**What to include:**
- All common imports (`pandas`, `numpy`, `sklearn`, `nltk`, `re`, `matplotlib`, etc.)
- A function to load and save the train/val split (as `.pkl`) — **critical for reproducibility** across notebooks
- Preprocessing functions (tokenizer, stopword remover, lemmatizer, stemmer, etc.) so they can be imported everywhere
- A `evaluate_model(y_true, y_pred)` function returning a dict/DataFrame with Accuracy, Precision, Recall, F1 (macro)

**Why:** Ensures the exact same preprocessing is applied consistently in all experiments and in the final pipeline.

---

## 📓 Notebook 01 — `01_eda_preprocessing.ipynb`
**Covers:** Data Exploration (2.00 pts) + Corpus Split (0.50 pts) + Data Preprocessing (3.00 pts)

### 1.1 Data Exploration
Load `train.csv` and `test.csv`. Analyze and visualize the dataset.

**Must include:**
- Shape, dtypes, null/missing values check
- Class distribution bar chart (how many Bearish / Bullish / Neutral tweets)
  - Note: classes are likely **imbalanced** — flag this, it affects model choice later
- Tweet length distribution (characters and tokens) per class
- Word clouds per class (use `wordcloud` library)
- Most frequent tokens per class (after basic cleaning)
- Any interesting patterns or findings you can narrate — this counts for 50% of the EDA grade in the report

**Key note:** The report section on Data Exploration accounts for **50% of criteria 4.1**, so write actual conclusions, not just charts.

### 1.2 Corpus Split
Split `train.csv` into train and validation sets. **Save the split to a `.pkl` file** so all subsequent notebooks use exactly the same partition.

**Options:**
- Simple stratified `train_test_split` (e.g. 80/20), using `stratify=y` to preserve class balance
- K-Fold cross-validation (more robust, but heavier — consider for final model selection)

**Save two versions:**
- `train_val_split.pkl` — with preprocessed text (for traditional ML)
- `train_val_split_no_preproc.pkl` — with raw text (for Transformer models that tokenize internally)

### 1.3 Data Preprocessing
Implement **at least 4** of the following techniques. Apply them in a configurable pipeline so you can toggle them on/off for experiments.

| Technique | Notes |
|---|---|
| Lowercasing | Always do this first |
| Noise removal with Regex | Remove URLs (`http\S+`), mentions (`@\w+`), hashtag symbols, special characters, numbers |
| Stop word removal | Use NLTK's English stopword list; consider keeping some words (e.g. "not") |
| Tokenization | NLTK `word_tokenize` or `TweetTokenizer` (better for tweets) |
| Lemmatization | NLTK `WordNetLemmatizer`; requires POS tagging for best results |
| Stemming | NLTK `PorterStemmer` or `SnowballStemmer`; faster but less clean than lemmatization |

**Key note:**
- Tweets are noisy — regex cleaning is particularly important here (URLs, cashtags like `$AAPL`, retweet markers `RT`)
- Lemmatization and Stemming **should not be used together** on the same pipeline variant; use them as separate experiments
- Show a before/after example of preprocessing on a few tweets
- This section accounts for **25% of criteria 4.2 and 4.3** in the report

---

## 📓 Notebook 02 — `02_feature_engineering_baseline_models.ipynb`
**Covers:** Feature Engineering – BoW & Word2Vec (part of 5.50 pts) + Traditional ML Classification (part of 4.50 pts)

### 2.1 Feature Engineering — Bag of Words (BoW)

Use `sklearn`'s `CountVectorizer` and `TfidfVectorizer`.

**Variations to try:**
- `CountVectorizer` (raw counts)
- `TfidfVectorizer` (TF-IDF weights)
- Unigrams vs. bigrams (`ngram_range=(1,2)`)
- Limit vocabulary with `max_features`

### 2.2 Feature Engineering — Word2Vec

Use `gensim`'s `Word2Vec` or a pretrained model (e.g. Google News vectors).

**Approach:** Train on the training corpus, then represent each tweet as the **mean vector** of its word embeddings.

**Variations to try:**
- Train your own Word2Vec on the corpus (small, domain-specific)
- Use pretrained embeddings (larger vocabulary, better generalization)
- Try different vector sizes (e.g. 100, 300 dimensions)

### 2.3 Baseline Classification Models

For **each feature representation** above (BoW, TF-IDF, Word2Vec), train and evaluate at least two traditional ML classifiers. You need at least 2 variations of each method class.

**Models to implement:**
- KNN (`KNeighborsClassifier`)
- Logistic Regression (`LogisticRegression`)
- MLP (`MLPClassifier`)
- Random Forest (`RandomForestClassifier`)
- XGBoost (`XGBClassifier`)

**For each model:**
- Train on the train split, evaluate on validation split
- Compute and log: Accuracy, Precision (macro), Recall (macro), F1-Score (macro)
- Save all results to a `metrics_df.csv` for comparison

**Key notes:**
- Given class imbalance from EDA, consider `class_weight='balanced'` in applicable models
- Pipelines with `sklearn.pipeline.Pipeline` keep feature engineering + model in a clean chain
- F1-macro is the most informative metric here given imbalance

---

## 📓 Notebook 03 — `03_transformer_classifiers.ipynb`
**Covers:** Feature Engineering – Transformer Encoders (main part of 5.50 pts) + Transformer-based Classification (main part of 4.50 pts)

### 3.1 Transformer Feature Encoding

Use pretrained transformer models from HuggingFace to get tweet embeddings (CLS token or mean pooling). Then feed these embeddings to a traditional classifier (same as notebook 02).

**Minimum required:** 1 transformer encoder variation. Recommended models:
- `bert-base-uncased` (general purpose)
- `ProsusAI/finbert` (finance-domain — very relevant for this task, likely best performer)
- `distilbert-base-uncased` (faster, lighter)

**Extra points (0.50 pts each, max 2):** apply additional encoder models beyond the minimum.

### 3.2 Fine-tuned Transformer Classifier

Go beyond just using embeddings: **fine-tune** a transformer end-to-end for the classification task using HuggingFace's `Trainer` API or a custom PyTorch training loop.

```
AutoModelForSequenceClassification + AutoTokenizer
→ Fine-tune on train split
→ Evaluate on val split
```

**Key notes:**
- Use `train_val_split_no_preproc.pkl` here — transformers do their own tokenization, don't feed them preprocessed/stemmed text
- GPU strongly recommended; use Google Colab or Kaggle if needed
- FinBERT fine-tuned on your data is likely your strongest model — treat it as a candidate for the final pipeline
- Track training loss and validation metrics per epoch

### 3.3 (Extra) Decoder Model for Classification — +1.00 pt

Use a **decoder-only LLM** (e.g. GPT-2, LLaMA, or a quantized model via `transformers`) for classification — either zero-shot/few-shot prompting or fine-tuning. Must be clearly labeled as extra work in the report.

---

## 📓 Notebook 04 — `04_hyperparameter_tuning.ipynb`
**Covers:** Refinement of best models from notebooks 02 and 03

### 4.1 Select Top Models

From `metrics_df.csv`, pick your top 3–5 models to tune.

### 4.2 Hyperparameter Search

Use `GridSearchCV` or `RandomizedSearchCV` for traditional ML, or HuggingFace training arguments for transformers.

**Key params to tune per model:**
- Logistic Regression: `C`, `solver`, `max_iter`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`
- XGBoost: `learning_rate`, `max_depth`, `n_estimators`, `subsample`
- MLP: `hidden_layer_sizes`, `alpha`, `learning_rate`
- Transformers: `learning_rate`, `num_epochs`, `batch_size`, `warmup_steps`

### 4.3 Handle Class Imbalance (Optional but Recommended)

Try oversampling with `imbalanced-learn`:
- `SMOTE` or `RandomOverSampler` on the training set (after feature engineering, before classification)
- Compare tuned model with/without oversampling

### 4.4 Update Metrics

Append best tuned results to `metrics_df.csv` and produce a final comparison table/chart.

---

## 📓 Final Notebook — `tm_final_xx.ipynb`
**Covers:** Single, clean, ready-to-run pipeline producing `pred_xx.csv`

This notebook must be **self-contained and runnable from scratch**.

**Structure:**
1. Load `train.csv` (full training data — no validation split here, retrain on everything)
2. Apply preprocessing pipeline
3. Apply best feature engineering method
4. Train best classification model
5. Load `test.csv`, apply same preprocessing + feature engineering
6. Generate predictions → save as `pred_xx.csv` with columns `[id, label]`

**Key notes:**
- Must be a single pipeline — no branching, no experiments
- Must include all preprocessing steps inline (or import from `utils.py`)
- Use the model that performed best on validation (likely fine-tuned FinBERT)
- Double-check output: exactly 299 rows, labels in {0, 1, 2}

---

## 📊 Evaluation Metrics Cheat Sheet

For every model, always report:

| Metric | `sklearn` function | Why it matters here |
|---|---|---|
| Accuracy | `accuracy_score` | Overall correctness |
| Precision (macro) | `precision_score(average='macro')` | Penalizes false positives per class equally |
| Recall (macro) | `recall_score(average='macro')` | Penalizes missed predictions per class equally |
| F1-Score (macro) | `f1_score(average='macro')` | Harmonic mean; most informative with class imbalance |

In the report, **explain what each metric means in the context of market sentiment** — e.g. what is the cost of labeling a Bearish tweet as Bullish?

---

## 📝 PDF Report Structure (`report_xx.pdf`)

Max 15 pages. Structure it as follows:

| Section | Content | Grade Weight |
|---|---|---|
| Data Exploration | Charts + key findings + data description | 50% of criterion 4.1 |
| Data Preprocessing | Methods used + justification + examples | 25% of criteria 4.2 + 4.3 |
| Feature Engineering | Description of BoW, Word2Vec, Transformers used | 30% of criterion 4.4 |
| Classification Models | Description + rationale of models implemented | 30% of criterion 4.5 |
| Evaluation and Results | Comparison table + conclusions + best model justification | 50% of criterion 4.6 |

> ⚠️ Any **extra work** (decoder model, additional encoder, agentic workflow) **must be clearly labeled** as such in the report, or it will not count for extra points.

---

## 🏆 Extra Challenges

### Extra Challenge 1 — Best Predictions
Top 3 groups by test set performance get bonus points (1.00 / 0.50 / 0.25). Focus on maximizing your macro F1 on the test set. Fine-tuned FinBERT is typically your best shot.

### Extra Challenge 2 — Agentic Workflow (+1.50 pts)
Build a LangChain agent with a conversational interface that orchestrates the classification pipeline. It must perform a non-trivial decision, such as:
- Routing tweets to different models based on content
- Comparing outputs from multiple classifiers and picking the best
- Automating evaluation and reporting

A simple LLM prompt without tool use or orchestration does **not** count.

---

## ✅ Pre-Submission Checklist

- [ ] `tm_tests_xx.ipynb` — experiments notebook with all techniques and evaluations
- [ ] `tm_final_xx.ipynb` — clean, single pipeline, runs without errors
- [ ] `pred_xx.csv` — 299 rows, columns `id` and `label`, labels in {0, 1, 2}
- [ ] `report_xx.pdf` — max 15 pages, extra work clearly labeled
- [ ] Everything inside folder `group_xx/`, zipped if needed
- [ ] Submitted on Moodle before **23:59 on June 15th, 2026**