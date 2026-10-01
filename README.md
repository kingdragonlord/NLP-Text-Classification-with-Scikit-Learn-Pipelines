# Text Classification with Scikit-Learn Pipelines

An end-to-end Natural Language Processing (NLP) workflow demonstrating modular text preprocessing, feature extraction, and multi-class document classification using custom Scikit-Learn transformers and hyperparameter optimization.

---

## Overview

This project implements and evaluates modular NLP pipelines on a subset of the **20 Newsgroups** dataset across five target categories:
- `comp.graphics`
- `sci.electronics`
- `sci.space`
- `misc.forsale`
- `rec.sport.baseball`

The objective is to design modular, reusable preprocessing components conforming to Scikit-Learn's `BaseEstimator` and `TransformerMixin` interfaces, chain them into full pipelines with `TfidfVectorizer` and classifiers, and evaluate how various preprocessing and hyperparameter tuning strategies impact generalization performance.

---

## Custom Transformers

To build clean, leakage-free pipelines, custom transformer classes were implemented:

1. **`TextCleaner`**: Applies lowercasing and strips standard punctuation using Python's `string.punctuation`.
2. **`Lemmatizer`**: Tokenizes text via `nltk.word_tokenize` and applies NLTK's `WordNetLemmatizer`.
3. **`WordFilter`**: Computes corpus word frequencies during `.fit()` to filter out NLTK stopwords, corpus-specific rare words (below a frequency threshold), and corpus-specific overly frequent words (above a percentage ceiling).
4. **`RegexSubstitutor`**: Applies flexible regex substitutions (such as normalizing numerical values to a token placeholder like `NUM`).

---

## Pipeline Architectures Evaluated

- **Pipeline 1**: `TextCleaner` $\rightarrow$ `Lemmatizer` $\rightarrow$ `TfidfVectorizer` (NLTK stopwords, word tokenizer) $\rightarrow$ `LogisticRegression`
- **Pipeline 2 (Base)**: `TextCleaner` $\rightarrow$ `WordFilter` (rare threshold = 1, freq ceiling = 0.5) $\rightarrow$ `TfidfVectorizer` $\rightarrow$ `LogisticRegression`
- **Pipeline 2 v2**: Adjusted `WordFilter` thresholds (rare threshold = 3, freq ceiling = 0.7)
- **Pipeline 3 (Base)**: `TextCleaner` $\rightarrow$ `RegexSubstitutor` (`\d+` $\rightarrow$ `NUM`) $\rightarrow$ `WordFilter` $\rightarrow$ `TfidfVectorizer` $\rightarrow$ `LogisticRegression`
- **Pipeline 3 v2**: Modified regex number matching (`\b\d+\w*\b` $\rightarrow$ `NUM`)
- **Optimized Model**: 5-fold cross-validation via `RandomizedSearchCV` across multiple classifiers (`LogisticRegression`, `Linear SVC`, `RandomForestClassifier`, and `SGDClassifier`) utilizing `roc_auc_ovr` scoring.

---

## Validation & Test Results

### Validation Set Comparison

| Pipeline / Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline 2 (Base)** | **0.867** | **0.867** | **0.866** | **0.866** |
| **Pipeline 1** | 0.866 | 0.865 | 0.866 | 0.865 |
| **Pipeline 2 v2** | 0.862 | 0.862 | 0.861 | 0.861 |
| **Optimized Model** | 0.860 | 0.859 | 0.859 | 0.858 |
| **Pipeline 3 (Base)** | 0.848 | 0.849 | 0.848 | 0.848 |
| **Pipeline 3 v2** | 0.843 | 0.844 | 0.842 | 0.843 |

### Final Test Performance (Optimized Model)

- **Test Accuracy:** 0.851 (Validation $\rightarrow$ Test: $-0.008$)
- **Test Macro Precision:** 0.852
- **Test Macro Recall:** 0.850
- **Test Macro F1:** 0.850 (Validation $\rightarrow$ Test: $-0.009$)

### Best Hyperparameters (`RandomizedSearchCV`)
- **Classifier:** `LogisticRegression(class_weight='balanced', max_iter=1000, penalty='l2', C=1.758)`
- **N-gram Range:** `(1, 3)`
- **Max Features:** `4,988`
- **Min Document Frequency (`min_df`):** `4`
- **Word Filter Rare Threshold:** `9`
- **Word Filter Frequent Threshold:** `0.281`

---

## Key Findings & Insights

1. **Preprocessing Complexity vs. Performance:** Minimal, clean preprocessing (Pipeline 1 and Pipeline 2) consistently outperformed aggressive modifications. Substituting numbers with regex tokens (Pipelines 3 & 3 v2) degraded model performance from ~86.7% down to ~84.3%, indicating that numerical figures carried meaningful domain-specific signal across technical categories (e.g., electronic specifications and space missions).
2. **Metric Selection During Optimization:** Optimizing `RandomizedSearchCV` against `roc_auc_ovr` significantly stabilized generalization, reducing the performance discrepancy between validation and unseen test sets down to under 1% (compared to ~3% variance observed when optimizing directly for accuracy or F1).
3. **Hyperparameter Tuning Plateau:** Extensive cross-validation search converged back to a balanced `LogisticRegression` configuration, yielding test metrics closely aligned with the initial base pipelines. This underscores that in standard TF-IDF text categorization, careful initial text preparation and feature representation dominate late-stage hyperparameter tuning.

---

## Setup & Execution

```bash
pip install scikit-learn nltk seaborn matplotlib pandas scipy
