import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

import re

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split # , StratifiedKFold

# import xgboost as xgb
# import shap
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.metrics import roc_auc_score, average_precision_score

# import plotly
# import os

class Preprocess:
    def __init__(self):
        # Precompile phrase patterns once (faster on 10k notes)
        phrase_pairs = [
            (r"\bלחץ דם\b", "לחץ_דם"),
            (r"\bחלבון בש(ת|ׁ)ן\b", "חלבון_בשתן"),
            (r"\bהפריה חוץ גופית\b", "הפריה_חוץ_גופית"),
            (r"\bכאבי ראש\b", "כאבי_ראש"),
            (r"\bטשטוש ראייה\b", "טשטוש_ראייה"),
            (r"\bסוכרת הריון\b", "סוכרת_הריון"),
            (r"\bהיסטוריה משפחתית\b", "היסטוריה_משפחתית"),
        ]
        self.phrase_map = [(re.compile(p), r) for p, r in phrase_pairs]

        # Other compiled regexes
        self.rx_header1 = re.compile(r"^\s*תלונות[:\-]?\s*", flags=re.IGNORECASE | re.MULTILINE)
        self.rx_header2 = re.compile(r"^\s*המטופלת[:\-]?\s*", flags=re.IGNORECASE | re.MULTILINE)
        self.rx_week1   = re.compile(r"שבוע\s+\d+\s+להריון")
        self.rx_week2   = re.compile(r"שבוע(?:ות)?\s+\d+")
        self.rx_nums    = re.compile(r"\b\d+[.,]?\d*\b")
        self.rx_units   = re.compile(r"\b(dL|mg|mmHg|kg|cm)\b")
        self.rx_spaces  = re.compile(r"\s{2,}")

    def clean_note(self, s: str) -> str:
        """Join key phrases, drop headers/week/units/numbers, normalize spaces."""
        s = "" if s is None else str(s)

        # Join phrases first
        for pat, repl in self.phrase_map:
            s = pat.sub(repl, s)

        # Remove templated headers
        s = self.rx_header1.sub(" ", s)
        s = self.rx_header2.sub(" ", s)

        # Remove explicit week patterns
        s = self.rx_week1.sub(" ", s)
        s = self.rx_week2.sub(" ", s)

        # Remove standalone numbers/units (often create spurious topics)
        s = self.rx_nums.sub(" ", s)
        s = self.rx_units.sub(" ", s)

        # Normalize whitespace
        s = self.rx_spaces.sub(" ", s).strip()
        return s

    def load_data(self,path):
        """
        Load the dataset from a CSV file.
        param path: The file path to the CSV file.
        """
        if path is None:
            raise ValueError("LAUNCH_FILE_PATH environment variable not set.")
        data = pd.read_csv(path,encoding='utf-8-sig')
        return data
    
    def identify_leakage_columns(self, df, listIdentifiedLeakageColumns):
        """
        Identify leakage columns in the DataFrame.
        param df: The input DataFrame.
        param listIdentifiedLeakageColumns: List of leakage column patterns.
        """
        if len(listIdentifiedLeakageColumns) != 3:
            raise ValueError("listIdentifiedLeakageColumns must contain exactly three elements.")
        leak_cols = [
            col for col in df.columns
            if col.startswith(listIdentifiedLeakageColumns[0]) and col.endswith(listIdentifiedLeakageColumns[1])
        ]
        leak_cols += [
            col for col in df.columns
            if col.endswith(listIdentifiedLeakageColumns[2])
        ]
        leak_cols += ["match_diag_141"]

        print(f"{len(leak_cols)} columns excluded due to leakage:")
        print(leak_cols)
        return leak_cols

    def stratified_splits(self, df, leak_cols, test_size=0.20, y_column_name='Y'):
        """Perform stratified train/test splits.
        param df: The input DataFrame.
        param leak_cols: List of leakage columns to exclude.
        param test_size: Proportion of the dataset to include in the test split.
        param y_column_name: The name of the target variable column.
        """
        y = df[y_column_name].astype(int)
        X = df.drop(columns=list(set(leak_cols + [y_column_name])))
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=test_size, random_state=42, stratify=y_dev
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def collapse_diag_columns(self, X_train, X_val, X_test, diag_4_cols, diag_24_cols):
        """
        Collapse diagnosis columns into summary features and drop original columns.
        param X_train, X_val, X_test: DataFrames to process.
        param diag_4_cols, diag_24_cols: Lists of diagnosis column names for 4 and 24 months.
        """
        for X in [X_train, X_val, X_test]:
            X['any_diag_4mo'] = X[diag_4_cols].notna().any(axis=1).astype(int)
            X['any_diag_24mo'] = X[diag_24_cols].notna().any(axis=1).astype(int)
            X['num_diag_codes_4mo'] = X[diag_4_cols].notna().sum(axis=1)
            X['num_diag_codes_24mo'] = X[diag_24_cols].notna().sum(axis=1)
            X['any_diag_4mo'] = X['any_diag_4mo'].fillna(0).astype(int)
            X['any_diag_24mo'] = X['any_diag_24mo'].fillna(0).astype(int)
            X['num_diag_codes_4mo'] = X['num_diag_codes_4mo'].fillna(0).astype(int)
            X['num_diag_codes_24mo'] = X['num_diag_codes_24mo'].fillna(0).astype(int)

            X.drop(columns=diag_4_cols + diag_24_cols, inplace=True)
        print(f'New shape after collapsing diag columns: {X_train.shape}')
        print(f'X_val shape after collapsing diag columns: {X_val.shape}')
        print(f'X_test shape after collapsing diag columns: {X_test.shape}')
        return X_train, X_val, X_test
    
    def drop_high_missing_columns(self, X_train, X_val, X_test, threshold=0.5, to_print=False):
        """
        Drop columns with missingness above a given threshold.

        Parameters
        ----------
        X_train, X_val, X_test : DataFrames
            Datasets to process.
        threshold : float
            Proportion of missing values above which columns will be dropped (default=0.5).
        """
        # Compute missingness on training set (to avoid leakage)
        missing_ratio = X_train.isna().mean()
        if to_print:
            print(missing_ratio)
        drop_cols = missing_ratio[missing_ratio >= threshold].index.tolist()

        for X in [X_train, X_val, X_test]:
            X.drop(columns=drop_cols, inplace=True, errors="ignore")
        if to_print:
            print(f"Dropped {len(drop_cols)} columns due to >{threshold*100:.0f}% missing values.")
            print("Dropped columns:", drop_cols)
        return X_train, X_val, X_test, drop_cols
    
    def filter_non_empty_notes(self, X_train, X_val, X_test, column_name="clinical_sheet", min_length=10):
        """
        Filter notes in the specified column by minimum length and return filtered Series and their indices.
        param X_train, X_val, X_test: DataFrames containing the notes.
        param column_name: Name of the column containing notes.
        param min_length: Minimum length of note to be considered non-empty.
        """
        trainDocs = X_train[column_name].fillna("").astype(str)
        valDocs = X_val[column_name].fillna("").astype(str)
        testDocs = X_test[column_name].fillna("").astype(str)
        mask = trainDocs.str.len() >= min_length
        filtered_trainDocs = trainDocs[mask]
        doc_index = filtered_trainDocs.index

        print(f"Total notes: {len(X_train):,} | Non-trivial notes: {len(filtered_trainDocs):,}")
        return filtered_trainDocs, valDocs, testDocs, doc_index
            
    def normalize_phrases(text: str, phrase_map=None) -> str:
        text = "" if text is None else str(text)
        if phrase_map is None:
            phrase_map = {
                r"\bלחץ דם\b": "לחץ_דם",
                r"\bחלבון בש(ת|ׁ)ן\b": "חלבון_בשתן",
                r"\bהפריה חוץ גופית\b": "הפריה_חוץ_גופית",
                r"\bכאבי ראש\b": "כאבי_ראש",
                r"\bטשטוש ראייה\b": "טשטוש_ראייה",
                r"\bסוכרת הריון\b": "סוכרת_הריון",
                r"\bהיסטוריה משפחתית\b": "היסטוריה_משפחתית",
            }
        for pat, repl in phrase_map.items():
            text = re.sub(pat, repl, text)
        return text

    def clean_clinical_hebrew(self,s: str) -> str:
        phrase_map = {
            r"\bלחץ דם\b": "לחץ_דם",
            r"\bחלבון בש(ת|ׁ)ן\b": "חלבון_בשתן",
            r"\bהפריה חוץ גופית\b": "הפריה_חוץ_גופית",
            r"\bכאבי ראש\b": "כאבי_ראש",
            r"\bטשטוש ראייה\b": "טשטוש_ראייה",
            r"\bסוכרת הריון\b": "סוכרת_הריון",
            r"\bהיסטוריה משפחתית\b": "היסטוריה_משפחתית",
        }
        s = s or ""
        # Join key phrases (keep your existing phrase_map first!)
        for pat, repl in phrase_map.items():
            s = re.sub(pat, repl, s)

        # Remove templated headers (add more as you discover them)
        s = re.sub(r"^\s*תלונות[:\-]?\s*", " ", s, flags=re.IGNORECASE | re.MULTILINE)
        s = re.sub(r"^\s*המטופלת[:\-]?\s*", " ", s, flags=re.IGNORECASE | re.MULTILINE)

        # Collapse “שבוע <num> להריון” and bare week mentions
        s = re.sub(r"שבוע\s+\d+\s+להריון", " ", s)
        s = re.sub(r"שבוע(?:ות)?\s+\d+", " ", s)

        # Strip standalone numbers / units that create fake topics
        s = re.sub(r"\b\d+[.,]?\d*\b", " ", s)           # numbers like 8, 6, 8.5
        s = re.sub(r"\b(dL|mg|mmHg|kg|cm)\b", " ", s)    # common units
        s = re.sub(r"\s{2,}", " ", s).strip()            # normalize spaces
        return s

    def topicModel(
        self,
        he_clinical_stop,           # list of Hebrew clinical stopwords
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",  # name of sentence transformer model
        ngram_range=(1, 2),         # ngram range for CountVectorizer
        min_df=1,                   # min document frequency for CountVectorizer
        max_df=0.6,                 # max document frequency for CountVectorizer
        reduce_frequent_words=True, # whether to reduce frequent words in c-TF-IDF
        diversity=0.7,              # diversity for MaximalMarginalRelevance
        umap_n_neighbors=25,        # n_neighbors for UMAP
        umap_n_components=5,        # n_components for UMAP
        umap_min_dist=0.05,         # min_dist for UMAP
        umap_metric="cosine",       # metric for UMAP
        hdbscan_min_cluster_size=30,# min_cluster_size for HDBSCAN
        hdbscan_metric="euclidean", # metric for HDBSCAN
        hdbscan_cluster_selection_method="leaf", # cluster_selection_method for HDBSCAN
        hdbscan_prediction_data=True,            # prediction_data for HDBSCAN
        top_n_words=12,             # number of top words per topic
        min_topic_size=15,          # minimum topic size for BERTopic
        calculate_probabilities=True,# whether to calculate topic probabilities
        verbose=True,               # verbosity for BERTopic
        language=None               # language for BERTopic (None: custom stopwords)
        ):
        """
        Build and configure a BERTopic model for Hebrew clinical notes.

        Parameters:
            he_clinical_stop (list): List of Hebrew clinical stopwords.
            embedding_model_name (str): Name of the sentence transformer model.
            ngram_range (tuple): N-gram range for CountVectorizer.
            min_df (int): Minimum document frequency for CountVectorizer.
            max_df (float): Maximum document frequency for CountVectorizer.
            reduce_frequent_words (bool): Reduce frequent words in c-TF-IDF.
            diversity (float): Diversity parameter for MaximalMarginalRelevance.
            umap_n_neighbors (int): Number of neighbors for UMAP.
            umap_n_components (int): Number of components for UMAP.
            umap_min_dist (float): Minimum distance for UMAP.
            umap_metric (str): Metric for UMAP.
            hdbscan_min_cluster_size (int): Minimum cluster size for HDBSCAN.
            hdbscan_metric (str): Metric for HDBSCAN.
            hdbscan_cluster_selection_method (str): Cluster selection method for HDBSCAN.
            hdbscan_prediction_data (bool): Whether to enable prediction data in HDBSCAN.
            top_n_words (int): Number of top words per topic in BERTopic.
            min_topic_size (int): Minimum topic size in BERTopic.
            calculate_probabilities (bool): Whether to calculate topic probabilities.
            verbose (bool): Verbosity for BERTopic.
            language (str or None): Language for BERTopic (None to use custom stopwords).

        Returns:
            BERTopic: Configured BERTopic model.
        """

        # Vectorizer for Hebrew clinical notes
        vectorizer_model = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=he_clinical_stop
        )

        # Class-based TF-IDF transformer
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=reduce_frequent_words)

        # Diversify top words per topic
        representation_model = MaximalMarginalRelevance(diversity=diversity)

        # Sentence embeddings
        embedding_model = SentenceTransformer(embedding_model_name)

        # UMAP dimensionality reduction
        umap_model = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=42
        )

        # HDBSCAN clustering
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            metric=hdbscan_metric,
            cluster_selection_method=hdbscan_cluster_selection_method,
            prediction_data=hdbscan_prediction_data
        )

        # Build BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            top_n_words=top_n_words,
            min_topic_size=min_topic_size,
            calculate_probabilities=calculate_probabilities,
            verbose=verbose,
            language=language
        )

        return topic_model

    def add_topic_probabilities(
        self,
        X_train, X_val, X_test,
        trainProbs, valProbs, testProbs,
        topic_names
        ):
        """
        Add topic probability columns to train, validation, and test DataFrames.

        Parameters:
            X_train (pd.DataFrame): Training feature DataFrame.
            X_val (pd.DataFrame): Validation feature DataFrame.
            X_test (pd.DataFrame): Test feature DataFrame.
            trainProbs (np.ndarray): Topic probabilities for training set.
            valProbs (np.ndarray): Topic probabilities for validation set.
            testProbs (np.ndarray): Topic probabilities for test set.
            topic_names (dict): Mapping from topic index to topic name.

        Returns:
            tuple: (X_train, X_val, X_test) with topic probability columns added.
        """
        topic_prob_df = pd.DataFrame(
            trainProbs,
            index=X_train.index,
            columns=[f"tm_prob_{topic_names.get(i, f'Topic_{i}')}" for i in range(trainProbs.shape[1])]
        )
        X_train = pd.concat([X_train, topic_prob_df], axis=1)

        val_topic_prob_df = pd.DataFrame(
            valProbs,
            index=X_val.index,
            columns=[f"tm_prob_{topic_names.get(i, f'Topic_{i}')}" for i in range(valProbs.shape[1])]
        )
        test_topic_prob_df = pd.DataFrame(
            testProbs,
            index=X_test.index,
            columns=[f"tm_prob_{topic_names.get(i, f'Topic_{i}')}" for i in range(testProbs.shape[1])]
        )
        X_val = pd.concat([X_val, val_topic_prob_df], axis=1)
        X_test = pd.concat([X_test, test_topic_prob_df], axis=1)

        print(X_train.shape)
        X_train.head()
        return X_train, X_val, X_test
