#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IoT malware traffic detection for CTU-IoT-Malware-Capture Zeek/Bro conn.log CSV.

Binary classification:
    Benign -> 0
    Malicious* -> 1

Important:
    - detailed-label is excluded to avoid target leakage.
    - uid is excluded because it is a unique connection identifier.
    - raw IP addresses are excluded by default to reduce overfitting to specific C&C hosts.
      Use --include-ip to include id.orig_h and id.resp_h as categorical features.

Example:
    python iot_malware_detection.py \
        --data-path data/CTU-IoT-Malware-Capture-20-1conn.log.labeled.csv \
        --output-dir results

    python iot_malware_detection.py \
        --data-path data/CTU-IoT-Malware-Capture-20-1conn.log.labeled.csv \
        --output-dir results_ip \
        --include-ip
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")


NUMERIC_BASE_COLUMNS = [
    "ts",
    "id.orig_p",
    "id.resp_p",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]

CATEGORICAL_BASE_COLUMNS = [
    "proto",
    "service",
    "conn_state",
    "local_orig",
    "local_resp",
    "history",
    "tunnel_parents",
]

IP_COLUMNS = [
    "id.orig_h",
    "id.resp_h",
]

LEAKAGE_OR_ID_COLUMNS = [
    "label",
    "detailed-label",
    "uid",
]


def make_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load CSV file with pandas and print basic information.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    df = pd.read_csv(data_path)

    print("\n=== Loaded data ===")
    print(f"Path: {data_path}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nDtypes:")
    print(df.dtypes)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe:
        - replace "-" with NaN
        - remove duplicates
        - normalize target labels
        - create binary target column: target
    """
    df = df.copy()

    if "label" not in df.columns:
        raise ValueError(
            "Required column 'label' was not found. "
            "This script expects a labeled Zeek/Bro conn.log CSV."
        )

    before_shape = df.shape

    # Replace Zeek missing value marker.
    df = df.replace("-", np.nan)

    # Drop duplicate rows.
    df = df.drop_duplicates()

    # Normalize label values:
    # Example: "Malicious   C&C" -> "Malicious C&C"
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Binary target:
    # Benign -> 0
    # any label containing "Malicious" -> 1
    df["target"] = np.where(
        df["label"].str.contains("Malicious", case=False, na=False),
        1,
        0,
    )

    class_counts = df["target"].value_counts(dropna=False).to_dict()

    if df["target"].nunique() < 2:
        raise ValueError(
            "After target processing only one class remains. "
            f"Class counts: {class_counts}. "
            "Binary classification requires both Benign and Malicious samples."
        )

    print("\n=== Cleaning ===")
    print(f"Shape before cleaning: {before_shape}")
    print(f"Shape after cleaning:  {df.shape}")
    print(f"Target distribution:   {class_counts}")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature engineering columns:
        - history_len
        - bytes_ratio
        - pkts_ratio
        - total_bytes
        - total_pkts
        - hour
        - dayofweek
    """
    df = df.copy()

    # Ensure numeric conversion for columns used in arithmetic.
    for col in NUMERIC_BASE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "history" in df.columns:
        df["history_len"] = df["history"].astype(str).replace("nan", "").str.len()
    else:
        df["history_len"] = np.nan

    if "orig_bytes" in df.columns and "resp_bytes" in df.columns:
        df["bytes_ratio"] = df["orig_bytes"] / (df["resp_bytes"] + 1)
        df["total_bytes"] = df["orig_bytes"] + df["resp_bytes"]
    else:
        df["bytes_ratio"] = np.nan
        df["total_bytes"] = np.nan

    if "orig_pkts" in df.columns and "resp_pkts" in df.columns:
        df["pkts_ratio"] = df["orig_pkts"] / (df["resp_pkts"] + 1)
        df["total_pkts"] = df["orig_pkts"] + df["resp_pkts"]
    else:
        df["pkts_ratio"] = np.nan
        df["total_pkts"] = np.nan

    if "ts" in df.columns:
        ts_numeric = pd.to_numeric(df["ts"], errors="coerce")
        dt = pd.to_datetime(ts_numeric, unit="s", errors="coerce")
        df["hour"] = dt.dt.hour
        df["dayofweek"] = dt.dt.dayofweek
    else:
        df["hour"] = np.nan
        df["dayofweek"] = np.nan

    # Replace inf values caused by numeric issues.
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def plot_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save EDA tables and plots.
    """
    eda_dir = output_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution.
    class_dist = df["label"].value_counts(dropna=False).reset_index()
    class_dist.columns = ["label", "count"]
    class_dist.to_csv(eda_dir / "class_distribution.csv", index=False)

    # Missing values.
    missing = df.isna().sum().sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_count"]
    missing["missing_percent"] = missing["missing_count"] / len(df) * 100
    missing.to_csv(eda_dir / "missing_values.csv", index=False)

    # Numeric statistics.
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.describe().T.to_csv(eda_dir / "numeric_statistics.csv")

    # Plot class distribution.
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="target")
    plt.title("Class distribution: 0=Benign, 1=Malicious")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(eda_dir / "class_distribution.png", dpi=150)
    plt.close()

    # Top-10 destination ports.
    if "id.resp_p" in df.columns:
        top_ports = df["id.resp_p"].value_counts(dropna=False).head(10)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_ports.index.astype(str), y=top_ports.values)
        plt.title("Top-10 destination ports id.resp_p")
        plt.xlabel("Destination port")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(eda_dir / "top10_destination_ports.png", dpi=150)
        plt.close()

    # proto distribution.
    if "proto" in df.columns:
        proto_counts = df["proto"].value_counts(dropna=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=proto_counts.index.astype(str), y=proto_counts.values)
        plt.title("Protocol distribution")
        plt.xlabel("proto")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(eda_dir / "proto_distribution.png", dpi=150)
        plt.close()

    # conn_state distribution.
    if "conn_state" in df.columns:
        conn_counts = df["conn_state"].value_counts(dropna=False).head(20)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=conn_counts.index.astype(str), y=conn_counts.values)
        plt.title("conn_state distribution")
        plt.xlabel("conn_state")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(eda_dir / "conn_state_distribution.png", dpi=150)
        plt.close()


def prepare_features(df: pd.DataFrame, include_ip: bool):
    """
    Prepare X, y and feature lists.

    Excludes:
        - label
        - detailed-label
        - uid
        - target
        - IP columns unless include_ip=True
    """
    df = df.copy()

    y = df["target"].astype(int)

    exclude_cols = set(LEAKAGE_OR_ID_COLUMNS + ["target"])

    if not include_ip:
        exclude_cols.update(IP_COLUMNS)

    candidate_columns = [col for col in df.columns if col not in exclude_cols]

    engineered_numeric = [
        "history_len",
        "bytes_ratio",
        "pkts_ratio",
        "total_bytes",
        "total_pkts",
        "hour",
        "dayofweek",
    ]

    numeric_columns = [
        col for col in NUMERIC_BASE_COLUMNS + engineered_numeric
        if col in candidate_columns
    ]

    categorical_columns = [
        col for col in CATEGORICAL_BASE_COLUMNS
        if col in candidate_columns
    ]

    if include_ip:
        categorical_columns.extend([col for col in IP_COLUMNS if col in candidate_columns])

    # Include any remaining non-numeric candidate columns as categorical.
    known = set(numeric_columns + categorical_columns)
    remaining = [col for col in candidate_columns if col not in known]

    for col in remaining:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)

    # Convert numeric columns safely.
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[numeric_columns + categorical_columns].copy()

    features_info = {
        "include_ip": include_ip,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "excluded_columns": sorted(list(exclude_cols)),
        "all_input_features": numeric_columns + categorical_columns,
    }

    print("\n=== Features ===")
    print(f"Include IP: {include_ip}")
    print(f"Numeric features ({len(numeric_columns)}): {numeric_columns}")
    print(f"Categorical features ({len(categorical_columns)}): {categorical_columns}")
    print(f"Excluded columns: {sorted(list(exclude_cols))}")

    return X, y, features_info


def make_one_hot_encoder():
    """
    Create OneHotEncoder compatible with different scikit-learn versions.

    sklearn >= 1.2 uses sparse_output.
    Older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features, categorical_features, scale_numeric: bool = False):
    """
    Build ColumnTransformer with:
        - median imputation for numeric features
        - optional StandardScaler for numeric features
        - most frequent imputation + OneHotEncoder for categorical features
    """
    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_score_for_roc_auc(model, X_test):
    """
    Return probability or decision score for ROC-AUC.
    If unavailable, return None.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 2:
            return proba[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return None


def plot_confusion_matrix(cm, model_name: str, output_path: Path) -> None:
    """
    Save confusion matrix plot.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_and_evaluate(
    X,
    y,
    numeric_features,
    categorical_features,
    test_size: float,
    random_state: int,
    output_dir: Path,
):
    """
    Train and evaluate several models.

    Best model selection:
        The main project goal is detection of malicious IoT connections.
        Therefore, the best model is selected by F1-score for the malicious class.
        This balances malicious precision and malicious recall.
        In highly imbalanced datasets, accuracy alone is not sufficient.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    reports_dir = output_dir / "reports"
    cm_dir = output_dir / "confusion_matrices"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=False,
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=False,
                    ),
                ),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    metrics_rows = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\n=== Training: {model_name} ===")

        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision_malicious = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall_malicious = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_malicious = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        score = get_score_for_roc_auc(model, X_test)
        try:
            roc_auc = roc_auc_score(y_test, score) if score is not None else np.nan
        except Exception:
            roc_auc = np.nan

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        report_text = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malicious"],
            zero_division=0,
        )

        with open(reports_dir / f"classification_report_{model_name}.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        plot_confusion_matrix(
            cm,
            model_name,
            cm_dir / f"confusion_matrix_{model_name}.png",
        )

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "precision_malicious": precision_malicious,
                "recall_malicious": recall_malicious,
                "f1_malicious": f1_malicious,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "roc_auc": roc_auc,
                "tn": cm[0, 0],
                "fp": cm[0, 1],
                "fn": cm[1, 0],
                "tp": cm[1, 1],
            }
        )

        print(report_text)
        print(f"ROC-AUC: {roc_auc}")

    metrics_df = pd.DataFrame(metrics_rows)

    # Best model is chosen by F1-score for malicious class.
    # If there is a tie, macro F1 is used as secondary criterion.
    metrics_sorted = metrics_df.sort_values(
        by=["f1_malicious", "macro_f1"],
        ascending=False,
    )
    best_model_name = metrics_sorted.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    print("\n=== Best model ===")
    print(metrics_sorted.iloc[0])

    return metrics_df, best_model_name, best_model


def save_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    best_model_name: str,
    best_model,
    features_info: dict,
    args,
    df: pd.DataFrame,
) -> None:
    """
    Save final outputs:
        - metrics.csv
        - best_model.joblib
        - features.json
        - summary.txt
    """
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    joblib.dump(best_model, output_dir / "best_model.joblib")

    with open(output_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(features_info, f, ensure_ascii=False, indent=2)

    class_counts_label = df["label"].value_counts(dropna=False).to_dict()
    class_counts_target = df["target"].value_counts(dropna=False).to_dict()

    best_row = metrics_df[metrics_df["model"] == best_model_name].iloc[0].to_dict()

    summary = f"""IoT Malware Detection Summary

Dataset:
  path: {args.data_path}
  rows_after_cleaning: {df.shape[0]}
  columns_after_cleaning: {df.shape[1]}

Target:
  binary task: Benign=0, Malicious=1
  label distribution: {class_counts_label}
  target distribution: {class_counts_target}

Feature mode:
  include_ip: {args.include_ip}
  raw IP columns used: {"yes" if args.include_ip else "no"}

Train/test:
  test_size: {args.test_size}
  random_state: {args.random_state}
  stratify: yes

Best model:
  name: {best_model_name}
  selection criterion: highest F1-score for malicious class, macro F1 as tie-breaker

Best model metrics:
  accuracy: {best_row["accuracy"]}
  precision_malicious: {best_row["precision_malicious"]}
  recall_malicious: {best_row["recall_malicious"]}
  f1_malicious: {best_row["f1_malicious"]}
  macro_f1: {best_row["macro_f1"]}
  weighted_f1: {best_row["weighted_f1"]}
  roc_auc: {best_row["roc_auc"]}

Generated files:
  metrics.csv
  best_model.joblib
  features.json
  summary.txt
  eda/*.csv
  eda/*.png
  reports/classification_report_*.txt
  confusion_matrices/confusion_matrix_*.png

Notes:
  - detailed-label was not used as a feature to avoid leakage.
  - uid was not used as a feature.
  - raw IP addresses were used only if --include-ip was specified.
  - Accuracy is not enough for imbalanced traffic datasets, so recall/F1/ROC-AUC are reported.
"""

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="IoT malware detection from CTU-IoT Zeek/Bro conn.log labeled CSV."
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to CTU-IoT-Malware-Capture conn.log.labeled.csv file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where results will be saved.",
    )

    parser.add_argument(
        "--include-ip",
        action="store_true",
        help="Include raw IP addresses id.orig_h and id.resp_h as categorical features.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size. Default: 0.2",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state. Default: 42",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    output_dir = make_output_dir(args.output_dir)

    df = load_data(args.data_path)
    df = clean_data(df)
    df = add_features(df)

    plot_eda(df, output_dir)

    X, y, features_info = prepare_features(df, include_ip=args.include_ip)

    if y.nunique() < 2:
        raise ValueError(
            "Only one target class is available before train/test split. "
            "Cannot train binary classifier."
        )

    metrics_df, best_model_name, best_model = train_and_evaluate(
        X=X,
        y=y,
        numeric_features=features_info["numeric_features"],
        categorical_features=features_info["categorical_features"],
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=output_dir,
    )

    save_outputs(
        output_dir=output_dir,
        metrics_df=metrics_df,
        best_model_name=best_model_name,
        best_model=best_model,
        features_info=features_info,
        args=args,
        df=df,
    )

    print("\nDone.")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
