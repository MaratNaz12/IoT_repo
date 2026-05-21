#!/usr/bin/env python3
"""
IoT Malware Traffic Detection from Zeek/Bro conn.log CSV files.

Dataset target:
- Binary classification:
    Benign -> 0
    Malicious* -> 1

Important leakage/overfitting prevention:
- "detailed-label" is never used as an input feature.
- "uid" is never used as an input feature.
- Raw IP addresses are excluded by default.
- Raw IP address features can be enabled with --include-ip.

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

ENGINEERED_NUMERIC_COLUMNS = [
    "history_len",
    "bytes_ratio",
    "pkts_ratio",
    "total_bytes",
    "total_pkts",
    "hour",
    "dayofweek",
]

BASE_CATEGORICAL_COLUMNS = [
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

ALWAYS_EXCLUDE_COLUMNS = [
    "label",
    "detailed-label",
    "uid",
    "target",
]


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the Zeek/Bro conn.log CSV file.

    The CTU-IoT conn.log labeled CSV is expected to contain a header row.
    Missing values are encoded as "-"; replacement is performed later.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {data_path}")

    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    print("\n[INFO] Raw dataset shape:")
    print(df.shape)

    print("\n[INFO] Columns:")
    print(list(df.columns))

    print("\n[INFO] Data types:")
    print(df.dtypes)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Replace "-" with NaN.
    - Remove duplicate rows.
    - Normalize label spacing.
    - Create binary target:
        Benign -> 0
        any label containing "Malicious" -> 1
    """
    df = df.copy()

    if "label" not in df.columns:
        raise ValueError(
            "Required target column 'label' is missing from the dataset."
        )

    original_shape = df.shape

    df = df.replace("-", np.nan)

    before_duplicates = len(df)
    df = df.drop_duplicates()
    after_duplicates = len(df)

    print(
        f"\n[INFO] Removed duplicate rows: "
        f"{before_duplicates - after_duplicates}"
    )

    # Normalize excessive spaces, e.g. "Malicious   C&C" -> "Malicious C&C"
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    def map_label(value: str):
        value = str(value).strip()
        if value == "Benign":
            return 0
        if "Malicious" in value:
            return 1
        return np.nan

    df["target"] = df["label"].apply(map_label)

    unmapped_count = df["target"].isna().sum()
    if unmapped_count > 0:
        print(
            f"[WARNING] Dropping {unmapped_count} rows with labels that are "
            f"neither 'Benign' nor containing 'Malicious'."
        )
        df = df.dropna(subset=["target"])

    df["target"] = df["target"].astype(int)

    class_counts = df["target"].value_counts().sort_index()
    print("\n[INFO] Binary target distribution:")
    print(class_counts)

    if df["target"].nunique() < 2:
        raise ValueError(
            "Only one class remains after preprocessing. "
            "Binary classification requires both Benign and Malicious samples."
        )

    print(f"\n[INFO] Shape before cleaning: {original_shape}")
    print(f"[INFO] Shape after cleaning:  {df.shape}")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight feature engineering useful for network traffic analysis:
    - history_len
    - bytes_ratio
    - pkts_ratio
    - total_bytes
    - total_pkts
    - hour and dayofweek from Unix timestamp ts
    """
    df = df.copy()

    # Convert expected numeric columns before feature engineering.
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
        timestamps = pd.to_datetime(df["ts"], unit="s", errors="coerce")
        df["hour"] = timestamps.dt.hour
        df["dayofweek"] = timestamps.dt.dayofweek
    else:
        df["hour"] = np.nan
        df["dayofweek"] = np.nan

    # Replace infinite values from ratios with NaN.
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def prepare_features(df: pd.DataFrame, include_ip: bool):
    """
    Prepare X, y, and feature lists.

    Exclusions:
    - label
    - detailed-label
    - uid
    - target
    - id.orig_h and id.resp_h unless include_ip=True
    """
    df = df.copy()

    exclude_columns = set(ALWAYS_EXCLUDE_COLUMNS)

    if not include_ip:
        exclude_columns.update(IP_COLUMNS)

    available_exclude_columns = [col for col in exclude_columns if col in df.columns]

    X = df.drop(columns=available_exclude_columns, errors="ignore")
    y = df["target"]

    # Ensure numeric conversion for all declared numeric columns that exist.
    numeric_columns = [
        col
        for col in NUMERIC_BASE_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
        if col in X.columns
    ]

    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    categorical_columns = [
        col for col in BASE_CATEGORICAL_COLUMNS if col in X.columns
    ]

    if include_ip:
        categorical_columns.extend([col for col in IP_COLUMNS if col in X.columns])

    # Include any remaining object/category columns as categorical to avoid
    # accidental model failures if the CSV contains extra string columns.
    for col in X.columns:
        if col not in numeric_columns and col not in categorical_columns:
            if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
                categorical_columns.append(col)

    # Final used feature list is numeric + categorical.
    used_features = numeric_columns + categorical_columns

    X = X[used_features]

    print("\n[INFO] Feature preparation complete.")
    print(f"[INFO] Include raw IP features: {include_ip}")
    print(f"[INFO] Number of numeric features: {len(numeric_columns)}")
    print(f"[INFO] Number of categorical features: {len(categorical_columns)}")
    print(f"[INFO] Total raw input features: {len(used_features)}")

    return X, y, numeric_columns, categorical_columns, used_features


def make_one_hot_encoder():
    """
    Create OneHotEncoder with scikit-learn version compatibility.

    scikit-learn >= 1.2 uses sparse_output.
    older scikit-learn versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_columns, categorical_columns) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - Numeric:
        median imputation + standard scaling
    - Categorical:
        most frequent imputation + one-hot encoding
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor


def plot_confusion_matrix(cm, model_name: str, output_path: Path):
    """
    Save a confusion matrix heatmap.
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
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_eda(df: pd.DataFrame, output_dir: Path):
    """
    Generate EDA files and plots:
    - class distribution CSV
    - missing values CSV
    - numeric descriptive statistics CSV
    - class distribution plot
    - top-10 destination ports plot
    - protocol distribution plot
    - conn_state distribution plot
    """
    eda_dir = output_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    class_distribution = (
        df["label"]
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    class_distribution.to_csv(eda_dir / "class_distribution.csv", index=False)

    missing_values = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .rename_axis("column")
        .reset_index(name="missing_count")
    )
    missing_values["missing_percent"] = (
        missing_values["missing_count"] / len(df) * 100
    )
    missing_values.to_csv(eda_dir / "missing_values.csv", index=False)

    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.describe().T.to_csv(eda_dir / "numeric_descriptive_statistics.csv")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="label", order=df["label"].value_counts().index)
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(eda_dir / "class_distribution.png", dpi=150)
    plt.close()

    if "id.resp_p" in df.columns:
        top_ports = (
            pd.to_numeric(df["id.resp_p"], errors="coerce")
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_ports.columns = ["id.resp_p", "count"]

        plt.figure(figsize=(8, 5))
        sns.barplot(data=top_ports, x="id.resp_p", y="count")
        plt.title("Top 10 Destination Ports")
        plt.xlabel("Destination Port")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(eda_dir / "top_10_destination_ports.png", dpi=150)
        plt.close()

    if "proto" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(
            data=df,
            x="proto",
            order=df["proto"].value_counts(dropna=False).index,
        )
        plt.title("Protocol Distribution")
        plt.xlabel("Protocol")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(eda_dir / "protocol_distribution.png", dpi=150)
        plt.close()

    if "conn_state" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(
            data=df,
            x="conn_state",
            order=df["conn_state"].value_counts(dropna=False).index,
        )
        plt.title("Connection State Distribution")
        plt.xlabel("Connection State")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(eda_dir / "conn_state_distribution.png", dpi=150)
        plt.close()

    print(f"[INFO] EDA outputs saved to: {eda_dir}")


def train_and_evaluate(
    X,
    y,
    numeric_columns,
    categorical_columns,
    test_size: float,
    random_state: int,
    output_dir: Path,
):
    """
    Train and evaluate multiple models.

    Model selection:
    The primary project goal is detecting malicious traffic. Therefore, the
    best model is selected by F1-score for the malicious class. This balances
    malicious precision and malicious recall. If two models tie, macro F1 can
    be inspected in metrics.csv as a secondary fairness-oriented metric.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=None,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=random_state
        ),
    }

    metrics_rows = []
    trained_pipelines = {}
    reports = {}
    confusion_matrices = {}

    for model_name, model in models.items():
        print(f"\n[INFO] Training model: {model_name}")

        preprocessor = build_preprocessor(numeric_columns, categorical_columns)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision_malicious = precision_score(
            y_test, y_pred, pos_label=1, zero_division=0
        )
        recall_malicious = recall_score(
            y_test, y_pred, pos_label=1, zero_division=0
        )
        f1_malicious = f1_score(
            y_test, y_pred, pos_label=1, zero_division=0
        )
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        roc_auc = np.nan
        try:
            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_score)
            elif hasattr(pipeline, "decision_function"):
                y_score = pipeline.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_score)
        except Exception as exc:
            warnings.warn(
                f"ROC-AUC could not be computed for {model_name}: {exc}"
            )
            roc_auc = np.nan

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        report_text = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malicious"],
            zero_division=0,
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
            }
        )

        trained_pipelines[model_name] = pipeline
        reports[model_name] = report_text
        confusion_matrices[model_name] = cm

        print(f"[INFO] {model_name} malicious recall: {recall_malicious:.4f}")
        print(f"[INFO] {model_name} malicious F1:     {f1_malicious:.4f}")
        print(f"[INFO] {model_name} ROC-AUC:          {roc_auc}")

    metrics_df = pd.DataFrame(metrics_rows)

    # Select the best model by malicious-class F1-score because the main
    # security objective is detecting malicious traffic while still penalizing
    # excessive false positives. Recall is especially important and is reported
    # separately in metrics.csv.
    best_row = metrics_df.sort_values(
        by=["f1_malicious", "macro_f1"],
        ascending=False,
    ).iloc[0]

    best_model_name = best_row["model"]
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\n[INFO] Best model selected: {best_model_name}")
    print(
        f"[INFO] Best malicious F1-score: "
        f"{best_row['f1_malicious']:.4f}"
    )

    results = {
        "metrics_df": metrics_df,
        "trained_pipelines": trained_pipelines,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "reports": reports,
        "confusion_matrices": confusion_matrices,
        "test_size": test_size,
        "random_state": random_state,
        "train_size": len(X_train),
        "test_size_count": len(X_test),
    }

    return results


def save_outputs(
    output_dir: Path,
    results: dict,
    used_features,
    numeric_columns,
    categorical_columns,
    include_ip: bool,
):
    """
    Save:
    - metrics.csv
    - classification reports
    - confusion matrix PNG files
    - best_model.joblib
    - features.json
    - summary.txt
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = results["metrics_df"]
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    reports_dir = output_dir / "classification_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for model_name, report_text in results["reports"].items():
        report_path = reports_dir / f"{model_name}_classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    for model_name, cm in results["confusion_matrices"].items():
        cm_path = cm_dir / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(cm, model_name, cm_path)

    joblib.dump(results["best_pipeline"], output_dir / "best_model.joblib")

    features_info = {
        "include_ip": include_ip,
        "used_features": used_features,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "excluded_features_always": ALWAYS_EXCLUDE_COLUMNS,
        "ip_features": IP_COLUMNS,
        "ip_features_used": include_ip,
    }

    with open(output_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(features_info, f, indent=4)

    best_model_name = results["best_model_name"]
    best_metrics = metrics_df[metrics_df["model"] == best_model_name].iloc[0]

    summary = f"""IoT Malware Detection Summary
=============================

Best model:
{best_model_name}

Model selection criterion:
The best model was selected by malicious-class F1-score, with macro F1 used as a secondary tie-breaker.

Raw IP features included:
{include_ip}

Training samples:
{results["train_size"]}

Test samples:
{results["test_size_count"]}

Best model metrics:
Accuracy:              {best_metrics["accuracy"]:.6f}
Precision malicious:   {best_metrics["precision_malicious"]:.6f}
Recall malicious:      {best_metrics["recall_malicious"]:.6f}
F1 malicious:          {best_metrics["f1_malicious"]:.6f}
Macro F1:              {best_metrics["macro_f1"]:.6f}
Weighted F1:           {best_metrics["weighted_f1"]:.6f}
ROC-AUC:               {best_metrics["roc_auc"]}

Important feature handling:
- 'uid' was excluded to avoid using a unique connection identifier.
- 'detailed-label' was excluded to avoid target leakage.
- Raw IP addresses were excluded unless --include-ip was specified.
- Missing values encoded as '-' were converted to NaN.
- Labels were normalized by collapsing excessive whitespace.
- Binary target mapping:
    Benign -> 0
    any label containing 'Malicious' -> 1

Output files:
- metrics.csv
- best_model.joblib
- features.json
- classification_reports/
- confusion_matrices/
- eda/
"""

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n[INFO] Outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Binary IoT malware traffic classification from Zeek/Bro "
            "conn.log labeled CSV files."
        )
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the CTU-IoT Zeek/Bro conn.log labeled CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where results will be saved.",
    )

    parser.add_argument(
        "--include-ip",
        action="store_true",
        help=(
            "Include raw IP address features id.orig_h and id.resp_h. "
            "By default, raw IP addresses are excluded to reduce overfitting."
        ),
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
        help="Random seed. Default: 42",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    df = clean_data(df)
    df = add_features(df)

    plot_eda(df, output_dir)

    X, y, numeric_columns, categorical_columns, used_features = prepare_features(
        df,
        include_ip=args.include_ip,
    )

    if y.nunique() < 2:
        raise ValueError(
            "Only one class is available before training. "
            "Cannot train binary classifiers."
        )

    results = train_and_evaluate(
        X=X,
        y=y,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=output_dir,
    )

    save_outputs(
        output_dir=output_dir,
        results=results,
        used_features=used_features,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        include_ip=args.include_ip,
    )


if __name__ == "__main__":
    main()
