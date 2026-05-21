import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

def main():
    parser = argparse.ArgumentParser(description="IoT Malware Network Traffic Classification")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the Zeek conn.log CSV file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--include-ip", action="store_true", help="Flag to include raw IP address features")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # 1. Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Load Data
    df = load_data(args.data_path)
    
    # 3. Clean Data
    df = clean_data(df)
    
    # 4. Exploratory Data Analysis (EDA)
    plot_eda(df, args.output_dir)
    
    # 5. Feature Engineering
    df = add_features(df)
    
    # 6. Prepare Features & Split
    X, y, num_cols, cat_cols = prepare_features(df, args.include_ip)
    
    # Save the feature list
    with open(os.path.join(args.output_dir, "features.json"), "w") as f:
        json.dump({"numeric": num_cols, "categorical": cat_cols, "target": "target"}, f, indent=4)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    # 7. Build Preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    # 8. Train and Evaluate Models
    results, models = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, args.random_state)
    
    # 9. Save Outputs and Best Model
    save_outputs(results, models, X_test, y_test, args.output_dir)


def load_data(filepath):
    print(f"[*] Loading data from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The file {filepath} does not exist.")
    
    df = pd.read_csv(filepath)
    print(f"[*] Data loaded successfully. Shape: {df.shape}")
    print("[*] Columns:", list(df.columns))
    print("[*] Data Types:\n", df.dtypes)
    return df


def clean_data(df):
    print("[*] Cleaning data...")
    # Replace missing values encoded as "-"
    df.replace("-", np.nan, inplace=True)
    
    # Remove duplicate rows
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    print(f"[*] Dropped duplicates. New shape: {df.shape} (Removed {initial_shape[0] - df.shape[0]} rows)")
    
    # Ensure label column exists
    if "label" not in df.columns:
        raise ValueError("Error: 'label' column is missing from the dataset.")
        
    # Normalize label values (strip whitespace, remove excessive spaces)
    df["label"] = df["label"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    
    # Create binary target variable (Benign -> 0, Malicious -> 1)
    df["target"] = df["label"].apply(lambda x: 1 if "Malicious" in x else 0)
    
    # Check for valid single-class distributions
    if df["target"].nunique() < 2:
        raise ValueError("Error: Only one class remains after preprocessing. Cannot train binary classification models.")
        
    return df


def plot_eda(df, output_dir):
    print("[*] Performing EDA...")
    eda_dir = os.path.join(output_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    
    # Class Distribution CSV
    class_dist = df["label"].value_counts().reset_index()
    class_dist.columns = ["label", "count"]
    class_dist.to_csv(os.path.join(eda_dir, "class_distribution.csv"), index=False)
    
    # Missing Values CSV
    missing_vals = df.isnull().sum().reset_index()
    missing_vals.columns = ["column", "missing_count"]
    missing_vals.to_csv(os.path.join(eda_dir, "missing_values.csv"), index=False)
    
    # Descriptive Statistics for numeric columns (converting dynamically for describe)
    num_cols_eda = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
    temp_num_df = df[num_cols_eda].apply(pd.to_numeric, errors='coerce')
    temp_num_df.describe().to_csv(os.path.join(eda_dir, "descriptive_statistics.csv"))
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, y="label", order=df["label"].value_counts().index)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "class_distribution.png"))
    plt.close()
    
    # 2. Top-10 Destination Ports Plot
    plt.figure(figsize=(10, 5))
    df["id.resp_p"].value_counts().head(10).plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Top 10 Destination Ports")
    plt.ylabel("Frequency")
    plt.xlabel("Port Number")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "top_10_destination_ports.png"))
    plt.close()
    
    # 3. Protocol Distribution Plot
    if "proto" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="proto", order=df["proto"].value_counts().index)
        plt.title("Protocol Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, "protocol_distribution.png"))
        plt.close()
        
    # 4. Connection State Distribution Plot
    if "conn_state" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x="conn_state", order=df["conn_state"].value_counts().index)
        plt.title("Connection State Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, "conn_state_distribution.png"))
        plt.close()


def add_features(df):
    print("[*] Adding engineered features...")
    # 1. Time-based features
    df["ts_num"] = pd.to_numeric(df["ts"], errors="coerce")
    dt_series = pd.to_datetime(df["ts_num"], unit="s", errors="coerce")
    df["hour"] = dt_series.dt.hour
    df["dayofweek"] = dt_series.dt.dayofweek
    
    # 2. History length
    df["history_len"] = df["history"].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
    
    # Explicitly convert base columns to numeric for calculations
    numeric_bases = [
        "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
        "ts", "id.orig_p", "id.resp_p", "duration", "missed_bytes",
        "orig_ip_bytes", "resp_ip_bytes"
    ]
    for col in numeric_bases:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Temporary zero-filled columns for math operations to prevent NaN propagation entirely
    ob = df["orig_bytes"].fillna(0)
    rb = df["resp_bytes"].fillna(0)
    op = df["orig_pkts"].fillna(0)
    rp = df["resp_pkts"].fillna(0)
    
    # 3. Ratio and Total features
    df["bytes_ratio"] = ob / (rb + 1)
    df["pkts_ratio"] = op / (rp + 1)
    df["total_bytes"] = ob + rb
    df["total_pkts"] = op + rp
    
    return df


def prepare_features(df, include_ip):
    print(f"[*] Preparing features (Include IPs: {include_ip})...")
    # Base excluded columns
    drop_cols = ["label", "detailed-label", "uid", "target", "ts", "ts_num"]
    
    # Exclude raw IP addresses if mode B (default)
    if not include_ip:
        drop_cols.extend(["id.orig_h", "id.resp_h"])
        
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df["target"]
    
    # Define Column Types
    numeric_cols = [
        "id.orig_p", "id.resp_p", "duration", "orig_bytes", "resp_bytes",
        "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
        "history_len", "bytes_ratio", "pkts_ratio", "total_bytes", "total_pkts",
        "hour", "dayofweek"
    ]
    
    categorical_cols = [
        "proto", "service", "conn_state", "local_orig", "local_resp",
        "history", "tunnel_parents"
    ]
    
    if include_ip:
        categorical_cols.extend(["id.orig_h", "id.resp_h"])
        
    # Filter only columns present in X
    num_cols_present = [col for col in numeric_cols if col in X.columns]
    cat_cols_present = [col for col in categorical_cols if col in X.columns]
    
    return X, y, num_cols_present, cat_cols_present


def build_preprocessor(num_cols, cat_cols):
    # Numeric Pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Handle scikit-learn OneHotEncoder compatibility (sparse vs sparse_output)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        
    # Categorical Pipeline
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    
    return preprocessor


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, random_state):
    print("[*] Training models...")
    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000),
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state)
    }
    
    results = {}
    trained_pipelines = {}
    
    for name, model in models.items():
        print(f"  -> Fitting {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro_f1 = report_dict.get("macro avg", {}).get("f1-score", 0)
        weighted_f1 = report_dict.get("weighted avg", {}).get("f1-score", 0)
        
        # ROC-AUC calculation with exception handling
        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                y_decision = pipeline.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_decision)
        except Exception:
            roc_auc = np.nan
            
        results[name] = {
            "Accuracy": acc,
            "Precision_Malicious": prec,
            "Recall_Malicious": rec,
            "F1_Malicious": f1,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "ROC_AUC": roc_auc
        }
        
        trained_pipelines[name] = pipeline
        print(f"     Metrics -> F1_Malicious: {f1:.4f} | Recall_Malicious: {rec:.4f} | ROC_AUC: {roc_auc:.4f}")
        
    return results, trained_pipelines


def save_outputs(results, models, X_test, y_test, output_dir):
    print("[*] Saving outputs and evaluating best model...")
    
    # 1. Save Metrics to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, "metrics.csv"))
    
    # 2. Find the best model based on F1_Malicious
    best_model_name = max(results, key=lambda k: results[k]["F1_Malicious"])
    best_pipeline = models[best_model_name]
    
    # 3. Save confusion matrices and reports
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        
        # Text Report
        report = classification_report(y_test, y_pred, target_names=["Benign", "Malicious"])
        with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
            f.write(report)
            
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malicious"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
        plt.close()
        
    # 4. Save best model
    joblib.dump(best_pipeline, os.path.join(output_dir, "best_model.joblib"))
    
    # 5. Write Summary text
    summary_text = (
        f"IoT Malware Detection - Execution Summary\n"
        f"=========================================\n\n"
        f"Best Model Selected: {best_model_name}\n"
        f"Selection Criterion: The model was chosen based on the highest Malicious-class F1-Score.\n\n"
        f"Performance of Best Model:\n"
        f"- Accuracy:          {results[best_model_name]['Accuracy']:.4f}\n"
        f"- Precision (Mal):   {results[best_model_name]['Precision_Malicious']:.4f}\n"
        f"- Recall (Mal):      {results[best_model_name]['Recall_Malicious']:.4f}\n"
        f"- F1-Score (Mal):    {results[best_model_name]['F1_Malicious']:.4f}\n"
        f"- Macro F1:          {results[best_model_name]['Macro_F1']:.4f}\n"
        f"- ROC AUC:           {results[best_model_name]['ROC_AUC']:.4f}\n\n"
        f"Note: Recall for the malicious class is heavily prioritized as missing malware traffic is typically costlier than false alarms."
    )
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary_text)
        
    print(f"[*] Done! Best model was {best_model_name}. All results saved to {output_dir}")

if __name__ == "__main__":
    main()
