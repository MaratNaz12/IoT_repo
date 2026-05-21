import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

# Отключаем некоторые предупреждения pandas для чистоты вывода
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="IoT Malware Detection Training Script")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the Zeek CSV file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--include-ip", action="store_true", help="Include raw IP features (id.orig_h, id.resp_h)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    return parser.parse_args()

def load_data(filepath):
    print(f"[*] Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[-] Размерность данных: {df.shape}")
    print(f"[-] Колонки: {df.columns.tolist()}")
    print("[-] Типы данных по умолчанию:\n", df.dtypes)
    return df

def clean_data(df):
    print("[*] Очистка данных...")
    # 1. Замена "-" на np.nan
    df.replace('-', np.nan, inplace=True)
    
    # 2. Удаление дубликатов
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    print(f"[-] Удалено дубликатов: {initial_shape[0] - df.shape[0]}")
    
    # 3. Проверка целевой колонки
    if 'label' not in df.columns:
        raise ValueError("Ошибка: колонка 'label' отсутствует в датасете!")
    
    # 4. Нормализация целевой метки
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df['target'] = df['label'].apply(lambda x: 1 if 'malicious' in x else 0)
    
    # 5. Проверка на наличие как минимум двух классов
    if df['target'].nunique() < 2:
        raise ValueError("Ошибка: После обработки остался только один класс. Обучение невозможно.")
        
    return df

def plot_eda(df, output_dir):
    print("[*] Генерация EDA отчётов...")
    eda_dir = os.path.join(output_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    
    # Пропуски
    missing = df.isnull().sum()
    missing[missing > 0].to_csv(os.path.join(eda_dir, "missing_values.csv"))
    
    # Классы
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='target')
    plt.title("Распределение классов (0 - Benign, 1 - Malicious)")
    plt.savefig(os.path.join(eda_dir, "class_distribution.png"), bbox_inches='tight')
    plt.close()
    
    # Топ-10 портов
    if 'id.resp_p' in df.columns:
        plt.figure(figsize=(10, 5))
        df['id.resp_p'].value_counts().nlargest(10).plot(kind='bar')
        plt.title("Топ-10 портов назначения (id.resp_p)")
        plt.ylabel("Количество соединений")
        plt.savefig(os.path.join(eda_dir, "top10_ports.png"), bbox_inches='tight')
        plt.close()
        
    # Распределение протоколов
    if 'proto' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='proto', order=df['proto'].value_counts().index)
        plt.title("Распределение протоколов (proto)")
        plt.savefig(os.path.join(eda_dir, "proto_distribution.png"), bbox_inches='tight')
        plt.close()
        
    # Состояния соединений
    if 'conn_state' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x='conn_state', order=df['conn_state'].value_counts().index)
        plt.title("Распределение состояний соединений (conn_state)")
        plt.savefig(os.path.join(eda_dir, "conn_state_distribution.png"), bbox_inches='tight')
        plt.close()

def add_features(df):
    print("[*] Добавление новых признаков (Feature Engineering)...")
    
    # Приведение числовых колонок к нужному типу
    num_cols_to_convert = ['ts', 'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 
                           'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 
                           'resp_pkts', 'resp_ip_bytes']
    
    for col in num_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Признак длины истории
    if 'history' in df.columns:
        df['history_len'] = df['history'].fillna('').astype(str).apply(len)
        
    # Ратио и суммы (используем fillna(0) для безопасного деления и сложения)
    df['bytes_ratio'] = df['orig_bytes'].fillna(0) / (df['resp_bytes'].fillna(0) + 1)
    df['pkts_ratio'] = df['orig_pkts'].fillna(0) / (df['resp_pkts'].fillna(0) + 1)
    df['total_bytes'] = df['orig_bytes'].fillna(0) + df['resp_bytes'].fillna(0)
    df['total_pkts'] = df['orig_pkts'].fillna(0) + df['resp_pkts'].fillna(0)
    
    # Извлечение фичей из таймстемпа
    if 'ts' in df.columns:
        df['ts_datetime'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')
        df['hour'] = df['ts_datetime'].dt.hour
        df['dayofweek'] = df['ts_datetime'].dt.dayofweek
        df.drop(columns=['ts_datetime'], inplace=True)
        
    # Базовая статистика числовых после преобразования
    return df

def prepare_features(df, include_ip):
    print("[*] Подготовка признаков...")
    y = df['target']
    
    # Колонки для исключения
    exclude_cols = ['label', 'detailed-label', 'uid', 'target']
    if not include_ip:
        exclude_cols.extend(['id.orig_h', 'id.resp_h'])
        print("[-] Режим B: IP-адреса исключены из признаков.")
    else:
        print("[-] Режим A: IP-адреса включены в признаки (ВНИМАНИЕ: возможен лик/оверфит).")
        
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"[-] Числовые признаки ({len(num_cols)}): {num_cols}")
    print(f"[-] Категориальные признаки ({len(cat_cols)}): {cat_cols}")
    
    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    # Пайплайн для числовых данных
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Обработка совместимости OneHotEncoder
    if int(sklearn.__version__.split('.')[1]) >= 2:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
    # Пайплайн для категориальных данных
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', ohe)
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor

def plot_confusion_matrix(cm, model_name, output_dir):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"), bbox_inches='tight')
    plt.close()

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, output_dir, random_state):
    print("[*] Обучение и оценка моделей...")
    
    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000),
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state)
    }
    
    results = []
    best_f1_malicious = -1
    best_model_name = ""
    best_pipeline = None
    
    for name, model in models.items():
        print(f"[-] Обучение {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Расчет метрик
        acc = accuracy_score(y_test, y_pred)
        prec_mal = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_mal = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_mal = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC
        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_prob = pipeline.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception as e:
            print(f"    Внимание: не удалось посчитать ROC-AUC для {name} ({e})")
            roc_auc = np.nan
            
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, name, output_dir)
        
        report = classification_report(y_test, y_pred, target_names=["Benign (0)", "Malicious (1)"])
        with open(os.path.join(output_dir, f"{name}_classification_report.txt"), "w") as f:
            f.write(report)
            
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision_Malicious": prec_mal,
            "Recall_Malicious": rec_mal,
            "F1_Malicious": f1_mal,
            "F1_Macro": f1_macro,
            "F1_Weighted": f1_weighted,
            "ROC-AUC": roc_auc
        })
        
        """
        ОБОСНОВАНИЕ ВЫБОРА МОДЕЛИ:
        В задачах кибербезопасности датасеты обычно сильно несбалансированы (вредоносного трафика мало).
        Пропустить вирус (False Negative) гораздо опаснее, чем заблокировать легитимный трафик (False Positive).
        Поэтому мы выбираем лучшую модель по метрике F1-score именно для класса Malicious (1).
        Она обеспечивает лучший баланс между Precision и Recall для целевой угрозы.
        """
        if f1_mal > best_f1_malicious:
            best_f1_malicious = f1_mal
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results)
    print(f"\n[*] Лучшая модель: {best_model_name} (F1 Malicious: {best_f1_malicious:.4f})")
    return results_df, best_model_name, best_pipeline

def save_outputs(results_df, best_model_name, best_pipeline, X_cols, output_dir):
    print("[*] Сохранение результатов...")
    
    # Сохраняем метрики
    results_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    # Сохраняем модель
    joblib.dump(best_pipeline, os.path.join(output_dir, "best_model.joblib"))
    
    # Сохраняем признаки
    with open(os.path.join(output_dir, "features.json"), "w") as f:
        json.dump(list(X_cols), f, indent=4)
        
    # Сохраняем текстовое summary
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("=== IoT Malware Detection Project Summary ===\n")
        f.write(f"Лучшая модель: {best_model_name}\n")
        f.write("Эта модель была выбрана на основе метрики F1-score для вредоносного класса,\n")
        f.write("так как в задачах ИБ пропуск угрозы критичен, а классы часто несбалансированы.\n\n")
        f.write("Метрики всех моделей:\n")
        f.write(results_df.to_string())

def main():
    args = parse_args()
    
    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Загрузка данных
    df = load_data(args.data_path)
    
    # 2. Очистка
    df = clean_data(df)
    
    # 3. EDA
    plot_eda(df, args.output_dir)
    
    # 4. Feature Engineering
    df = add_features(df)
    
    # Базовая статистика числовых колонок после FE
    df.describe().to_csv(os.path.join(args.output_dir, "eda", "numeric_stats.csv"))
    
    # 5. Разделение признаков
    X, y, num_cols, cat_cols = prepare_features(df, args.include_ip)
    
    # 6. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    # 7. Препроцессор
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    # 8. Обучение и оценка
    results_df, best_model_name, best_pipeline = train_and_evaluate(
        X_train, X_test, y_train, y_test, preprocessor, args.output_dir, args.random_state
    )
    
    # 9. Сохранение артефактов
    save_outputs(results_df, best_model_name, best_pipeline, X.columns, args.output_dir)
    print(f"[*] Готово! Все результаты сохранены в папку: {args.output_dir}")

if __name__ == "__main__":
    main()
