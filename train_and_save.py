# train_and_save.py
import os, pickle, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb

DATA_PATH = "cardio_train.csv"  # <- your dataset file (must be in same folder)

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Put cardio_train.csv in this folder.")
    df = pd.read_csv(DATA_PATH)
    print("Loaded df shape:", df.shape)
    print("Columns:", list(df.columns))

    # Basic cleanup
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df.replace(["", " ", "NA", "NaN", "nan"], np.nan, inplace=True)
    df.dropna(inplace=True)
    print("After dropna shape:", df.shape)

    # Ensure age_yr exists (common cardio dataset has age in days)
    if 'age_yr' not in df.columns:
        if 'age' in df.columns:
            if df['age'].mean() > 100:  # likely days
                df['age_yr'] = (df['age'] / 365).round(0).astype(int)
                print("Converted 'age' (days) -> 'age_yr'")
            else:
                df['age_yr'] = df['age'].astype(int)
                print("Copied 'age' -> 'age_yr'")
        else:
            raise Exception("No 'age' or 'age_yr' column found. Edit dataset or script.")

    # Compute BMI
    if 'height' in df.columns and 'weight' in df.columns:
        df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
        print("Computed BMI.")
    elif 'BMI' not in df.columns:
        raise Exception("Need BMI or height+weight columns.")

    # Feature list (change if your original code used different features)
    features = ['age_yr', 'ap_hi', 'ap_lo', 'cholesterol', 'BMI']
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise Exception(f"Missing features in dataset: {missing}. Edit the features list.")

    X = df[features].values
    if 'cardio' not in df.columns:
        raise Exception("Target column 'cardio' not found. Rename target or update code.")
    y = df['cardio'].astype(int).values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Train/test shapes:", X_train.shape, X_test.shape)

    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    print("Trained XGBoost.")

    # Train Logistic Regression
    logreg = LogisticRegression(max_iter=2000)
    logreg.fit(X_train_scaled, y_train)
    print("Trained LogisticRegression.")

    # Ensemble evaluation
    w_xgb, w_lr = 0.7, 0.3
    p_xgb = xgb_model.predict_proba(X_test)[:,1]
    p_lr = logreg.predict_proba(X_test_scaled)[:,1]
    p_ensemble = w_xgb * p_xgb + w_lr * p_lr
    pred_ensemble = (p_ensemble >= 0.5).astype(int)

    print("Ensemble Accuracy:", round(accuracy_score(y_test, pred_ensemble),5))
    print("Ensemble AUC:", round(roc_auc_score(y_test, p_ensemble),5))
    print(classification_report(y_test, pred_ensemble))

    # Save artifacts
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open("logreg_model.pkl", "wb") as f:
        pickle.dump(logreg, f)
    meta = {"features": features, "ensemble_weights": {"xgb": w_xgb, "logreg": w_lr}}
    with open("meta.json", "w") as f:
        json.dump(meta, f)

    print("Saved: scaler.pkl, xgb_model.pkl, logreg_model.pkl, meta.json")

if __name__ == "__main__":
    main()
