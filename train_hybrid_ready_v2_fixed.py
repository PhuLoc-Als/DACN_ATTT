# ======================================================
#  train_hybrid_ready_v2_fixed.py
#  → Huấn luyện mô hình ML (RF + XGB + Logistic + Stacking)
#  → Tối ưu RAM + độ chính xác cao, tương thích Python 3.13
# ======================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from collections import Counter
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

# === Universal Patch for sklearn + Python 3.13 ===
import sklearn, inspect

def _safe_tags(self=None):
    return {
        "non_deterministic": False,
        "multioutput": False,
        "allow_nan": True,
        "poor_score": False,
        "requires_positive_X": False,
        "stateless": False,
    }

# Vá toàn bộ class trong sklearn
for _, cls in inspect.getmembers(sklearn, inspect.isclass):
    if not hasattr(cls, "__sklearn_tags__"):
        cls.__sklearn_tags__ = _safe_tags

# Vá cả BaseEstimator và clone logic
if not hasattr(sklearn.base.BaseEstimator, "__sklearn_tags__"):
    sklearn.base.BaseEstimator.__sklearn_tags__ = _safe_tags

try:
    from sklearn.utils import _tags
    _tags.get_tags = lambda est: getattr(est, "__sklearn_tags__", _safe_tags)()
except Exception:
    pass

# === 1. Kiểm tra GPU ===
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName
    nvmlInit()
    gpu_name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(0)).decode()
    print(f"[+] GPU detected: {gpu_name}")
except Exception:
    gpu_name = None
    print("[!] GPU not detected — fallback to CPU.")

# === 2. Đường dẫn dataset ===
path1 = "DATASET/Phishing Detection Dataset.csv"
path2 = "DATASET/StealthPhisher2025.csv"

print("[+] Loading datasets...")
df1 = pd.read_csv(path1, encoding="utf-8", on_bad_lines="skip")
df2 = pd.read_csv(path2, encoding="utf-8", on_bad_lines="skip")
print(f"[+] df1={df1.shape}, df2={df2.shape}")

# === 3. Chuẩn hóa nhãn ===
def normalize_label_df1(df):
    df = df.rename(columns={"Type": "Label"})
    df["Label"] = df["Label"].astype(int)
    return df

def normalize_label_df2(df):
    label_cols = [c.lower() for c in df.columns]
    if "label" in label_cols:
        df = df.rename(columns={df.columns[label_cols.index("label")]: "Label"})
        df["Label"] = df["Label"].astype(str).str.lower().map({"phishing": 1, "legitimate": 0})
    else:
        phish_cols = [c for c in df.columns if "phish" in c.lower()]
        if len(phish_cols):
            df["Label"] = df[phish_cols[0]].apply(lambda x: 1 if x == 1 else 0)
        else:
            raise ValueError("Không tìm thấy cột nhãn phù hợp trong StealthPhisher2025.csv")
    return df

df1 = normalize_label_df1(df1)
df2 = normalize_label_df2(df2)

# === 4. Loại bỏ cột text / domain ===
def drop_text_columns(df):
    drop_cols = []
    for c in df.columns:
        if any(k in c.lower() for k in ["url", "domain", "tld"]) or df[c].dtype == object:
            drop_cols.append(c)
    drop_cols = [c for c in drop_cols if c != "Label"]
    return df.drop(columns=drop_cols, errors="ignore")

df1_clean = drop_text_columns(df1)
df2_clean = drop_text_columns(df2)

# === 5. Hợp nhất cột (union) ===
cols1 = [c for c in df1_clean.columns if c != "Label"]
cols2 = [c for c in df2_clean.columns if c != "Label"]
all_cols = sorted(list(set(cols1 + cols2)))

for c in all_cols:
    if c not in df1_clean.columns:
        df1_clean[c] = np.nan
    if c not in df2_clean.columns:
        df2_clean[c] = np.nan

df = pd.concat([df1_clean[all_cols + ["Label"]], df2_clean[all_cols + ["Label"]]], ignore_index=True)
print(f"[+] Combined dataset shape: {df.shape}")

# === 6. Chuẩn hóa numeric + xử lý NaN ===
for c in all_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

# 💡 Giảm RAM tiêu thụ
X = df[all_cols].astype(np.float32).values
y = df["Label"].astype(np.int8).values

# === 7. Cân bằng dữ liệu bằng SMOTE (nếu cần) ===
counts = Counter(y)
minority_ratio = min(counts.values()) / max(counts.values())
print(f"[+] Class distribution before SMOTE: {counts}, ratio={minority_ratio:.2f}")

if minority_ratio < 0.9:
    print("[+] Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, sampling_strategy=0.8, n_jobs=4)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[+] After SMOTE: {X_res.shape}, [class balance: {np.bincount(y_res)}]")
else:
    print("[!] Dataset already balanced — skipping SMOTE.")
    X_res, y_res = X, y

# === 8. Chia dữ liệu train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# === 9. Chuẩn hóa dữ liệu ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 10. Huấn luyện mô hình ML riêng biệt ===
print("\n[+] Training individual models (RF + XGB + Logistic)...")

models = [
    ("Random Forest", RandomForestClassifier(
        n_estimators=300,     # Giảm số cây để tránh tràn RAM
        max_depth=25,
        random_state=42,
        n_jobs=4              # Giới hạn CPU threads
    )),
    ("XGBoost", XGBClassifier(
        n_estimators=600,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        reg_lambda=1.2,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        tree_method="hist",
        device="cuda" if gpu_name else "cpu"
    )),
    ("Logistic", LogisticRegression(
        C=2.0, max_iter=3000, n_jobs=4, random_state=42
    ))
]

trained_models = {}
start_all = time.time()

for name, model in tqdm(models, desc="🔄 Training Progress", ncols=100):
    print(f"\n⚙️ Bắt đầu huấn luyện {name} ...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    elapsed = time.time() - start_time
    print(f"✅ Hoàn thành {name} trong {elapsed/60:.2f} phút.")
    trained_models[name] = model

end_all = time.time()
print(f"\n⏱ Tổng thời gian huấn luyện: {(end_all - start_all)/60:.2f} phút.")
print("\n🎯 Tất cả mô hình đã được huấn luyện xong!")

# === 11. Voting mềm ===
print("[+] Combining predictions (soft voting)...")
rf_pred = trained_models["Random Forest"].predict_proba(X_test_scaled)
xgb_pred = trained_models["XGBoost"].predict_proba(X_test_scaled)
log_pred = trained_models["Logistic"].predict_proba(X_test_scaled)

weights = np.array([3, 4, 1])
y_pred_prob = (rf_pred * weights[0] + xgb_pred * weights[1] + log_pred * weights[2]) / weights.sum()
y_pred = np.argmax(y_pred_prob, axis=1)

# === 12. Stacking meta-model (thay thế thủ công không dùng StackingClassifier) ===
print("\n[+] Training stacking meta-model (RF + XGB + Logistic)...")

# Tạo đầu vào mới từ xác suất dự đoán của các mô hình con
stack_train = np.column_stack([
    trained_models["Random Forest"].predict_proba(X_train_scaled)[:, 1],
    trained_models["XGBoost"].predict_proba(X_train_scaled)[:, 1],
    trained_models["Logistic"].predict_proba(X_train_scaled)[:, 1]
])

stack_test = np.column_stack([
    trained_models["Random Forest"].predict_proba(X_test_scaled)[:, 1],
    trained_models["XGBoost"].predict_proba(X_test_scaled)[:, 1],
    trained_models["Logistic"].predict_proba(X_test_scaled)[:, 1]
])

# Dùng Logistic Regression làm meta-model
meta_model = LogisticRegression(max_iter=3000, random_state=42)
meta_model.fit(stack_train, y_train)
stack_pred = meta_model.predict(stack_test)

print("\n--- Stacking Evaluation (Custom Manual) ---")
print(f"Accuracy: {accuracy_score(y_test, stack_pred):.4f}")
print(classification_report(y_test, stack_pred))

# Lưu mô hình meta
os.makedirs("MODELS", exist_ok=True)
joblib.dump(meta_model, "MODELS/ml_meta_model.pkl")
print("✅ Saved custom meta-model to MODELS/ml_meta_model.pkl")

# === 13. Voting Evaluation ===
print("\n--- Voting Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# === 14. Lưu mô hình và dữ liệu cho DL ===
os.makedirs("MODELS", exist_ok=True)
joblib.dump(trained_models, "MODELS/ml_hybrid_models.pkl")
joblib.dump(scaler, "MODELS/ml_scaler.pkl")
pd.Series(all_cols).to_csv("MODELS/ml_features.csv", index=False)

np.savez_compressed(
    "MODELS/ml_training_data.npz",
    X_train=X_train_scaled,
    y_train=y_train,
    X_test=X_test_scaled,
    y_test=y_test,
    y_pred_prob=y_pred_prob,
    feature_names=np.array(all_cols)
)

print("\n✅ Saved models to MODELS/ml_hybrid_models.pkl")
print("✅ Saved scaler to MODELS/ml_scaler.pkl")
print("✅ Saved feature list to MODELS/ml_features.csv")
print("✅ Saved train/test data for DL to MODELS/ml_training_data.npz")
print("\n=== DONE ===")
