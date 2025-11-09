# ======================================================
#  train_hybrid_ML.py
#  → Huấn luyện mô hình ML (RF + XGB + GB + Logistic + optional SVM)
#  → Tối ưu độ chính xác, dùng đa luồng CPU + GPU, RAM ≥ 16GB
# ======================================================

import os, time, warnings, inspect
import pandas as pd, numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib
warnings.filterwarnings("ignore")

# === Patch sklearn cho Python 3.13 ===
import sklearn
def _safe_tags(self=None): return {"allow_nan": True}
for _, cls in inspect.getmembers(sklearn, inspect.isclass):
    if not hasattr(cls, "__sklearn_tags__"):
        cls.__sklearn_tags__ = _safe_tags

# === 1. GPU Detection ===
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName
    nvmlInit()
    gpu_name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(0)).decode()
    print(f"[+] GPU detected: {gpu_name}")
except Exception:
    gpu_name = None
    print("[!] GPU not detected — using CPU.")

# === 2. Load Datasets ===
path1 = "DATASET/Phishing Detection Dataset.csv"
path2 = "DATASET/StealthPhisher2025.csv"
print("[+] Loading datasets...")
df1, df2 = pd.read_csv(path1, on_bad_lines="skip"), pd.read_csv(path2, on_bad_lines="skip")
print(f"[+] df1={df1.shape}, df2={df2.shape}")

# === 3. Chuẩn hóa nhãn ===
def normalize(df):
    if "Label" not in df.columns:
        for c in df.columns:
            if "type" in c.lower() or "label" in c.lower():
                df = df.rename(columns={c: "Label"})
                break
    df["Label"] = df["Label"].astype(str).str.lower().map({"phishing": 1, "legitimate": 0, "1": 1, "0": 0}).astype(int)
    return df
df1, df2 = normalize(df1), normalize(df2)

# === 4. Loại bỏ cột text ===
def drop_text(df):
    text_cols = [c for c in df.columns if df[c].dtype == object and c != "Label"]
    return df.drop(columns=text_cols, errors="ignore")
df1, df2 = drop_text(df1), drop_text(df2)

# === 5. Hợp nhất ===
cols = sorted(list((set(df1.columns) | set(df2.columns)) - {"Label"}))
for c in cols:
    for d in [df1, df2]:
        if c not in d: d[c] = np.nan
df = pd.concat([df1[cols+["Label"]], df2[cols+["Label"]]], ignore_index=True)
print(f"[+] Combined shape: {df.shape}")

# === 6. Xử lý NaN và chuẩn hóa ===
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)
X, y = df[cols].astype(np.float32).values, df["Label"].astype(np.int8).values

# === 7. SMOTE ===
counts = Counter(y)
ratio = min(counts.values()) / max(counts.values())
print(f"[+] Class distribution before SMOTE: {counts}, ratio={ratio:.2f}")
if ratio < 0.95:
    print("[+] Applying full SMOTE (1.0)...")
    smote = SMOTE(random_state=42, sampling_strategy=1.0, n_jobs=-1)
    X, y = smote.fit_resample(X, y)
    print(f"[+] After SMOTE: {X.shape}, balance={np.bincount(y)}")
else:
    print("[!] Skipping SMOTE — already balanced.")

# === 8. Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# === 9. Định nghĩa mô hình ===
models = [
    ("RandomForest", RandomForestClassifier(
        n_estimators=800, max_depth=35, n_jobs=-1, random_state=42)),
    ("XGBoost", XGBClassifier(
        n_estimators=900, max_depth=12, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9, gamma=0.1, reg_lambda=1.2,
        eval_metric="logloss", n_jobs=-1, tree_method="gpu_hist" if gpu_name else "hist",
        device="cuda" if gpu_name else "cpu", random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)),
    ("Logistic", LogisticRegression(
        C=3.0, max_iter=5000, solver="saga", n_jobs=-1, random_state=42))
]

trained = {}
start_all = time.time()
for name, model in tqdm(models, desc="🔄 Training Progress", ncols=100):
    t0 = time.time()
    print(f"\n⚙️  Training {name} ...")
    model.fit(X_train, y_train)
    print(f"✅ Done {name} in {(time.time()-t0)/60:.2f} min.")
    trained[name] = model
print(f"\n⏱ Total training: {(time.time()-start_all)/60:.2f} min.")

# === 10. Soft Voting ===
print("\n[+] Soft voting ensemble ...")
probs = np.array([m.predict_proba(X_test) for m in trained.values()])
weights = np.array([3, 4, 2, 1])  # RF, XGB, GB, Logistic
y_pred_prob = np.tensordot(probs, weights, axes=(0,0)) / weights.sum()
y_pred = np.argmax(y_pred_prob, axis=1)

# === 11. Manual Stacking ===
print("\n[+] Training stacking meta-model ...")
stack_train = np.column_stack([m.predict_proba(X_train)[:,1] for m in trained.values()])
stack_test  = np.column_stack([m.predict_proba(X_test)[:,1] for m in trained.values()])
meta = LogisticRegression(max_iter=5000, solver="saga", n_jobs=-1, random_state=42)
meta.fit(stack_train, y_train)
stack_pred = meta.predict(stack_test)

print("\n--- Stacking Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, stack_pred):.4f}")
print(classification_report(y_test, stack_pred))

# === 12. (Optional) SVM Subset thử nghiệm ===
use_svm = True  # đổi thành False nếu không muốn thử
if use_svm:
    from sklearn.svm import LinearSVC
    print("\n[+] Training LinearSVC on 10% subset ...")
    subset = np.random.choice(len(X_train), int(0.1 * len(X_train)), replace=False)
    svm = LinearSVC(max_iter=5000)
    svm.fit(X_train[subset], y_train[subset])
    svm_acc = svm.score(X_test, y_test)
    print(f"✅ SVM subset accuracy: {svm_acc:.4f}")

# === 13. Save ===
os.makedirs("MODELS", exist_ok=True)
joblib.dump(trained, "MODELS/ml_hybrid_models.pkl")
joblib.dump(meta, "MODELS/ml_meta_model.pkl")
joblib.dump(scaler, "MODELS/ml_scaler.pkl")
pd.Series(cols).to_csv("MODELS/ml_features.csv", index=False)
np.savez_compressed("MODELS/ml_training_data.npz",
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    y_pred_prob=y_pred_prob, feature_names=np.array(cols))

print("\n✅ Saved all models and data.")
print("=== DONE ===")
