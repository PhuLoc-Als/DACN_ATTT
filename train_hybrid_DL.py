# ======================================================
#  train_hybrid_DL.py
#  Mô hình Hybrid Deep Learning (CNN + LSTM)
#  Dataset: Phish-IRIS (ảnh) + DeepURLBench (URL)
#  GPU RTX 3050
# ======================================================

import os, random, time, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Embedding, LSTM, Dense, Dropout, concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# === 1. GPU setup ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU khả dụng:", gpus)
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass
else:
    print("⚠️ Không tìm thấy GPU, TensorFlow sẽ dùng CPU.")

# === 2. Cấu hình ===
IMG_DIR = r"D:\DACN\DEMO_DACN\DATASET\PhishIRIS"
URL_DIRS = [
    "DATASET/DeepURLBench/data/urls_with_dns",
    "DATASET/DeepURLBench/data/urls_without_dns"
]
IMG_SIZE = (128, 128)
MAX_WORDS = 5000
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 5

# === 3. Đọc dữ liệu URL ===
print("[1/4] Đang đọc dữ liệu URL từ DeepURLBench...")
df_list = []
for DIR in URL_DIRS:
    if not os.path.exists(DIR): continue
    for f in os.listdir(DIR):
        if f.endswith(".parquet"):
            try:
                df = pd.read_parquet(os.path.join(DIR, f), engine="pyarrow")
                df_list.append(df.sample(min(5000, len(df)), random_state=42))
            except Exception as e:
                print("⚠️", f, ":", e)
df = pd.concat(df_list, ignore_index=True)
df = df.dropna(subset=["url"])
print(f"✅ Tổng URL đọc được: {len(df):,}")

# phân lớp URL theo nhãn
urls_phish = df[df["label"].astype(str).str.lower().str.contains("phish")]["url"].tolist()
urls_legit = df[~df["label"].astype(str).str.lower().str.contains("phish")]["url"].tolist()
if not urls_legit:
    urls_legit = df.sample(len(urls_phish))["url"].tolist()
print(f"🔹 URL phishing: {len(urls_phish)}, legit: {len(urls_legit)}")

# === 4. Đọc ảnh từ Phish-IRIS ===
print("\n[2/4] Đang đọc ảnh từ Phish-IRIS...")
train_dir = os.path.join(IMG_DIR, "train")
val_dir   = os.path.join(IMG_DIR, "val")

def load_images_from(root_dir, label):
    data = []
    for brand in os.listdir(root_dir):
        brand_path = os.path.join(root_dir, brand)
        if not os.path.isdir(brand_path): continue
        for f in os.listdir(brand_path):
            if f.lower().endswith(('.png','.jpg')):
                data.append((os.path.join(brand_path, f), label))
    return data

imgs_phish = load_images_from(os.path.join(train_dir, "phishing"), 1) if os.path.exists(os.path.join(train_dir, "phishing")) else []
imgs_legit = load_images_from(os.path.join(train_dir, "legitimate"), 0) if os.path.exists(os.path.join(train_dir, "legitimate")) else []

if not imgs_phish or not imgs_legit:
    # fallback: dùng tất cả thư mục train, val chung, tạm chia nửa
    all_imgs = []
    for split in [train_dir, val_dir]:
        for brand in os.listdir(split):
            brand_path = os.path.join(split, brand)
            if os.path.isdir(brand_path):
                for f in os.listdir(brand_path):
                    if f.lower().endswith(('.png','.jpg')):
                        all_imgs.append(os.path.join(brand_path, f))
    half = len(all_imgs)//2
    imgs_phish = [(p,1) for p in all_imgs[:half]]
    imgs_legit = [(p,0) for p in all_imgs[half:]]
print(f"✅ Ảnh phishing: {len(imgs_phish)}, legit: {len(imgs_legit)}")

# === 5. Tạo cặp ảnh + URL ===
min_len = min(len(imgs_phish), len(urls_phish), len(imgs_legit), len(urls_legit))
imgs_phish, urls_phish = imgs_phish[:min_len], urls_phish[:min_len]
imgs_legit, urls_legit = imgs_legit[:min_len], urls_legit[:min_len]

data_pairs = [(i[0], u, 1) for i, u in zip(imgs_phish, urls_phish)] + [(i[0], u, 0) for i, u in zip(imgs_legit, urls_legit)]
random.shuffle(data_pairs)
print(f"✅ Tổng số cặp ảnh+URL: {len(data_pairs)}")

# === 6. Xử lý văn bản URL ===
texts = [u for _,u,_ in data_pairs]
tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
tokenizer.fit_on_texts(texts)
X_url = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN)

# === 7. Xử lý ảnh ===
def load_image(path):
    try:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        return np.array(img)/255.0
    except:
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3))

X_img = np.array([load_image(p) for p,_,_ in data_pairs])
y = np.array([lbl for _,_,lbl in data_pairs])
print(f"✅ Dữ liệu ảnh: {X_img.shape}, URL: {X_url.shape}, nhãn: {y.shape}")

# === 8. Chia train/test ===
split = int(0.8 * len(y))
X_img_train, X_img_test = X_img[:split], X_img[split:]
X_url_train, X_url_test = X_url[:split], X_url[split:]
y_train, y_test = y[:split], y[split:]

# === 9. Xây mô hình CNN + LSTM ===
print("\n[3/4] Xây dựng mô hình hybrid...")

# CNN branch (ảnh)
img_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x1 = Conv2D(32, (3,3), activation='relu')(img_input)
x1 = MaxPooling2D(2)(x1)
x1 = Conv2D(64, (3,3), activation='relu')(x1)
x1 = MaxPooling2D(2)(x1)
x1 = Flatten()(x1)

# LSTM branch (URL)
url_input = Input(shape=(MAX_LEN,))
x2 = Embedding(MAX_WORDS, 128)(url_input)
x2 = LSTM(64)(x2)

# Merge
merged = concatenate([x1, x2])
merged = Dense(128, activation='relu')(merged)
merged = Dropout(0.3)(merged)
output = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[img_input, url_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === 10. Huấn luyện ===
os.makedirs("MODELS", exist_ok=True)
checkpoint = ModelCheckpoint("MODELS/hybrid_phishiris_best.h5", save_best_only=True,
                             monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

start = time.time()
history = model.fit([X_img_train, X_url_train], y_train,
                    validation_data=([X_img_test, X_url_test], y_test),
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[checkpoint, early_stop], verbose=2)
end = time.time()
print(f"\n⏱ Thời gian huấn luyện: {(end - start)/60:.2f} phút")

# === 11. Đánh giá ===
y_pred = (model.predict([X_img_test, X_url_test]) > 0.5).astype(int)
print("\n--- Đánh giá mô hình ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Legit", "Phish"]))

# === 12. Biểu đồ ===
os.makedirs("MODELS/LOGS", exist_ok=True)
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title("Hybrid Model Accuracy")
plt.savefig("MODELS/LOGS/hybrid_phishiris_acc.png", dpi=150)
plt.close()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Hybrid Model Loss")
plt.savefig("MODELS/LOGS/hybrid_phishiris_loss.png", dpi=150)
plt.close()

print("✅ Mô hình đã lưu tại: MODELS/hybrid_phishiris_best.h5")
print("📊 Biểu đồ đã lưu tại: MODELS/LOGS/")
print("=== HOÀN THÀNH ===")