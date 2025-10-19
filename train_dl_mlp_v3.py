# ======================================================
#  train_dl_mlp_v3.py
#  Huấn luyện mô hình Deep Learning (MLP)
#  So sánh với mô hình ML từ tuần 2
# ======================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time, os

# === 1. Tải dữ liệu ML đã xử lý ===
data = np.load("MODELS/ml_training_data.npz", allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

print(f"[+] Dữ liệu huấn luyện: {X_train.shape}, kiểm thử: {X_test.shape}")

# === 2. Cấu trúc mạng MLP ===
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # nhị phân
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === 3. Huấn luyện ===
os.makedirs("MODELS", exist_ok=True)
checkpoint = ModelCheckpoint("MODELS/dl_mlp_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

start = time.time()
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=1024,
    validation_split=0.2,
    callbacks=[checkpoint, early_stop],
    verbose=2
)
end = time.time()

print(f"⏱ Thời gian huấn luyện: {(end-start)/60:.2f} phút")

# === 4. Đánh giá mô hình ===
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\n--- Deep Learning (MLP) Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# === 5. Lưu lại kết quả ===
model.save("MODELS/dl_mlp_model.h5")
print("✅ Saved Deep Learning model to MODELS/dl_mlp_model.h5")

# === 6. Vẽ biểu đồ huấn luyện ===
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Training Progress (MLP)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("MODELS/mlp_accuracy_curve.png", dpi=200)
plt.show()
