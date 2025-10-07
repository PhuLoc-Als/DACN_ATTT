from datasets import load_dataset
import pandas as pd

print("🔍 Đang tải dataset DeepURLBench (urls_with_dns)...")

# ✅ Lưu cache vào ổ ngắn để tránh lỗi đường dẫn dài trên Windows
dataset = load_dataset(
    "davanstrien/DeepURLBench",
    "urls_with_dns",
    cache_dir="D:/HF_CACHE"  # ⚡ dùng cache mới bạn vừa setx
)

# In thông tin dataset
print(dataset)

# Lấy tập train (dữ liệu chính)
df = dataset['train'].to_pandas()

# Lấy mẫu nhỏ 50.000 hàng để dùng cho Deep Learning
sample_df = df.sample(50000, random_state=42)
sample_df.to_csv("DeepURLBench_DL_sample.csv", index=False)

print("✅ Đã lưu dataset mẫu DeepURLBench_DL_sample.csv trong thư mục DATASET.")
