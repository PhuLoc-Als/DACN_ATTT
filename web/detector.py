import os, re, io, time, base64, requests
import numpy as np
from urllib.parse import urlparse
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# ==== Đường dẫn model ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "MODELS")

# ==== Load mô hình ====
def load_ml_model():
    path = os.path.join(MODELS_DIR, "ml_meta_model.pkl")
    if os.path.exists(path):
        print("[INFO] ML model loaded.")
        return joblib.load(path)
    else:
        print("[WARN] ML model not found.")
        return None

def load_dl_model():
    path = os.path.join(MODELS_DIR, "hybrid_phishiris_best.h5")
    if os.path.exists(path):
        print("[INFO] DL model loaded.")
        return load_model(path, compile=False)
    else:
        print("[WARN] DL model not found.")
        return None

ML_MODEL = load_ml_model()
DL_MODEL = load_dl_model()

# ==== Heuristic feature extractor ====
def extract_url_features(url):
    u = url.lower()
    parsed = urlparse(url)
    feats = {
        "url_len": len(u),
        "hostname_len": len(parsed.netloc),
        "path_len": len(parsed.path),
        "count_dot": u.count("."),
        "count_dash": u.count("-"),
        "count_at": u.count("@"),
        "count_question": u.count("?"),
        "count_percent": u.count("%"),
        "has_ip": 1 if re.search(r"\d+\.\d+\.\d+\.\d+", u) else 0,
        "entropy": round(-sum(p*np.log2(p) for p in [u.count(ch)/len(u) for ch in set(u)] if p>0),3),
        "susp_token": sum(tok in u for tok in ["login","secure","account","verify","bank","paypal","vpass"]),
        "digit_ratio": sum(ch.isdigit() for ch in u)/len(u)
    }
    return feats

def heuristic_score(feats):
    score = 0
    if feats["url_len"] > 75: score += 0.1
    if feats["count_dash"] > 2: score += 0.1
    if feats["has_ip"]: score += 0.2
    if feats["susp_token"] >= 1: score += 0.3
    if feats["entropy"] > 4.5: score += 0.1
    if "vpass" in feats and "jp" not in feats: score += 0.2
    return min(score, 1.0)

def features_to_vector(feats: dict):
    # khớp với LogisticRegression (4 đặc trưng)
    keys = ["url_len", "count_dot", "count_dash", "susp_token"]
    v = [float(feats.get(k, 0.0)) for k in keys]
    X = np.array(v, dtype=float).reshape(1, -1)
    return X

# ==== HTTP Status ====
def check_http_status(url, timeout=10):
    try:
        if not url.startswith("http"):
            url = "http://" + url
        start = time.time()
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return {"ok": True, "status": r.status_code, "time": round(time.time()-start, 2)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ==== Screenshot bằng Playwright (base64, không lưu file) ====
def get_screenshot_base64(url):
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_navigation_timeout(10000)
            page.goto(url, wait_until="domcontentloaded")
            buffer = page.screenshot(full_page=True)
            browser.close()
        return "data:image/png;base64," + base64.b64encode(buffer).decode()
    except Exception as e:
        print("[WARN] Screenshot error:", e)
        return None

# ==== Hàm chính kiểm tra URL ====
def check_url(url, threshold=0.5):
    feats = extract_url_features(url)
    heur = heuristic_score(feats)
    ml_score, dl_score = None, None

    # ML
    if ML_MODEL:
        try:
            X = features_to_vector(feats)
            ml_score = float(ML_MODEL.predict_proba(X)[:,1][0])
        except Exception as e:
            print("[WARN] ML predict error:", e)

    # Screenshot & DL
    img_b64 = get_screenshot_base64(url)
    if DL_MODEL and img_b64:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(img_b64.split(",")[1]))).convert("RGB").resize((128,128))
            Ximg = np.expand_dims(np.array(img)/255.0, axis=0)
            seq_dummy = np.zeros((1,100))  # mô hình DL cần 2 input
            pred = DL_MODEL.predict([Ximg, seq_dummy])
            dl_score = float(pred[0][0])
        except Exception as e:
            print("[WARN] DL predict error:", e)

    # Final score
    valid_scores = [s for s in [heur, ml_score, dl_score] if s is not None]
    final_score = np.mean(valid_scores) if valid_scores else heur
    label = "phish" if final_score >= threshold else "legit"
    status = check_http_status(url)

    return {
        "url": url,
        "label": label,
        "score": round(final_score, 3),
        "heuristic": round(heur, 3),
        "ml_score": round(ml_score, 3) if ml_score is not None else 0,
        "dl_score": round(dl_score, 3) if dl_score is not None else "N/A",
        "http": status,
        "screenshot": img_b64
    }
