import os, re, io, time, base64, requests
import numpy as np
from urllib.parse import urlparse
from PIL import Image
from utils.warnings import check_warnings
from tensorflow.keras.models import load_model
from functools import lru_cache
import joblib
import requests

# ==== Đường dẫn model ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR,"..","MODELS")

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

    # Domain hợp lệ → không tính "login" là suspicious token
    trusted_domains = ["github.com", "google.com", "microsoft.com", "facebook.com"]
    domain = parsed.netloc.lower()

    susp_token = sum(tok in u for tok in ["login","secure","account","verify","bank","paypal","vpass"])
    if any(td in domain for td in trusted_domains):
        susp_token = 0

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
        "entropy": round(-sum(p*np.log2(p) for p in [u.count(ch)/len(u) for ch in set(u)] if p>0), 3),
        "susp_token": susp_token,
        "digit_ratio": sum(ch.isdigit() for ch in u) / len(u)
    }
    return feats

def heuristic_score(feats):
    score = 0
    if feats["url_len"] > 75: score += 0.1
    if feats["count_dash"] > 2: score += 0.1
    if feats["has_ip"]: score += 0.2
    if feats["susp_token"] >= 1: score += 0.3
    if feats["entropy"] > 4.5: score += 0.1
    return min(score, 1.0)

def features_to_vector(feats: dict):
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
        return {"ok": True, "status": r.status_code, "time": round(time.time() - start, 2)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ==== Screenshot bằng Playwright ====
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

# ==== Giải link rút gọn ====
def unshorten(url, max_redirects=5):
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        session = requests.Session()
        session.max_redirects = max_redirects
        resp = session.head(url, allow_redirects=True, timeout=10)
        return resp.url
    except:
        return url


@lru_cache(maxsize=256)
def cached_check_url(url, threshold):
    return check_url(url, threshold)

# ==== Hàm chính kiểm tra URL ====
def check_url(url, threshold=0.5):

    # Giải link rút gọn
    real_url = unshorten(url)
    print("[INFO] Unshortened:", real_url)
    url = real_url

    # Heuristic features
    feats = extract_url_features(url)
    heur = heuristic_score(feats)
    ml_score, dl_score = None, None

    # ML Prediction
    if ML_MODEL:
        try:
            X = features_to_vector(feats)
            ml_score = float(ML_MODEL.predict_proba(X)[:,1][0])
        except Exception as e:
            print("[WARN] ML predict error:", e)

    # Screenshot + DL Prediction
    img_b64 = get_screenshot_base64(url)
    if DL_MODEL and img_b64:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(img_b64.split(",")[1]))).convert("RGB").resize((128,128))
            Ximg = np.expand_dims(np.array(img) / 255.0, axis=0)
            seq_dummy = np.zeros((1,100))
            pred = DL_MODEL.predict([Ximg, seq_dummy])
            dl_score = float(pred[0][0])
        except Exception as e:
            print("[WARN] DL predict error:", e)

    # ==== Final score từ ML + DL + heuristic ====
    valid_scores = [s for s in [heur, ml_score, dl_score] if s is not None]
    final_score = np.mean(valid_scores) if valid_scores else heur
    final_score = max(0, final_score - 0.08)
    # ==== Kiểm tra cảnh báo nội dung (HTML WARNING) ====
    html_warning = check_warnings(url)

    if html_warning:
        print("[INFO] HTML phishing warning detected!")
        final_score = max(final_score, 0.95)
        label = "phish"
    else:
        label = "phish" if final_score >= threshold else "legit"

    status = check_http_status(url)

# Nếu website không truy cập được → coi là phishing
    if not status.get("ok", True):
        print("[INFO] Website unreachable → mark as phishing.")

        from utils.logger import log_result
        log_result({
            "url": url,
            "label": "phish",
            "score": 0.67
        })

        return {
            "url": url,
            "label": "phish",
            "score": 0.67,
            "heuristic": round(heur, 3),
            "ml_score": ml_score or "N/A",
            "dl_score": dl_score or "N/A",
            "http": status,
            "screenshot": img_b64
        }


    from utils.logger import log_result
    log_result({
        "url": url,
        "label": label,
        "score": final_score
    })


    return {
        "url": url,
        "label": label,
        "score": round(final_score, 3),
        "heuristic": round(heur, 3),
        "ml_score": round(ml_score, 3) if ml_score is not None else "N/A",
        "dl_score": round(dl_score, 3) if dl_score is not None else "N/A",
        "http": status,
        "screenshot": img_b64
    }
