# ==========================================
# Danh sách từ khóa cảnh báo phishing (đa ngôn ngữ)
# Giảm false-positive – đặc biệt tránh bắt nhầm Cloudflare, Google, Canva
# ==========================================

PHISHING_KEYWORDS = [
    # English – CHỈ GIỮ TỪ KHÓA LIÊN QUAN PHISHING THẬT
    "account suspended",
    "verify your identity",
    "verify your account",
    "confirm your information",
    "password reset required",
    "billing issue detected",
    "unusual login attempt",
    "restricted account",
    "login to restore access",
    "payment information required",

    # ▼ Cloudflare FIX – bỏ các từ khóa gây bắt nhầm
    # ❌ Bỏ: "verifying your connection" vì nó là WAF bình thường
    # ❌ Bỏ: "your request has been blocked" (chặn IP, không phải phishing)
    # Giữ lại đúng keyword phishing thật:
    "reported for phishing",
    "suspected phishing",
    "this website has been reported",

    # Indonesia
    "akun anda diblokir",
    "perlu verifikasi",
    "konfirmasi data",

    # Portuguese (Brazil)
    "validar conta",
    "verificação obrigatória",

    # Japanese
    "あなたのアカウント",
    "セキュリティ警告",
]

import requests


def check_warnings(url, timeout=10):
    """
    Kiểm tra xem nội dung HTML có chứa cảnh báo phishing không.
    DETECTOR.PY đang import hàm này → KHÔNG THAY ĐỔI CẤU TRÚC.
    """
    from .warnings import PHISHING_KEYWORDS

    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        html = r.text.lower()

        for kw in PHISHING_KEYWORDS:
            if kw.lower() in html:
                return True

        return False

    except:
        # Nếu lỗi mạng → KHÔNG coi là phishing (fix false positive)
        return False
