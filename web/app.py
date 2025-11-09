from flask import Flask, render_template, request
from detector import check_url
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    if request.method=="POST":
        url = request.form.get("url")
        threshold = float(request.form.get("threshold",0.5))
        # save_screenshot=False để không lưu file (nhẹ máy) — vẫn trả về base64 to display
        result = check_url(url, threshold=threshold)
    return render_template("index.html", result=result)

if __name__=="__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
