from flask import Flask, render_template, request, jsonify
from detector import check_url, cached_check_url
from flask_cors import CORS
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form.get("url")
        threshold = float(request.form.get("threshold", 0.5))
        result = check_url(url, threshold=threshold)
    return render_template("index.html", result=result)


# ==== API JSON CHO EXTENSION ====
@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json() or {}
    url = data.get("url", "")
    threshold = float(data.get("threshold", 0.5))

    result = cached_check_url(url, threshold)
    return jsonify(result)

@app.route("/detail", methods=["GET", "POST"])
def detail_page():
    if request.method == "POST":
        url = request.form.get("url")
        result = cached_check_url(url, threshold=0.5)
        return render_template("detail.html", result=result, url=url)

    # GET request
    url = request.args.get("url", "")
    result = cached_check_url(url, threshold=0.5)
    return render_template("detail.html", result=result, url=url)



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

