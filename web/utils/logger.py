import json, os, time

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs.json")

def log_result(entry):
    try:
        logs = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)

        logs.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "url": entry["url"],
            "label": entry["label"],
            "score": entry["score"]
        })

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    except:
        pass
