import json
from datetime import datetime

def log_interaction(tweet, sentiment, toxicity, category, reply):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "tweet": tweet,
        "sentiment": sentiment,
        "toxicity": toxicity,
        "category": category,
        "reply": reply
    }
    with open("logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")