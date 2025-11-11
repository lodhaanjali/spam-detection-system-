from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# main1.py
# Simple Flask app: Spam Detection using Machine Learning
# Run: python main1.py
# Install: pip install flask scikit-learn

app = Flask(__name__)

# Example training data
TRAIN_TEXTS = [
    "Congrats! You've won a free ticket. Call now!",
    "Free entry in a weekly competition to win tickets",
    "URGENT! Your account has been compromised. Reply with password",
    "Win cash now!!! Click here to claim your prize",
    "Hi, are we still meeting for lunch today?",
    "Don't forget the report due tomorrow",
    "Can you send the files I asked for?",
    "Let's catch up later. Hope you're well!"
]
TRAIN_LABELS = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = spam, 0 = ham

# Build and train model
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train = vectorizer.fit_transform(TRAIN_TEXTS)
clf = LogisticRegression()
clf.fit(X_train, TRAIN_LABELS)

def predict_text(text):
    """Predict if text is spam or ham."""
    if not text or not text.strip():
        return {"error": "empty text provided"}
    X = vectorizer.transform([text])
    prob = clf.predict_proba(X)[0][1]  # probability of spam
    pred = int(prob >= 0.5)
    return {
        "prediction": "spam" if pred == 1 else "ham",
        "spam_probability": float(prob)
    }

# HTML frontend
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Spam Detector</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    textarea { width: 100%; height: 120px; }
    .result { margin-top: 20px; padding: 10px; border-radius: 4px; }
    .spam { background: #ffd6d6; }
    .ham  { background: #d6ffd9; }
    .error { background: #fff3cd; }
  </style>
</head>
<body>
  <h1>Spam Detection Demo</h1>
  <form method="post" action="/predict">
    <label for="text">Enter message to analyze:</label><br>
    <textarea name="text" id="text" required></textarea><br><br>
    <button type="submit">Check</button>
  </form>
  {result_block}
  <hr>
  <small>API: POST /api/predict with JSON {{ "text": "..." }} returns JSON prediction.</small>
</body>
</html>
"""

def render_result_block(result):
    """Render result in HTML safely."""
    if not result:
        return ""
    if "error" in result:
        return f"<div class='result error'><strong>Error:</strong> {result['error']}</div>"
    cls = "spam" if result["prediction"] == "spam" else "ham"
    return f'<div class="result {cls}"><strong>Prediction:</strong> {result["prediction"]} ' \
           f'(<em>spam probability: {result["spam_probability"]:.2f}</em>)</div>'

@app.route("/", methods=["GET"])
def index():
    # ✅ FIXED: use replace() instead of format()
    return HTML_TEMPLATE.replace("{result_block}", "")

@app.route("/predict", methods=["POST"])
def predict_form():
    text = request.form.get("text", "")
    result = predict_text(text)
    # ✅ FIXED: use replace() instead of format()
    return HTML_TEMPLATE.replace("{result_block}", render_result_block(result))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    result = predict_text(text)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
