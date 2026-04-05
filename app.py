import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# Simple in-memory list — database ki jagah
complaints_list = []

train_texts = [
    "road is broken", "pothole on the street", "road needs repair",
    "street light not working", "road damaged after rain", "highway has big holes",
    "footpath is broken", "road is very bad condition", "speed breaker is damaged",
    "divider on road is broken", "road cave in near market", "manhole cover is missing on road",
    "sadak toot gayi hai", "sadak pe gadda ho gaya hai", "road bahut kharab hai",
    "sadak ki repair nahi hui", "street light band hai", "sadak pe bada gaddha hai",
    "footpath toot gaya", "highway pe bahut holes hai",
    "garbage not collected", "trash is overflowing", "waste lying on street",
    "dustbin is full", "garbage smell is very bad", "no garbage pickup since days",
    "waste not being removed", "garbage truck has not come", "litter all over the place",
    "open garbage dump near house", "stray dogs near garbage pile", "garbage blocking the road",
    "kachra nahi utha", "kachra bahut zyada jama ho gaya hai", "dustbin full ho gaya hai",
    "safai nahi ho rahi", "kachra gadi nahi aayi", "kachra sadak pe pada hai",
    "safai wala nahi aaya", "ganda kachra jama ho gaya",
    "no water supply", "water pipe is leaking", "dirty water coming from tap",
    "water shortage in area", "water tank is empty", "sewage water on road",
    "water pressure is very low", "tap water is yellow in colour", "borewell is not working",
    "water supply cut for 2 days", "contaminated water from pipeline", "overhead tank is leaking",
    "paani nahi aa raha", "nal se paani band hai", "paani ki pipe toot gayi",
    "ganda paani aa raha hai", "paani ka pressure bahut kam hai", "paani tank khali hai",
    "2 din se paani nahi aaya", "naali ka paani sadak pe aa gaya",
]

train_labels = ["road"] * 20 + ["garbage"] * 20 + ["water"] * 20

vectorizer = TfidfVectorizer()
X_train    = vectorizer.fit_transform(train_texts)
model      = LogisticRegression()
model.fit(X_train, train_labels)

def predict_category(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    return send_file(os.path.join(BASE_DIR, "index.html"))

@app.route("/submit-complaint", methods=["POST"])
def submit_complaint():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON received"}), 400
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "text field required"}), 400
        category = predict_category(text)
        complaints_list.append({"id": len(complaints_list) + 1, "text": text, "category": category})
        return jsonify({"message": "Complaint received", "category": category}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/complaints", methods=["GET"])
def get_complaints():
    return jsonify(complaints_list), 200

if __name__ == "__main__":
    app.run(debug=True)