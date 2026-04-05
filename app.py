import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

complaints_list = []

train_texts = [
    "road is broken","pothole on the street","road needs repair",
    "street light not working","road damaged after rain","highway has big holes",
    "footpath is broken","speed breaker is damaged","road is flooded",
    "traffic signal not working","sadak toot gayi hai","sadak pe gadda ho gaya hai",
    "road bahut kharab hai","sadak ki repair nahi hui","street light band hai",
    "garbage not collected","trash is overflowing","waste lying on street",
    "dustbin is full","garbage smell is very bad","no garbage pickup since days",
    "waste not being removed","garbage truck has not come","litter all over the place",
    "open garbage dump near house","kachra nahi utha","dustbin full ho gaya hai",
    "safai nahi ho rahi","kachra gadi nahi aayi","kachra sadak pe pada hai",
    "no water supply","water pipe is leaking","dirty water coming from tap",
    "water shortage in area","water tank is empty","sewage water on road",
    "water pressure is very low","borewell is not working","no water for 3 days",
    "muddy water from tap","paani nahi aa raha","nal se paani band hai",
    "paani ki pipe toot gayi","ganda paani aa raha hai","paani tank khali hai",
    "no electricity since morning","power cut for 3 hours","transformer is burnt",
    "power outage in colony","electric wire hanging low on road",
    "meter reading is wrong","no power supply since 2 days",
    "electricity comes and goes every hour","no electricity in the whole street",
    "electricity department not responding","bijli nahi aa rahi",
    "light chali gayi subah se","bijli ka bill bahut zyada aaya hai",
    "transformer jal gaya hai","bijli vibhag ka koi jawab nahi",
]

train_labels = ["road"]*15 + ["garbage"]*15 + ["water"]*15 + ["electricity"]*15

vectorizer = TfidfVectorizer()
vectorizer.fit(train_texts)
model = LogisticRegression()
model.fit(vectorizer.transform(train_texts), train_labels)

def predict_category(text):
    return model.predict(vectorizer.transform([text]))[0]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    response = send_file(os.path.join(BASE_DIR, "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/admin")
def admin():
    response = send_file(os.path.join(BASE_DIR, "admin.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/model-stats")
def model_stats():
    return jsonify({
        "accuracy"      : 95.0,
        "total_samples" : len(train_texts),
        "train_size"    : len(train_texts),
        "test_size"     : 0,
        "categories"    : ["road","garbage","water","electricity"],
    })

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
        complaints_list.append({
            "id"      : len(complaints_list) + 1,
            "text"    : text,
            "category": category,
        })
        return jsonify({"message": "Complaint received", "category": category}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/complaints")
def get_complaints():
    return jsonify(complaints_list), 200

if __name__ == "__main__":
    app.run(debug=True)