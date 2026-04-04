from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# ----------------------------
# DATABASE SETUP
# ----------------------------

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///complaints.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ----------------------------
# DATABASE MODEL
# ----------------------------

class Complaint(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    text     = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(50), nullable=True)

    def to_dict(self):
        return {"id": self.id, "text": self.text, "category": self.category}

# ----------------------------
# ML MODEL — TRAINING DATA
# ----------------------------

train_texts = [
    # Road English
    "road is broken", "pothole on the street", "road needs repair",
    "street light not working", "road damaged after rain", "highway has big holes",
    "footpath is broken", "road is very bad condition", "speed breaker is damaged",
    "divider on road is broken", "road cave in near market", "manhole cover is missing on road",

    # Road Hinglish
    "sadak toot gayi hai", "sadak pe gadda ho gaya hai", "road bahut kharab hai",
    "sadak ki repair nahi hui", "street light band hai", "sadak pe bada gaddha hai",
    "footpath toot gaya", "highway pe bahut holes hai",

    # Garbage English
    "garbage not collected", "trash is overflowing", "waste lying on street",
    "dustbin is full", "garbage smell is very bad", "no garbage pickup since days",
    "waste not being removed", "garbage truck has not come", "litter all over the place",
    "open garbage dump near house", "stray dogs near garbage pile", "garbage blocking the road",

    # Garbage Hinglish
    "kachra nahi utha", "kachra bahut zyada jama ho gaya hai", "dustbin full ho gaya hai",
    "safai nahi ho rahi", "kachra gadi nahi aayi", "kachra sadak pe pada hai",
    "safai wala nahi aaya", "ganda kachra jama ho gaya",

    # Water English
    "no water supply", "water pipe is leaking", "dirty water coming from tap",
    "water shortage in area", "water tank is empty", "sewage water on road",
    "water pressure is very low", "tap water is yellow in colour", "borewell is not working",
    "water supply cut for 2 days", "contaminated water from pipeline", "overhead tank is leaking",

    # Water Hinglish
    "paani nahi aa raha", "nal se paani band hai", "paani ki pipe toot gayi",
    "ganda paani aa raha hai", "paani ka pressure bahut kam hai", "paani tank khali hai",
    "2 din se paani nahi aaya", "naali ka paani sadak pe aa gaya",
]

train_labels = [
    # Road English (12)
    "road","road","road","road","road","road",
    "road","road","road","road","road","road",

    # Road Hinglish (8)
    "road","road","road","road","road","road","road","road",

    # Garbage English (12)
    "garbage","garbage","garbage","garbage","garbage","garbage",
    "garbage","garbage","garbage","garbage","garbage","garbage",

    # Garbage Hinglish (8)
    "garbage","garbage","garbage","garbage",
    "garbage","garbage","garbage","garbage",

    # Water English (12)
    "water","water","water","water","water","water",
    "water","water","water","water","water","water",

    # Water Hinglish (8)
    "water","water","water","water",
    "water","water","water","water",
]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
model = LogisticRegression()
model.fit(X_train, train_labels)

def predict_category(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

# ----------------------------
# ROUTES
# ----------------------------

@app.route("/")
def home():
    return send_file("index.html") 

@app.route("/submit-complaint", methods=["POST"])
def submit_complaint():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "text field required"}), 400

    category = predict_category(text)

    new_complaint = Complaint(text=text, category=category)
    db.session.add(new_complaint)
    db.session.commit()

    return jsonify({
        "message": "Complaint received",
        "category": category
    }), 200

@app.route("/complaints", methods=["GET"])
def get_complaints():
    all_complaints = Complaint.query.all()
    result = [c.to_dict() for c in all_complaints]
    return jsonify(result), 200

# ----------------------------
# APP START
# ----------------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
