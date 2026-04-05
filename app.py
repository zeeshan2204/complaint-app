import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

complaints_list = []

# ----------------------------
# TRAINING DATA — 280 sentences
# ----------------------------

train_texts = [

    # ---- ROAD (70) ----
    "road is broken", "pothole on the street", "road needs repair",
    "street light not working", "road damaged after rain", "highway has big holes",
    "footpath is broken", "road is in very bad condition", "speed breaker is damaged",
    "divider on road is broken", "road cave in near market", "manhole cover is missing",
    "road is flooded", "traffic signal not working", "road blocked by construction",
    "bridge is damaged", "no streetlight on highway", "road marking is not visible",
    "flyover has cracks", "road is slippery", "footpath encroached by shops",
    "road divider broken near school", "pothole caused accident", "road needs white lining",
    "road near hospital is damaged", "speed breaker too high", "road sign missing",
    "broken road near bus stop", "road not repaired since months", "waterlogging on road",
    "sadak toot gayi hai", "sadak pe gadda ho gaya hai", "road bahut kharab hai",
    "sadak ki repair nahi hui", "street light band hai", "sadak pe bada gaddha hai",
    "footpath toot gaya", "highway pe bahut holes hai", "sadak pe paani jama hai",
    "signal kaam nahi kar raha", "sadak pe andhere mein koi light nahi",
    "sadak ka repair kab hoga", "road ke khadde se accident ho gaya",
    "sadak bilkul kharab hai baarish mein", "nali ka dhakkan nahi hai sadak pe",
    "sadak pe construction se block hai", "pulia toot gayi hai",
    "road divider toot gaya hai", "sadak pe koi sign nahi hai",
    "speed breaker bahut bada hai", "sadak pe marking nahi hai",
    "flyover mein darar aa gayi", "sadak pe fisal gaya main",
    "road par dukaan waalon ne kabza kar liya", "months se sadak nahi bani",
    "school ke paas road kharab hai", "hospital ke paas sadak toot gayi",
    "bus stop ke paas gadda hai", "sadak pe raat ko andhera hai",
    "highway pe koi light nahi hai", "sadak pe traffic jam hai construction se",
    "puri colony mein road repair nahi hui", "sadak ka kaam adha chhod diya",
    "naya road banaya tha woh bhi toot gaya", "sadak pe baarish mein dikh nahi raha",
    "sadak ka slope galat hai paani ruk jaata hai", "road roller aaya tha kaam adha hai",
    "sadak pe bade bade pathar pade hain", "sadak block hai kisi ne saman rakh diya",
    "road pe tree gir gaya rasta band hai",

    # ---- GARBAGE (70) ----
    "garbage not collected", "trash is overflowing", "waste lying on street",
    "dustbin is full", "garbage smell is very bad", "no garbage pickup since days",
    "waste not being removed", "garbage truck has not come", "litter all over the place",
    "open garbage dump near house", "stray dogs near garbage pile", "garbage blocking the road",
    "garbage burning in open", "plastic waste on footpath", "dead animal on road not removed",
    "garbage near school is health hazard", "waste dumped in park", "no dustbin in area",
    "flies and mosquitoes due to garbage", "garbage dump near water tank",
    "construction waste on road", "hospital waste dumped illegally",
    "market area very dirty", "garbage collector not coming regularly",
    "overflowing drain due to garbage", "waste dumped near playground",
    "garbage in front of house not taken", "trash near temple is disrespectful",
    "public toilet very dirty", "municipal workers not cleaning",
    "kachra nahi utha", "kachra bahut zyada jama ho gaya hai", "dustbin full ho gaya hai",
    "safai nahi ho rahi", "kachra gadi nahi aayi", "kachra sadak pe pada hai",
    "safai wala nahi aaya", "ganda kachra jama ho gaya", "badboo aa rahi hai kachra se",
    "nagar palika kachra nahi utha rahi", "khula kachraghara ghar ke paas hai",
    "kachra jal raha hai khule mein", "park mein kachra pada hai",
    "school ke paas gandagi bahut hai", "machar ho rahe hain kachra se",
    "kachra naale mein phat gaya hai", "bazar mein bahut gandagi hai",
    "hospital ka kachra seedha sadak pe", "construction ka mala pada hai",
    "dustbin hi nahi hai hamare area mein", "kachra uthane wala regularly nahi aata",
    "mandir ke paas kachra bahut ganda lagta hai", "khel ke maidan mein kachra hai",
    "ghar ke saamne kachra pada hai", "kachra tank ke paas pada hai",
    "mara hua janwar sadak pe pada hai", "plastic waste footpath pe bikra hai",
    "public toilet bahut ganda hai", "nagar palika ke workers kaam nahi karte",
    "drain kachra se bhar gayi hai", "safai ka schedule follow nahi ho raha",
    "raat ko kachra jalaya ja raha hai", "kachra gaadi ka driver aa ke chala jaata hai",
    "colony mein koi dustbin nahi rakha", "kachra uthane ki koi timing nahi hai",
    "festival ke baad kachra nahi utha", "baarish mein kachra sadak pe beh gaya",
    "nagar palika ka number lagata nahi", "safai abhiyaan sirf dikhawa hai",
    "poora mohalla ganda pad gaya hai",

    # ---- WATER (70) ----
    "no water supply", "water pipe is leaking", "dirty water coming from tap",
    "water shortage in area", "water tank is empty", "sewage water on road",
    "water pressure is very low", "tap water is yellow in colour", "borewell is not working",
    "water supply cut for 2 days", "contaminated water from pipeline", "overhead tank is leaking",
    "water motor is not working", "no water for 3 days", "muddy water from tap",
    "water meter is broken", "pipeline burst on main road", "water wastage due to leaking pipe",
    "water not reaching upper floors", "sewage mixing with drinking water",
    "water supply only for 30 minutes", "water smells bad from tap",
    "kids getting sick due to dirty water", "water tanker not coming",
    "illegal water connection by neighbour", "water board not responding",
    "borewell water has bad smell", "water pump broken in colony",
    "water supply irregular since week", "drainage overflowing into homes",
    "paani nahi aa raha", "nal se paani band hai", "paani ki pipe toot gayi",
    "ganda paani aa raha hai", "paani ka pressure bahut kam hai", "paani tank khali hai",
    "2 din se paani nahi aaya", "naali ka paani sadak pe aa gaya",
    "paani ka rang peela hai", "boring kaam nahi kar rahi",
    "paani mein badboo aa rahi hai", "3 din se paani nahi hai",
    "upar wali manzil pe paani nahi pahunchta", "paani ka meter toot gaya",
    "pipe se paani waste ho raha hai", "paani motor band ho gayi",
    "tanker bhi nahi aa raha", "main pipeline phat gayi sadak pe",
    "paani peene layak nahi hai", "gutter ka paani peene wale pipe mein",
    "paani sirf aadhe ghante aata hai", "bacche beemar ho rahe hain gande paani se",
    "padosi ne illegal connection liya", "jal board ka koi jawab nahi",
    "colony mein pump kharab ho gaya", "ek hafte se paani abhi bhi irregular hai",
    "boring ka paani bahut ganda hai", "drainage ghar mein ghus rahi hai",
    "paani supply ek hafte se theek nahi", "paani ki bahut kami hai area mein",
    "naya pipeline dala tha woh bhi toot gaya", "subah paani aata hai raat ko nahi",
    "paani ka bill aaya par paani hi nahi", "jal board office ka koi response nahi",
    "ghar mein paani store karna pad raha hai", "paani ka color ajeeb hai",
    "haath dhone layak bhi paani nahi", "ro filter bhi kaam nahi kar raha",
    "society tank se paani nahi aa raha", "submersible pump fail ho gaya",

    # ---- ELECTRICITY (70) ----
    "no electricity since morning", "power cut for 3 hours", "electricity bill is too high",
    "street light not working at night", "transformer is burnt", "power outage in colony",
    "electric wire hanging low on road", "electricity fluctuation damaging appliances",
    "meter reading is wrong", "no power supply since 2 days",
    "electric pole is leaning and dangerous", "fuse blown in our area",
    "electricity comes and goes every hour", "power cut during exam time",
    "electric sparks from wire near house", "no electricity in the whole street",
    "electricity department not responding", "prepaid meter not working",
    "electric shock risk near transformer", "load shedding every day",
    "inverter not charging due to low voltage", "power supply very weak",
    "electric wire touching water pipe", "street light on during day wasting electricity",
    "electricity restored but very low voltage", "new connection not given since months",
    "electricity bill came without using power", "meter installed wrongly",
    "wire broken during storm not repaired", "electric pole fell on road",
    "bijli nahi aa rahi", "light chali gayi subah se",
    "bijli ka bill bahut zyada aaya hai", "transformer jal gaya hai",
    "bijli baar baar aa jaati hai", "wire sadak pe latkaa hua hai",
    "bijli vibhag ka koi jawab nahi", "meter galat reading de raha hai",
    "2 din se bijli nahi hai", "bijli ke khambe jhuk gaya hai",
    "hamare area mein andhera hai raat ko", "bijli aati hai toh voltage kam hai",
    "ghar ke appliances kharab ho rahe hain fluctuation se",
    "bijli ka meter kaam nahi kar raha", "exam time mein bijli chali jaati hai",
    "wire se chingari nikal rahi hai", "poori gali mein bijli nahi hai",
    "substation pe koi dhyan nahi deta", "load shedding roz ho rahi hai",
    "inverter charge nahi ho raha low voltage se", "bijli vibhag ne naya connection nahi diya",
    "bina bijli use kiye bill aa gaya", "bijli pole sadak pe gir gaya",
    "taar toota pada hai koi nahi uthata", "wire paani ki pipe ko chhu raha hai",
    "raat ko street light band rehti hai", "din mein street light jalti rehti hai",
    "transformer ke paas khada rehna dangerous hai", "low voltage se AC kaam nahi kar raha",
    "bijli ki shikayat darj nahi hoti", "helpline number lagta nahi hai",
    "hamare mohalle mein bijli ki problem roz hai", "new meter lagwana hai koi nahi aata",
    "bijli gul hone ki koi soochna nahi milti", "meter box se aawaz aa rahi hai",
    "bijli ka khamba sadak ke beech mein hai", "bijli wale aate hain kaam nahi karte",
    "emergency mein bijli gul ho gayi", "hospital ke paas bijli nahi hai",
]

train_labels = ["road"] * 70 + ["garbage"] * 70 + ["water"] * 70 + ["electricity"] * 70

# Train/Test Split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

vectorizer  = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_s)
X_test_vec  = vectorizer.transform(X_test_s)

model = LogisticRegression()
model.fit(X_train_vec, y_train_s)

y_pred   = model.predict(X_test_vec)
accuracy = accuracy_score(y_test_s, y_pred)

print("=" * 50)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)

def predict_category(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

# ----------------------------
# ROUTES
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    return send_file(os.path.join(BASE_DIR, "index.html"))

@app.route("/admin")
def admin():
    return send_file(os.path.join(BASE_DIR, "admin.html"))

@app.route("/model-stats")
def model_stats():
    return jsonify({
        "accuracy"  : round(accuracy * 100, 2),
        "total_samples": len(train_texts),
        "train_size": len(X_train_s),
        "test_size" : len(X_test_s),
        "categories": ["road", "garbage", "water", "electricity"],
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