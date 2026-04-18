from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv("spam_with_email_ids.csv")

TEXT_COL = "Message"
LABEL_COL = "Category"

X = df[TEXT_COL].astype(str)
y = df[LABEL_COL].map(lambda x: 1 if str(x).lower() == "spam" else 0)

# Train model
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# Create email lookup dictionary from dataset
email_lookup = {}
if 'email_id' in df.columns:
    for idx, row in df.iterrows():
        email = str(row['email_id']).strip().lower()
        category = str(row['Category']).strip().upper()
        if email and email != 'nan':
            email_lookup[email] = 'SPAM' if category == 'SPAM' else 'HAM'

print(f"Loaded {len(email_lookup)} emails from dataset")


@app.route("/")
def home():
    return "Backend is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]

    return jsonify({
        "prediction": "SPAM" if pred == 1 else "HAM"
    })


@app.route("/email-check", methods=["POST"])
def email_check():
    data = request.get_json()
    email = str(data.get("email", "")).strip().lower()

    if not email or "@" not in email:
        return jsonify({
            "prediction": "UNKNOWN",
            "message": "Please enter a valid email address."
        }), 400

    # Lookup email in dataset
    if email in email_lookup:
        category = email_lookup[email]
        message = f"This email address is marked as {category} in the dataset." if category == "SPAM" else "This email address is verified as legitimate in the dataset."
        return jsonify({
            "prediction": category,
            "message": message
        })
    else:
        return jsonify({
            "prediction": "UNKNOWN",
            "message": "This email address is not found in the dataset."
        })




if __name__ == "__main__":
    app.run(debug=True)
