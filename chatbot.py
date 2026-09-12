import json
import random
import joblib

# Load intents data
with open('data/intents.json', 'r') as f:
    data = json.load(f)

# Load trained vectorizer and model
vectorizer = joblib.load('model/vectorizer.pkl')
model = joblib.load('model/chatbot_model.pkl')

response_map = {intent['tag']: intent['responses'] for intent in data['intents']}

def get_response(user_input):
    X_input = vectorizer.transform([user_input.lower()])
    tag = model.predict(X_input)[0]
    confidence = model.predict_proba(X_input).max()
    if confidence < 0.25:
        return "I'm not sure I understood — could you rephrase?"
    return random.choice(response_map[tag])