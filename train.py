import json, nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt'); nltk.download('wordnet')

with open('data/intents.json') as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()
patterns, labels = [], []
for intent in data['intents']:
    for p in intent['patterns']:
        patterns.append(p.lower())
        labels.append(intent['tag'])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Model used only to REPORT accuracy on held-out data
eval_model = LogisticRegression()
eval_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report
preds = eval_model.predict(X_test)
print(accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Final model for deployment: trained on ALL patterns so nothing gets left out
model = LogisticRegression()
model.fit(X, y)

import joblib
joblib.dump(model, 'model/chatbot_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')