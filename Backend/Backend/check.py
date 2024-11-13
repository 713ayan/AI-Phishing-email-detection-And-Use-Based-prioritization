import joblib

# Load the model
model = joblib.load("tfidf_vectorizer.pkl")

# Save it again (this ensures it's compatible with the current environment)
joblib.dump(model, "tfidf_vectorizer_new.pkl")
