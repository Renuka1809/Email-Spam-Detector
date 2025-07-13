from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import pandas as pd

# Load the dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['Category', 'Message']]  # Keep only the needed columns
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})  # Convert to binary

# Split into features and labels
X = data['Message']
y = data['Category']

# Vectorize the text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
