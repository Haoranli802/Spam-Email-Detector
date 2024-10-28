import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess the email text
def preprocess_text(text):
    text = text.lower()                       # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)      # Remove punctuation and numbers
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Step 1: Load and preprocess the training dataset
train_df = pd.read_csv('emails.csv')  # Replace with the actual path to the training CSV
train_df['text'] = train_df['text'].apply(preprocess_text)

# Step 2: Vectorize the training data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Limit to top 1000 features for simplicity
X_train = vectorizer.fit_transform(train_df['text']).toarray()
y_train = train_df['spam']

# Step 3: Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Load and preprocess the evaluation dataset
eval_df = pd.read_csv('spam_ham_dataset.csv')  # Replace with the actual path to the evaluation CSV
eval_df['text'] = eval_df['text'].apply(preprocess_text)

# Vectorize the evaluation data using the same TF-IDF vectorizer
X_eval = vectorizer.transform(eval_df['text']).toarray()
y_eval = eval_df['label_num']

# Step 5: Evaluate the model on the evaluation dataset
y_pred_eval = clf.predict(X_eval)

# Display the results
print("Evaluation Accuracy:", accuracy_score(y_eval, y_pred_eval))
print("Evaluation Classification Report:\n", classification_report(y_eval, y_pred_eval))
print("Evaluation Confusion Matrix:\n", confusion_matrix(y_eval, y_pred_eval))