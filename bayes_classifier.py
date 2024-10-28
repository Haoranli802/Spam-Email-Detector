import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import re

def preprocess_text(text):
    """
    Preprocesses the email text by:
    1. Converting to lowercase
    2. Removing special characters
    3. Removing multiple spaces
    4. Basic email-specific cleaning
    """
    # Convert to string in case we have any non-string types
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove email headers (if present)
    text = re.sub(r'^subject:', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    # Remove special characters but keep letters, numbers and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_spam_detector(training_data):
    """
    Creates and trains the spam detector using the training dataset.
    """
    # Create preprocessing pipeline
    preprocessor = FunctionTransformer(lambda x: [preprocess_text(text) for text in x])
    
    # Create TF-IDF vectorizer with improved parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.7,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('vectorizer', vectorizer),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    # Train the model on the entire training dataset
    X_train = training_data['text']
    y_train = training_data['spam']
    pipeline.fit(X_train, y_train)
    
    return pipeline

def evaluate_model(pipeline, eval_data):
    """
    Evaluates the model performance using the evaluation dataset.
    """
    X_eval = eval_data['text']
    y_eval = eval_data['label_num']  # Using label_num column which has 0/1 values
    
    # Make predictions
    y_pred = pipeline.predict(X_eval)
    y_pred_proba = pipeline.predict_proba(X_eval)
    
    # Calculate and print metrics
    print("Model Performance Evaluation on External Dataset:")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(y_eval, y_pred))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_eval, y_pred)
    print(conf_matrix)
    
    print("\nDetailed Metrics:")
    print(f"Accuracy: {accuracy_score(y_eval, y_pred):.4f}")
    
    return {
        'accuracy': accuracy_score(y_eval, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def predict_email(pipeline, email_text):
    """
    Predicts whether a new email is spam or not.
    """
    prediction = pipeline.predict([email_text])[0]
    probability = pipeline.predict_proba([email_text])[0]
    
    confidence = probability[1] if prediction == 1 else probability[0]
    
    result = {
        'is_spam': bool(prediction),
        'confidence': confidence,
        'probability_spam': probability[1],
        'probability_ham': probability[0]
    }
    
    return result

if __name__ == "__main__":
    # Load both datasets
    print("Loading datasets...")
    training_data = pd.read_csv('emails.csv')
    evaluation_data = pd.read_csv('spam_ham_dataset.csv')
    
    # Print dataset shapes
    print(f"\nTraining dataset shape: {training_data.shape}")
    print(f"Evaluation dataset shape: {evaluation_data.shape}")
    
    # Create and train the model using the training dataset
    print("\nTraining model on emails.csv...")
    spam_detector = create_spam_detector(training_data)
    
    # Evaluate the model using the evaluation dataset
    print("\nEvaluating model on spam_ham_dataset.csv...")
    evaluation_results = evaluate_model(spam_detector, evaluation_data)
    
    # Example of predicting a new email
    new_email = """Subject: Docker's Impact on Development From day one, Docker revolutionized software development — transforming the landscape with containers and simplified, cross-platform workflows. Since then, we've become the #1 platform for software developers worldwide.

Read about how Docker continues to pave the way for software development in this white paper by Steven J. Vaughan-Nichol, Docker: The software development revolution continued. Inside, you'll uncover:
An in-depth dive into Docker's comprehensive ecosystem
A glimpse into a developer's day empowered by Docker
How Docker's dev tools accelerate innovation by enhancing flexibility, security, and rapid software delivery
Don't miss out on discovering how you can unlock innovation by leveraging the complete potential of Docker's comprehensive container development stack.
Get the most out of Docker
Check out our subscription offerings or contact our sales team to start accelerating innovation with Docker today."""
    
    print("\nExample Prediction:")
    result = predict_email(spam_detector, new_email)
    print(f"Is spam: {result['is_spam']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Spam probability: {result['probability_spam']:.4f}")
    print(f"Ham probability: {result['probability_ham']:.4f}")