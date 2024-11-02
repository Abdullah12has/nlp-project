import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import logging
from transformers import pipeline  # Import pipeline for RoBERTa

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_sentiment_with_roberta(df, text_column):
    """
    Use RoBERTa for sentiment prediction.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): The column name that contains the text data.

    Returns:
        pd.DataFrame: DataFrame with predicted sentiments.
    """
    # Load sentiment-analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    
    # Predict sentiment for each text entry
    sentiments = sentiment_pipeline(df[text_column].tolist())
    
    # Extract the label and score for each prediction
    df['roberta_sentiment'] = [result['label'] for result in sentiments]
    df['roberta_score'] = [result['score'] for result in sentiments]
    
    return df

def train_sentiment_model(df, text_column, label_column):
    """Train a sentiment analysis model using Logistic Regression."""
    # Check if the specified columns exist in the DataFrame
    if text_column not in df.columns or label_column not in df.columns:
        logging.error(f"Columns {text_column} or {label_column} not found in DataFrame.")
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[text_column], df[label_column], test_size=0.3, random_state=42)
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
    model.fit(X_train_vec, y_train)

    # Make predictions
    y_pred = model.predict(X_test_vec)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:")
    logging.info("\n" + classification_report(y_test, y_pred))

# Example usage
# df = pd.read_csv('your_data_file.csv')  # Load your DataFrame here
# df = predict_sentiment_with_roberta(df, 'text')  # Predict sentiments using RoBERTa
# train_sentiment_model(df, 'text', 'sentiment')  # Train the logistic regression model
