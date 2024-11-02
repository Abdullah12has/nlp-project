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
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Initialize lists to hold predictions and scores
    roberta_sentiments = []
    roberta_scores = []

    # Predict sentiment for each text entry in batches
    batch_size = 16  # You can adjust this based on your memory capacity
    for i in range(0, len(df), batch_size):
        batch_texts = df[text_column].tolist()[i:i + batch_size]
        
        # Ensure that all texts are padded correctly
        try:
            sentiments = sentiment_pipeline(batch_texts, truncation=True, padding=True, max_length=512)
            # Extract the label and score for each prediction in the batch
            for result in sentiments:
                roberta_sentiments.append(result['label'])
                roberta_scores.append(result['score'])
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            roberta_sentiments.extend(['unknown'] * len(batch_texts))  # Handle the error case
            roberta_scores.extend([0.0] * len(batch_texts))  # Dummy score for unknown cases

    # Assign the predictions to the DataFrame
    df['roberta_sentiment'] = roberta_sentiments
    df['roberta_score'] = roberta_scores
    
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
