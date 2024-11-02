import pandas as pd
from sklearn.metrics import mean_absolute_error
from transformers import pipeline
import logging
import torch
import gc

# Function to truncate text to a maximum length of 512 characters (or tokens if tokenized properly)
def truncate_text(text, max_length=512):
    """Truncate text to a specified max length."""
    return text[:max_length]

def analyze_with_transformers(df, text_column, ground_truth_column):
    """Analyze sentiment using a transformer-based model (e.g., BERT)."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Apply truncation before sentiment analysis to avoid input length errors
    df['short_text'] = df[text_column].apply(lambda x: truncate_text(str(x), max_length=512))
    
    # Run sentiment analysis on the truncated text
    df['transformer_score'] = df['short_text'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(df[ground_truth_column], df['transformer_score'])
    print(f'Transformer MAE: {mae}')
    
    return df

def explore_llm_transformers(texts, classifier):
    """
    Process texts in batches using the provided classifier.
    
    Args:
        texts (list): List of text strings to analyze
        classifier: HuggingFace pipeline classifier
    
    Returns:
        list: Sentiment predictions
    """
    try:
        results = classifier(texts)
        return [r['label'] for r in results]
    except Exception as e:
        logging.error(f"Error in LLM processing: {e}")
        return ['NEUTRAL'] * len(texts)  # fallback value
