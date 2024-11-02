import pandas as pd
from sklearn.metrics import mean_absolute_error
from transformers import pipeline

def analyze_with_transformers(df, text_column, ground_truth_column):
    """Analyze sentiment using a transformer-based model (e.g., BERT)."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    df['transformer_score'] = df[text_column].apply(lambda x: sentiment_pipeline(str(x))[0]['score'])
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(df[ground_truth_column], df['transformer_score'])
    
    print(f'Transformer MAE: {mae}')
    return df

def explore_llm_transformers(df, text_column, ground_truth_column):
    """Main function to explore LLM and transformer sentiment analysis."""
    df = analyze_with_transformers(df, text_column, ground_truth_column)
    return df
