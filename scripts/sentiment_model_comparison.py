import logging
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import mean_absolute_error  
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mean_squared_error_manual(y_true, y_pred):
    """Calculate mean squared error manually."""
    return np.mean((y_true - y_pred) ** 2)

def compare_pretrained_models(df, text_column, ground_truth_column):
    """Compare the performance of different sentiment analysis models."""

    # VADER Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    df['vader_score'] = df[text_column].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

    # TextBlob Sentiment Analysis
    df['textblob_score'] = df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    # Calculate MAE and MSE
    vader_mae = mean_absolute_error(df[ground_truth_column], df['vader_score'])
    textblob_mae = mean_absolute_error(df[ground_truth_column], df['textblob_score'])

    vader_mse = mean_squared_error_manual(df[ground_truth_column], df['vader_score'])
    textblob_mse = mean_squared_error_manual(df[ground_truth_column], df['textblob_score'])

    # Log results
    logging.info(f'VADER MSE: {vader_mse:.4f}, MAE: {vader_mae:.4f}')
    logging.info(f'TextBlob MSE: {textblob_mse:.4f}, MAE: {textblob_mae:.4f}')

    # Error Analysis
    df['vader_error'] = abs(df[ground_truth_column] - df['vader_score'])
    df['textblob_error'] = abs(df[ground_truth_column] - df['textblob_score'])

    # Visualize model performance
    visualize_model_performance(df, ground_truth_column)

    return df

def visualize_model_performance(df, ground_truth_column):
    """Visualize the performance of sentiment analysis models."""
    plt.figure(figsize=(14, 7))

    # Plotting VADER vs Ground Truth
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df[ground_truth_column], y=df['vader_score'], alpha=0.6)
    plt.plot([df[ground_truth_column].min(), df[ground_truth_column].max()], 
             [df[ground_truth_column].min(), df[ground_truth_column].max()], 
             'r--')
    plt.title('VADER Sentiment Score vs Ground Truth')
    plt.xlabel('Ground Truth Sentiment')
    plt.ylabel('VADER Sentiment Score')

    # Plotting TextBlob vs Ground Truth
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=df[ground_truth_column], y=df['textblob_score'], alpha=0.6)
    plt.plot([df[ground_truth_column].min(), df[ground_truth_column].max()], 
             [df[ground_truth_column].min(), df[ground_truth_column].max()], 
             'r--')
    plt.title('TextBlob Sentiment Score vs Ground Truth')
    plt.xlabel('Ground Truth Sentiment')
    plt.ylabel('TextBlob Sentiment Score')

    plt.tight_layout()
    plt.show()

# Example usage:
# df = pd.read_csv('your_data_file.csv')  # Load your DataFrame here
# compare_pretrained_models(df, 'text', 'sentiment')  # Call the function
