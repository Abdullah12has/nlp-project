import logging
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import mean_absolute_error, mean_squared_error  
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compare_pretrained_models(df, text_column, ground_truth_column):
    """Compare the performance of VADER and TextBlob sentiment analysis models with ground truth labels."""
    
    # VADER Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    df['vader_score'] = df[text_column].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

    # TextBlob Sentiment Analysis
    df['textblob_score'] = df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    # Normalize sentiment scores and ground truth to the range [-1, 1]
    df['normalized_ground_truth'] = np.interp(df[ground_truth_column], (df[ground_truth_column].min(), df[ground_truth_column].max()), (-1, 1))

    # Calculate MAE and MSE for VADER and TextBlob
    vader_mae = mean_absolute_error(df['normalized_ground_truth'], df['vader_score'])
    textblob_mae = mean_absolute_error(df['normalized_ground_truth'], df['textblob_score'])

    vader_mse = mean_squared_error(df['normalized_ground_truth'], df['vader_score'])
    textblob_mse = mean_squared_error(df['normalized_ground_truth'], df['textblob_score'])

    # Log results
    logging.info(f'VADER MSE: {vader_mse:.4f}, MAE: {vader_mae:.4f}')
    logging.info(f'TextBlob MSE: {textblob_mse:.4f}, MAE: {textblob_mae:.4f}')

    # Error Analysis: Calculate the absolute error for further analysis
    df['vader_error'] = abs(df['normalized_ground_truth'] - df['vader_score'])
    df['textblob_error'] = abs(df['normalized_ground_truth'] - df['textblob_score'])

    # Visualize model performance
    visualize_model_performance(df, 'normalized_ground_truth')

    return df

def visualize_model_performance(df, ground_truth_column):
    """Simple visualization of sentiment analysis model performance using MSE, MAE, and error distributions."""

    # Calculate MSE and MAE for each model
    vader_mse = mean_squared_error(df[ground_truth_column], df['vader_score'])
    vader_mae = mean_absolute_error(df[ground_truth_column], df['vader_score'])
    textblob_mse = mean_squared_error(df[ground_truth_column], df['textblob_score'])
    textblob_mae = mean_absolute_error(df[ground_truth_column], df['textblob_score'])

    # Bar plot for MSE and MAE comparison
    error_metrics = pd.DataFrame({
        'Model': ['VADER', 'TextBlob'],
        'MSE': [vader_mse, textblob_mse],
        'MAE': [vader_mae, textblob_mae]
    })

    plt.figure(figsize=(10, 5))
    error_metrics.set_index('Model')[['MSE', 'MAE']].plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
    plt.title('Mean Squared Error (MSE) and Mean Absolute Error (MAE) by Model')
    plt.ylabel('Error')
    plt.xticks(rotation=0)
    plt.legend(title="Error Metrics")
    plt.tight_layout()
    plt.show()

    # Calculate absolute errors
    df['vader_error'] = abs(df[ground_truth_column] - df['vader_score'])
    df['textblob_error'] = abs(df[ground_truth_column] - df['textblob_score'])

    # Box plot of error distributions
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[['vader_error', 'textblob_error']], palette="Set2")
    plt.xticks([0, 1], ['VADER Error', 'TextBlob Error'])
    plt.title('Error Distribution of VADER and TextBlob Scores')
    plt.ylabel('Absolute Error')
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(14, 7))
    # Plot VADER vs Ground Truth
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=ground_truth_column, y='vader_score', data=df, alpha=0.6)
    plt.plot([-1, 1], [-1, 1], 'r--', label="Perfect Alignment")  # Line for perfect prediction
    plt.title('VADER Sentiment Score vs Ground Truth')
    plt.xlabel('Normalized Ground Truth Sentiment')
    plt.ylabel('VADER Sentiment Score')
    plt.legend()

    # Plot TextBlob vs Ground Truth
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=ground_truth_column, y='textblob_score', data=df, alpha=0.6)
    plt.plot([-1, 1], [-1, 1], 'r--', label="Perfect Alignment")
    plt.title('TextBlob Sentiment Score vs Ground Truth')
    plt.xlabel('Normalized Ground Truth Sentiment')
    plt.ylabel('TextBlob Sentiment Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Example usage:
# df = pd.read_csv('your_data_file.csv')  # Load your DataFrame here
# compare_pretrained_models(df, 'text', 'sentiment')  # Call the function
