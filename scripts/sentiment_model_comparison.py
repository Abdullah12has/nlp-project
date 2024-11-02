from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import mean_absolute_error  # Only import MAE now
import numpy as np

def compare_pretrained_models(df, text_column, ground_truth_column):
    # VADER Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    df['vader_score'] = df[text_column].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

    # TextBlob Sentiment Analysis
    df['textblob_score'] = df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    # Calculate MAE
    vader_mae = mean_absolute_error(df[ground_truth_column], df['vader_score'])
    textblob_mae = mean_absolute_error(df[ground_truth_column], df['textblob_score'])

    # Calculate MSE manually if required
    df['vader_squared_error'] = (df[ground_truth_column] - df['vader_score']) ** 2
    df['textblob_squared_error'] = (df[ground_truth_column] - df['textblob_score']) ** 2
    
    vader_mse = df['vader_squared_error'].mean()  # Manual MSE calculation
    textblob_mse = df['textblob_squared_error'].mean()  # Manual MSE calculation
    
    # Error Analysis
    df['vader_error'] = abs(df[ground_truth_column] - df['vader_score'])
    df['textblob_error'] = abs(df[ground_truth_column] - df['textblob_score'])

    print(f'VADER MSE: {vader_mse}, MAE: {vader_mae}')
    print(f'TextBlob MSE: {textblob_mse}, MAE: {textblob_mae}')
    
    return df
