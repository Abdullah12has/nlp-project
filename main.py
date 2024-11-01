import os
import pandas as pd
from scripts.text_preprocessing import TextPreprocessor
from scripts.data_exploration import plot_feature_distribution
from scripts.sentiment_analysis import classify_sentiment, generate_wordcloud, ngram_analysis
from scripts.sentiment_correlation import plot_correlation_heatmap
from scripts.topic_modeling import train_lda_model, train_bertopic_model
from scripts.sentiment_prediction import train_sentiment_model

# Constants and paths
DATA_PATH = 'data/senti_df.csv'
TEXT_COLUMN = 'speech'  # Column in your data that contains the text data
SENTIMENT_SCORE_COLUMN = 'sentiment_score'  # Replace with actual score column name in the dataset

# Step 1: Load Data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print("Data loaded successfully!")

# Step 2: Text Preprocessing
print("Starting text preprocessing...")
preprocessor = TextPreprocessor()
df['cleaned_text'] = df[TEXT_COLUMN].apply(preprocessor.clean_text)
print("Text preprocessing completed!")

# Step 3: Data Exploration and Visualization
print("Exploring data distributions...")
for feature in ['Speech_date', 'year', 'time', 'gender', 'party_group']:
    if feature in df.columns:
        plot_feature_distribution(df, feature)
print("Data exploration completed!")

# Step 4: Sentiment Classification and Word Frequency Analysis
print("Classifying sentiment and analyzing word frequencies...")
df = classify_sentiment(df, SENTIMENT_SCORE_COLUMN)
generate_wordcloud(df, 'positive')
generate_wordcloud(df, 'negative')

print("Running n-gram analysis...")
ngram_analysis(df, 'positive', 2)  # Bi-gram analysis for positive speeches
ngram_analysis(df, 'negative', 3)  # Tri-gram analysis for negative speeches

print("Word frequency and n-gram analysis completed!")

# Step 5: Correlation Analysis
print("Calculating and plotting correlation heatmap...")
sentiment_columns = ['afinn_sentiment', 'jockers_sentiment', 'nrc_sentiment', 'huliu_sentiment', 'rheault_sentiment']
if all(col in df.columns for col in sentiment_columns):
    plot_correlation_heatmap(df, sentiment_columns)
else:
    print("Some sentiment columns are missing; skipping correlation heatmap.")

# Step 6: Topic Modeling (LDA and BERTopic)
print("Training LDA model...")
lda_model, dictionary, corpus = train_lda_model(df, 'cleaned_text')
print("LDA model training completed!")

print("Training BERTopic model...")
bertopic_model = train_bertopic_model(df, 'cleaned_text')
print("BERTopic model training completed!")

# Step 7: Sentiment Prediction Using Machine Learning Models
print("Training sentiment classification model...")
train_sentiment_model(df, 'cleaned_text', 'sentiment')
print("Sentiment prediction model training completed!")

print("Project execution completed successfully!")
