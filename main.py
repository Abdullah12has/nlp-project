import os
import pandas as pd
import logging
from scripts.text_preprocessing import TextPreprocessor
from scripts.data_exploration import plot_feature_distribution
from scripts.sentiment_analysis import classify_sentiment, generate_wordcloud, ngram_analysis
from scripts.sentiment_correlation import plot_correlation_heatmap
from scripts.topic_modeling import train_lda_model, train_bertopic_model, analyze_topic_distribution_with_representation
from scripts.sentiment_prediction import train_sentiment_model_with_word2vec, train_sentiment_model_with_bert
from scripts.sentiment_model_comparison import compare_pretrained_models
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models

# Setup logging
logging.basicConfig(level=logging.INFO)

# Constants and paths
DATA_PATH = 'data/senti_df.csv'
TEXT_COLUMN = 'speech'
SENTIMENT_SCORE_COLUMN = 'sentiment_score'

# Step 1: Load Data
try:
    logging.info("Loading data...")
    df = pd.read_csv(DATA_PATH)
    logging.info("Data loaded successfully!")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

# Step 2: Text Preprocessing
try:
    logging.info("Starting text preprocessing...")
    preprocessor = TextPreprocessor()
    df['cleaned_text'] = df[TEXT_COLUMN].apply(preprocessor.clean_text)
    logging.info("Text preprocessing completed!")
except Exception as e:
    logging.error(f"Error during text preprocessing: {e}")
    raise

# Step 3: Data Exploration and Visualization
try:
    logging.info("Exploring data distributions...")
    for feature in ['Speech_date', 'year', 'time', 'gender', 'party_group']:
        if feature in df.columns:
            plot_feature_distribution(df, feature)
    logging.info("Data exploration completed!")
except Exception as e:
    logging.error(f"Error during data exploration: {e}")

# Step 4: Sentiment Classification and Word Frequency Analysis
try:
    logging.info("Classifying sentiment and analyzing word frequencies...")
    df = classify_sentiment(df, SENTIMENT_SCORE_COLUMN)
    generate_wordcloud(df, 'positive')
    generate_wordcloud(df, 'negative')

    logging.info("Running n-gram analysis...")
    ngram_analysis(df, 'positive', 2)  # Bi-gram analysis for positive speeches
    ngram_analysis(df, 'negative', 3)  # Tri-gram analysis for negative speeches

    logging.info("Word frequency and n-gram analysis completed!")
except Exception as e:
    logging.error(f"Error during sentiment classification or analysis: {e}")

# Step 5: Correlation Analysis
try:
    logging.info("Calculating and plotting correlation heatmap...")
    sentiment_columns = ['afinn_sentiment', 'jockers_sentiment', 'nrc_sentiment', 'huliu_sentiment', 'rheault_sentiment']
    if all(col in df.columns for col in sentiment_columns):
        plot_correlation_heatmap(df, sentiment_columns)
    else:
        logging.warning("Some sentiment columns are missing; skipping correlation heatmap.")
except Exception as e:
    logging.error(f"Error during correlation analysis: {e}")

# Step 6: Topic Modeling (LDA and BERTopic)
try:
    logging.info("Training LDA model...")
    lda_model, dictionary, corpus = train_lda_model(df, 'cleaned_text')
    
    # Visualization of LDA topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.display(vis)
    
    logging.info("LDA model training completed!")

    logging.info("Training BERTopic model...")
    bertopic_model = train_bertopic_model(df, 'cleaned_text')
    logging.info("BERTopic model training completed!")
except Exception as e:
    logging.error(f"Error during topic modeling: {e}")

# Step 7: Sentiment Prediction Using Machine Learning Models
try:
    logging.info("Training sentiment classification models...")
    train_sentiment_model_with_word2vec(df, 'cleaned_text', 'sentiment')  # Word2Vec Model
    train_sentiment_model_with_bert(df, 'cleaned_text', 'sentiment')  # BERT Model
    logging.info("Sentiment prediction model training completed!")
except Exception as e:
    logging.error(f"Error during sentiment prediction: {e}")

# Step 8: Comparison of Pre-Trained Sentiment Models with Ground Truth
try:
    logging.info("Comparing pre-trained sentiment models with ground truth...")
    df = compare_pretrained_models(df, TEXT_COLUMN, SENTIMENT_SCORE_COLUMN)
    logging.info("Pre-trained sentiment models comparison completed!")
except Exception as e:
    logging.error(f"Error during sentiment model comparison: {e}")

# Step 9: Topic Distributions Across Political Parties and Speakers
try:
    logging.info("Analyzing topic distributions across parties and speakers...")
    analyze_topic_distribution_with_representation(df, topic_column='topic', group_columns=['party_group', 'speaker'], topic_model=bertopic_model)
    logging.info("Topic distribution analysis completed!")
except Exception as e:
    logging.error(f"Error during topic distribution analysis: {e}")

# Step 10: Topic Evolution Over Time (NEW)
def analyze_topic_evolution(bertopic_model, df):
    # Implementation of topic evolution analysis
    # Example: Track topics across years or months
    logging.info("Analyzing topic evolution over time...")
    topic_over_time = df.groupby(['year', 'topic']).size().unstack(fill_value=0)
    topic_over_time.plot(kind='bar', stacked=True)
    plt.title('Topic Evolution Over Years')
    plt.ylabel('Number of Speeches')
    plt.xlabel('Year')
    plt.legend(title='Topics')
    plt.show()
    logging.info("Topic evolution analysis completed!")

# Call the new function
try:
    analyze_topic_evolution(bertopic_model, df)
except Exception as e:
    logging.error(f"Error during topic evolution analysis: {e}")

# Step 11: Sentiment Correlation with Topics (NEW)
def correlate_sentiment_with_topics(df, sentiment_column, topic_column):
    # Calculate and visualize correlation between sentiment and topics
    logging.info("Correlating sentiment with topics...")
    df['sentiment_bin'] = pd.cut(df[sentiment_column], bins=[-1, 0, 1], labels=['Negative', 'Positive'])
    correlation = df.groupby([topic_column, 'sentiment_bin']).size().unstack(fill_value=0)
    correlation.plot(kind='bar', stacked=True)
    plt.title('Sentiment Correlation with Topics')
    plt.ylabel('Count')
    plt.xlabel('Topics')
    plt.legend(title='Sentiment')
    plt.show()
    logging.info("Sentiment correlation with topics analysis completed!")

# Call the new function
try:
    correlate_sentiment_with_topics(df, SENTIMENT_SCORE_COLUMN, 'topic')
except Exception as e:
    logging.error(f"Error during sentiment correlation with topics analysis: {e}")

# Step 12: Manual Topic Interpretation (NEW)
def interpret_topics(model):
    # Interpret topics manually
    logging.info("Interpreting topics...")
    topics = model.get_topic_info()
    for index, row in topics.iterrows():
        logging.info(f"Topic {row['Topic']}: {row['Name']}")
    logging.info("Topic interpretation completed!")

# Call the new function
try:
    interpret_topics(bertopic_model)
except Exception as e:
    logging.error(f"Error during topic interpretation: {e}")

# Step 13: Advanced NLP Techniques with LLMs (NEW)
def explore_llm_transformers(df):
    # Implement LLMs exploration
    logging.info("Exploring LLMs for sentiment analysis...")
    # Here you can add a function to explore other transformers like RoBERTa or GPT
    # Placeholder for actual implementation
    logging.info("LLMs exploration completed!")

# Call the new function
try:
    explore_llm_transformers(df)
except Exception as e:
    logging.error(f"Error during LLM exploration: {e}")
