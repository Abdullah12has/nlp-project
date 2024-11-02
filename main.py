import os
import pandas as pd
import logging
from scripts.text_preprocessing import TextPreprocessor
from scripts.data_exploration import plot_feature_distribution
from scripts.sentiment_analysis import (
    classify_sentiment,
    generate_wordcloud,
    ngram_analysis,
    plot_most_common_words,
    train_sentiment_model_with_word2vec,
    train_sentiment_model_with_bert
)
from scripts.sentiment_correlation import (
    plot_correlation_heatmap,
    correlate_sentiment_with_topics,
    calculate_and_plot_correlations,
    encode_categorical_features
)
from scripts.topic_modeling import (
    train_lda_model,
    train_bertopic_model,
    analyze_topic_distribution_with_representation,
    topic_evolution_over_time,
    visualize_topic_trends
)
from scripts.sentiment_prediction import predict_sentiment_with_roberta
from scripts.sentiment_model_comparison import compare_pretrained_models
from scripts.llm_exploration import explore_llm_transformers
import numpy as np
import matplotlib.pyplot as plt
import time

# Setup logging
logging.basicConfig(level=logging.INFO)

# Constants and paths
DATA_PATH = 'data/senti_df.csv'
TEXT_COLUMN = 'speech'
SENTIMENT_SCORE_COLUMN = 'afinn_sentiment'
DEBUG_MODE = True  # Set to True to enable debug testing


if __name__ == '__main__':
    start_time = time.time()  # Start the timer

    # Step 1: Load Data
    try:
        logging.info("Loading data...")
        df = pd.read_csv(DATA_PATH)
        
        # If in DEBUG_MODE, take a sample of the data
        if DEBUG_MODE:
            df = df.sample(n=min(1000, len(df)))  # Adjust sample size as needed
        
        logging.info("Data loaded successfully!")
        logging.info(f"Data types:\n{df.dtypes}")

        if 'speech_date' in df.columns:
            df['speech_date'] = pd.to_datetime(df['speech_date'], errors='coerce').view(np.int64) // 10**9

        if 'time' in df.columns:
            df['time'] = pd.to_numeric(df['time'], errors='coerce')


    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Step 2: Encode Categorical Features
    categorical_features = ['gender', 'party_group']
    df = encode_categorical_features(df, categorical_features)

    # Step 3: Text Preprocessing
    try:
        logging.info("Starting text preprocessing...")
        preprocessor = TextPreprocessor()
        df['cleaned_text'] = df[TEXT_COLUMN].apply(preprocessor.clean_text)
        logging.info("Text preprocessing completed!")
    except Exception as e:
        logging.error(f"Error during text preprocessing: {e}")
        raise

    # Step 4: Initial Data Exploration
    try:
        logging.info("Exploring data distributions...")
        for feature in ['speech_date', 'year', 'time', 'gender', 'party_group']:
            if feature in df.columns:
                plot_feature_distribution(df, feature)
        logging.info("Data exploration completed!")
    except Exception as e:
        logging.error(f"Error during data exploration: {e}")

    # Step 5: Speech Word Frequency Analysis
    try:
        logging.info("Classifying sentiment and analyzing word frequencies...")
        df = classify_sentiment(df, SENTIMENT_SCORE_COLUMN)
        generate_wordcloud(df, 'positive')
        generate_wordcloud(df, 'negative')

        logging.info("Running n-gram analysis...")
        ngram_analysis(df, 'positive', 2)  # Bi-gram analysis for positive speeches
        ngram_analysis(df, 'negative', 3)  # Tri-gram analysis for negative speeches

        for feature in ['party_group', 'gender']:
            ngram_analysis(df, 'positive', feature=feature)
            ngram_analysis(df, 'negative', feature=feature)

        # Plotting most common words
        logging.info("Plotting most common words for positive speeches...")
        plot_most_common_words(df, 'positive')  # Call the function for positive sentiment
        logging.info("Plotting most common words for negative speeches...")
        plot_most_common_words(df, 'negative')  # Call the function for negative sentiment

        logging.info("Word frequency and n-gram analysis completed!")
    except Exception as e:
        logging.error(f"Error during sentiment classification or analysis: {e}")

    # Step 6: Correlation Between Features and Sentiment
    try:
        # Update this list based on the data types checked
        feature_list = ['year', 'gender', 'party_group']  # Exclude 'Speech_date' and 'time'

        # Call the function to calculate correlations and plot results
        correlations = calculate_and_plot_correlations(df, feature_list, SENTIMENT_SCORE_COLUMN)

        logging.info(f"Calculated correlations: {correlations}")

    except Exception as e:
        logging.error(f"Error during correlation calculations: {e}")

    # Step 7: Correlation Heatmap
    try:
        logging.info("Calculating and plotting correlation heatmap...")
        sentiment_columns = ['afinn_sentiment', 'bing_sentiment', 'nrc_sentiment', 'sentiword_sentiment', 'hu_sentiment']
        if all(col in df.columns for col in sentiment_columns):
            plot_correlation_heatmap(df, sentiment_columns)
        else:
            logging.warning("Some sentiment columns are missing; skipping correlation heatmap.")
    except Exception as e:
        logging.error(f"Error during correlation analysis: {e}")

    # Step 8: Train Topic Models
    try:
        logging.info("Training topic models...")
        lda_model, dictionary, corpus, vis = train_lda_model(df, 'cleaned_text')
        logging.info("LDA model training completed!")

        # Set n_topics based on DEBUG_MODE
        # n_topics = 5 if DEBUG_MODE else None  # Set to 5 in DEBUG_MODE, None for full in production
        topic_model, topics, probs = train_bertopic_model(df['speech'])
        bertopic_model = topic_model
        df['topic'] = topics
        logging.info("BERTopic model training completed!")
    except Exception as e:
        logging.error(f"Error during topic modeling: {e}")

    # Step 9: Analyze Topic Evolution
    try:
        logging.info("Analyzing topic evolution over time...")
        if 'topic' in df.columns:
            topic_evolution_over_time(df, topic_column='topic', time_column='year')
            logging.info("Topic evolution analysis completed!")

            logging.info("Visualizing topic trends...")
            visualize_topic_trends(df, topic_column='topic', time_column='year')
            logging.info("Topic trends visualization completed!")
        else:
            logging.error("The 'topic' column is missing; cannot analyze topic evolution.")
    except Exception as e:
        logging.error(f"Error during topic evolution analysis: {e}")

    # Step 10: Sentiment Correlation with Topics
    try:
        logging.info("Correlating sentiment with topics...")
        correlate_sentiment_with_topics(df, sentiment_column=SENTIMENT_SCORE_COLUMN, topic_column='topic')
        logging.info("Sentiment correlation with topics completed!")
    except Exception as e:
        logging.error(f"Error during sentiment correlation analysis: {e}")

    # Step 11: Comparison of Pre-Trained Sentiment Models with Ground Truth
    try:
        logging.info("Comparing pre-trained sentiment models with ground truth...")
        df = compare_pretrained_models(df, TEXT_COLUMN, SENTIMENT_SCORE_COLUMN)
        logging.info("Pre-trained sentiment models comparison completed!")
    except Exception as e:
        logging.error(f"Error during sentiment model comparison: {e}")

    # Step 12: Sentiment Prediction Using Extracted Features
    try:
        logging.info("Training sentiment classification models...")
        train_sentiment_model_with_word2vec(df, 'cleaned_text', 'sentiment')  # Word2Vec Model
        train_sentiment_model_with_bert(df, 'cleaned_text', 'sentiment')  # BERT Model
        logging.info("Sentiment prediction model training completed!")
    except Exception as e:
        logging.error(f"Error during sentiment prediction: {e}")

    # Step 13: Topic Distributions Across Political Parties and Speakers
    try:
        logging.info("Analyzing topic distributions across parties and speakers...")
        analyze_topic_distribution_with_representation(df, topic_column='topic', group_columns=['party_group', 'proper_name'], topic_model=bertopic_model)
        logging.info("Topic distribution analysis completed!")
    except Exception as e:
        logging.error(f"Error during topic distribution analysis: {e}")

    # Step 14: Explore LLM and Transformers
    try:
        logging.info("Exploring sentiment analysis with LLMs and transformers...")
        df = explore_llm_transformers(df, TEXT_COLUMN, SENTIMENT_SCORE_COLUMN)
        logging.info("LLM and transformer exploration completed!")
    except Exception as e:
        logging.error(f"Error during LLM exploration: {e}")
        
    # Final execution time
    execution_time = time.time() - start_time
    logging.info(f"Total execution time: {execution_time:.2f} seconds")
