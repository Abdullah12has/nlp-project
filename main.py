import os
import pandas as pd
import logging

import pyLDAvis
from scripts.task6 import train_and_visualize_bertopic, train_and_visualize_lda
from scripts.text_preprocessing import TextPreprocessor
from scripts.data_exploration import plot_feature_distribution
from scripts.sentiment_analysis import (
    classify_sentiment,
    filtered_ngram_analysis,
    generate_wordcloud,
    ngram_analysis,
    plot_most_common_words,
    plot_most_common_words_with_filter,
    plot_ngram_analysis,
    train_sentiment_model_with_word2vec,
    train_sentiment_model_with_bert,
)
from scripts.sentiment_correlation import (
    plot_correlation_heatmap,
    calculate_and_plot_correlations,
    encode_categorical_features,
    plot_sentiment_distribution,
    correlate_sentiment_with_topics,
    analyze_sentiment
)
from scripts.topic_modeling import (
    train_lda_model,
    train_bertopic_model,
    analyze_topic_distribution_with_representation,
    interpret_topics_with_experts_and_automation
)
from scripts.sentiment_prediction import predict_sentiment_with_roberta
from scripts.sentiment_model_comparison import compare_pretrained_models
from scripts.llm_exploration import explore_llm_transformers
import numpy as np
import matplotlib.pyplot as plt
import time
from transformers import pipeline
import torch
import gc


# Setup logging
logging.basicConfig(level=logging.INFO)

# Constants and paths
# DATA_PATH = 'data/subset_senti_df_100.csv'
DATA_PATH = 'data/subset_senti_df.csv'
TEXT_COLUMN = 'speech'
SENTIMENT_SCORE_COLUMN = 'afinn_sentiment'
DEBUG_MODE = False  # Set to True to enable debug testing


if __name__ == '__main__':
    start_time = time.time()  # Start the timer

    # Step 1: Load Data
    try:
        logging.info("Loading data...")
        
        # Check if DATA_PATH is defined and valid
        if not os.path.exists(DATA_PATH):
            logging.error("DATA_PATH is not defined or the file does not exist.")
            raise ValueError(f"DATA_PATH must point to a valid CSV file: {DATA_PATH}")
        
        df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
        
        # Drop rows with missing sentiment values
        df = df.dropna(subset=['afinn_sentiment', 'bing_sentiment', 'nrc_sentiment'])  # Clean sentiment columns
        
        # If in DEBUG_MODE, take a sample of the data
        if DEBUG_MODE:
            df = df.sample(n=min(1000, len(df)))  # Adjust sample size as needed
        
        logging.info("Data loaded successfully!")
        logging.info(f"Data types:\n{df.dtypes}")

        # Handle speech_date conversion
        if 'speech_date' in df.columns:
            df['speech_date'] = pd.to_datetime(df['speech_date'], errors='coerce')
            if df['speech_date'].isnull().any():
                logging.warning("Some 'speech_date' entries were coerced to NaT.")
            df['party'] = df['party'].astype(str)
            df['speech_date'] = df['speech_date'].view(np.int64) // 10**9  # Convert to seconds since epoch

        # Handle time conversion
        if 'time' in df.columns:
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            if df['time'].isnull().any():
                logging.warning("Some 'time' entries were coerced to NaN.")

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Step 2: Encode Categorical Features
    categorical_features = ['gender', 'party_group']
    df = encode_categorical_features(df, categorical_features)

    # Step 3: Handle Missing Values
    try:
        logging.info("Checking for missing values in the DataFrame...")
        missing_values = df.isnull().sum()
        logging.info(f"Missing values in each column:\n{missing_values[missing_values > 0]}")

        if df['year'].isnull().any():
            logging.warning("Missing values found in 'year' column; filling with median.")
            df['year'].fillna(df['year'].median(), inplace=True)

        if df['gender'].isnull().any():
            logging.warning("Missing values found in 'gender' column; filling with 'Unknown'.")
            df['gender'].fillna('Unknown', inplace=True)

        if df['party_group'].isnull().any():
            logging.warning("Missing values found in 'party_group' column; filling with 'Unknown'.")
            df['party_group'].fillna('Unknown', inplace=True)

        logging.info("Missing values handled successfully.")
        
    except Exception as e:
        logging.error(f"Error handling missing values: {e}")
        raise

    # Step 4: Text Preprocessing
    try:
        logging.info("Starting text preprocessing...")
        preprocessor = TextPreprocessor()
        df['cleaned_text'] = df[TEXT_COLUMN].apply(preprocessor.clean_text)
        logging.info("Text preprocessing completed!")
    except Exception as e:
        logging.error(f"Error during text preprocessing: {e}")
        raise

    # # Step 5: Initial Data Exploration
    try:
        logging.info("Exploring data distributions...")
       
        plot_feature_distribution(df)

        logging.info("Data exploration completed!")
    except Exception as e:
        logging.error(f"Error during data exploration: {e}")

    # Step 6: Speech Word Frequency Analysis
    try:
        logging.info("Classifying sentiment and analyzing word frequencies...")
        df, sentiment_summary = classify_sentiment(df)
        if sentiment_summary:
            logging.info("Sentiment Classification Summary:")
            for key, value in sentiment_summary.items():
                logging.info(f"{key}: {value}")
        else:
            logging.error("Sentiment classification failed to produce summary")
        
        generate_wordcloud(df, 'positive')
        generate_wordcloud(df, 'negative')

        # Plot most common words for each sentiment
        logging.info("Plotting most common words for positive speeches...")
        plot_most_common_words(df, 'positive')
        logging.info("Plotting most common words for negative speeches...")
        plot_most_common_words(df, 'negative')
    
        logging.info("Word frequency and sentiment analysis completed!")

        logging.info("Running bi-gram analysis...") 
        bigramPositive = ngram_analysis(df, 'positive', 2)  # Bi-gram analysis for positive speeches
        bigramNegative = ngram_analysis(df, 'negative', 2)  # Bi-gram analysis for negative speeches

        logging.info("Running tri-gram analysis...") 
        trigramPositive = ngram_analysis(df, 'positive', 3)  # Tri-gram analysis for positive speeches
        trigramNegative = ngram_analysis(df, 'negative', 3)  # Tri-gram analysis for negative speeches

        plot_ngram_analysis(bigramPositive, "Top 10 Bigrams in Positive Speeches")
        plot_ngram_analysis(bigramNegative, "Top 10 Bigrams in Negative Speeches")
        plot_ngram_analysis(trigramPositive, "Top 10 Trigrams in Positive Speeches")
        plot_ngram_analysis(trigramNegative, "Top 10 Trigrams in Negative Speeches")


        plot_most_common_words_with_filter(df, 'positive', feature='gender', filter_value=0)
        plot_most_common_words_with_filter(df, 'negative', feature='gender', filter_value=0)

        plot_most_common_words_with_filter(df, 'positive', feature='gender', filter_value=1)
        plot_most_common_words_with_filter(df, 'negative', feature='gender', filter_value=1)


        male_positive_bigrams = filtered_ngram_analysis(df, 'positive', n=2, feature="gender", filter_value=0) #male == 0 female == 1
        male_negative_bigrams = filtered_ngram_analysis(df, 'negative', n=2, feature="gender", filter_value=0) #male == 0 female == 1

        male_positive_trigrams = filtered_ngram_analysis(df, 'positive', n=3, feature="gender", filter_value=0) #male == 0 female == 1
        male_negative_trigrams = filtered_ngram_analysis(df, 'negative', n=3, feature="gender", filter_value=0) #male == 0 female == 1



        female_positive_bigrams = filtered_ngram_analysis(df, 'positive', n=2, feature="gender", filter_value=1) #male == 0 female == 1
        female_negative_bigrams = filtered_ngram_analysis(df, 'negative', n=2, feature="gender", filter_value=1) #male == 0 female == 1
        female_positive_trigrams = filtered_ngram_analysis(df, 'positive', n=3, feature="gender", filter_value=1) #male == 0 female == 1
        female_negative_trigrams = filtered_ngram_analysis(df, 'negative', n=3, feature="gender", filter_value=1) #male == 0 female == 1

        print(df['party_group'])
        # Consertive == 0. Labour == 1.  Independant  == 2.
        plot_most_common_words_with_filter(df, 'positive', feature='party_group', filter_value=0)
        plot_most_common_words_with_filter(df, 'negative', feature='party_group', filter_value=0)

        plot_most_common_words_with_filter(df, 'positive', feature='party_group', filter_value=1)
        plot_most_common_words_with_filter(df, 'negative', feature='party_group', filter_value=1)

        plot_most_common_words_with_filter(df, 'positive', feature='party_group', filter_value=2)
        plot_most_common_words_with_filter(df, 'negative', feature='party_group', filter_value=2)

        filtered_ngram_analysis(df, 'positive', n=2, feature="party_group", filter_value=0)
        filtered_ngram_analysis(df, 'negative', n=2, feature="party_group", filter_value=0)
        filtered_ngram_analysis(df, 'positive', n=3, feature="party_group", filter_value=0)
        filtered_ngram_analysis(df, 'negative', n=3, feature="party_group", filter_value=0)

        filtered_ngram_analysis(df, 'positive', n=2, feature="party_group", filter_value=1)
        filtered_ngram_analysis(df, 'negative', n=2, feature="party_group", filter_value=1)
        filtered_ngram_analysis(df, 'positive', n=3, feature="party_group", filter_value=1)
        filtered_ngram_analysis(df, 'negative', n=3, feature="party_group", filter_value=1)

        filtered_ngram_analysis(df, 'positive', n=2, feature="party_group", filter_value=2)
        filtered_ngram_analysis(df, 'negative', n=2, feature="party_group", filter_value=2)
        filtered_ngram_analysis(df, 'positive', n=3, feature="party_group", filter_value=2)
        filtered_ngram_analysis(df, 'negative', n=3, feature="party_group", filter_value=2)


        logging.info("Word frequency and n-gram analysis completed!")
        
    except Exception as e:
        logging.error(f"Error during sentiment classification or analysis: {e}")

    # Step 7: Correlation Between Features and Sentiment
    '''
    4. Correlation Between Features and Sentiment: Calculate the correlation between sentiment scores and features of
    Speech_date, year, time, gender and party_group and analyze whether a certain feature (e.g., Male, Labour and …) tend to be
    positive or negative. Plot the distribution plots for positive and negative reviews to explore potential patterns.
    '''
    try:
        logging.info("Classifying sentiment...")
        df, summary = classify_sentiment(df)    
        # Drop rows with NaN in key columns before correlation analysis
        df = df.dropna(subset=['year', 'gender', 'party_group', 'sentiment_confidence'])
        
        # Define features to analyze
        features_to_analyze = ['speech_date', 'year', 'gender', 'party_group']
        sentiment_column = 'sentiment_confidence'
        sentiment_label_column = 'sentiment'
        
        # Calculate and plot correlations including sentiment classification
        correlation_results = calculate_and_plot_correlations(df, features_to_analyze, sentiment_column, sentiment_label_column)
        
        # Plot sentiment distribution for each categorical feature
        categorical_features = ['gender', 'party_group']
        for feature in categorical_features:
            plot_sentiment_distribution(df, feature, sentiment_label_column)
        
        logging.info("Correlation analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error during correlation analysis: {e}")
        
  # # Step 8: Correlation Heatmap
    try:
        logging.info("Calculating and plotting correlation heatmap...")
        sentiment_columns = ['afinn_sentiment', 'bing_sentiment', 'nrc_sentiment', 'sentiword_sentiment', 'hu_sentiment']
        if all(col in df.columns for col in sentiment_columns):
            plot_correlation_heatmap(df, sentiment_columns)
        else:
            logging.warning("Some sentiment columns are missing; skipping correlation heatmap.")
    except Exception as e:
        logging.error(f"Error during correlation analysis: {e}")
    # Step 9: Train Topic Models
    '''
    5 Topic Modeling with LDA and BERTopic: Implement topic modeling using LDA and BERTopic and then optimize
    hyperparameters for both models using coherence scores (e.g., Cv measure) to ensure optimal topic extraction. Use
    visualization tools like pyLDAvis and BERTopic's built-in functions for interactive topic exploration.
    '''
    try:
        logging.info("Training topic models...")
        lda_model, dictionary, corpus, vis, ldatopics = train_lda_model(df, 'cleaned_text', 2, 10, 10)
        pyLDAvis.save_html(vis, 'graphs/lda_visualization.html')
        pyLDAvis.display(vis)
        logging.info("LDA model training completed!")
        print(ldatopics)

        logging.info("Starting BERTopic model training...")
        
        # Train BERTopic model with savepoints
        bertopic_model, topics, probs = train_bertopic_model(df['speech'], "progress/bertopic_checkpoint.pkl", min_topic_size=15)
        
        print(topics)
        # Save topics to DataFrame
        df['topic'] = topics
        logging.info("BERTopic model training completed!")

        # Visualization
        logging.info("Generating visualizations for BERTopic...")
        topic_vis = bertopic_model.visualize_topics()
        topic_vis.write_html("data/bertopic_topics.html")  # Save General Topic visualization

        hierarchy_vis = bertopic_model.visualize_hierarchy()
        hierarchy_vis.write_html("data/bertopic_hierarchy.html")  # Save Topic Hierarchy visualization

        heatmap_vis = bertopic_model.visualize_heatmap()
        heatmap_vis.write_html("data/bertopic_heatmap.html")  # Save Topic Similarity Heatmap

        barchart_vis = bertopic_model.visualize_barchart()
        barchart_vis.write_html("data/bertopic_barchart.html")
        # df.to_pickle("dataframe.pkl")

    except Exception as e:
        logging.error(f"Error during topic modeling: {e}")


    # Step 10: Analyze Topic Evolution
    '''
    6- Topic Evolution Over Time: Track how topics evolve over time using Dynamic Topic Modeling (LDA) and BERTopics
    time-based analysis. Try to visualize topic trends using dynamic topic models to study policy shifts.
    '''
    try:
        
        logging.info("Task 6: LDA training")
        lda_model, df, topic_names = train_and_visualize_lda(df, text_column='cleaned_text', time_column='year')
        # visualize_topic_trends(df, topic_column='lda_topic', time_column='year', topic_names=topic_names)

        logging.info("Task 6: BERTopic training")
        bertopic_model, df = train_and_visualize_bertopic(df)
       
        df.to_pickle("dataframe.pkl")
        # visualize_topic_trends(df, 'bertopic_topic', 'year')

    except Exception as e:
        logging.error(f"Error during topic evolution analysis: {e}")


    # Step 11: Sentiment Correlation with Topics
    '''
    7. Sentiment Correlation with Topics: Perform sentiment analysis on debate transcripts using VADER or TextBlob, with
    potential fine-tuning for parliamentary language. Correlate sentiment trends with identified topics, highlighting emotional
    patterns and contextual nuances like sarcasm or negation.
    
    '''
    try:
        df = pd.read_pickle("dataframe.pkl")
        # print(df)


        logging.info("Analyzing sentiment...")
        # df, summary = classify_sentiment(df)
        df = analyze_sentiment(df, text_column='cleaned_text') 
        logging.info("Sentiment analysis completed!")
        print(df["sentiment_score"])

        logging.info("Correlating sentiment with topics...")
        # correlate_sentiment_with_topics(df, sentiment_column='sentiment_score', topic_column='bertopic_topic')
        correlate_sentiment_with_topics(df, topic_column='bertopic_topic', sentiment_column='sentiment_score', time_column="year", output_dir='outputT7')
        logging.info("Sentiment correlation with topics completed!")

    except Exception as e:
        logging.error(f"Error during sentiment correlation analysis: {e}")

    # # Step 12: Comparison of Pre-Trained Sentiment Models with Ground Truth
    '''
    Comparison of Pre-Trained Sentiment Models with Ground Truth Labels: Use pre-trained models like VADER and
    TextBlob to analyze the sentiment of speeches. Then, compare these sentiment scores with the ground truth scores in the
    dataset by error analysis to evaluate the alignment between the pre-trained models and the true scores. Calculate evaluation
    metrics such as MSE and RMSE for both VADER and TextBlob. Finally, conduct an error analysis to examine cases where
    the pre-trained models significantly diverge from the ground truth (e.g., positive speeches but classified as negative by
    VADER). Discuss any patterns or characteristics in these speeches, such as the presence of sarcasm, ambiguous language, or
    other linguistic features, where the models tend to perform poorly.
    
    '''
    try:
        df = pd.read_pickle("dataframe.pkl")
        logging.info("Comparing pre-trained sentiment models with ground truth...")
    
        # Add device specification and batch size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = pipeline("sentiment-analysis", device=device)
    
        # Perform sentiment model comparison
        df = compare_pretrained_models(df, TEXT_COLUMN, SENTIMENT_SCORE_COLUMN) # using afinn need to normalize both the vals
    
        # Identify high-error cases for VADER
        high_error_vader = df[df['vader_error'] > 0.5]
        print("High Error Cases for VADER:", high_error_vader[[TEXT_COLUMN, SENTIMENT_SCORE_COLUMN, 'vader_score', 'vader_error']])
    
        # Save high-error cases to a text file
        high_error_vader[[TEXT_COLUMN, SENTIMENT_SCORE_COLUMN, 'vader_score', 'vader_error']].to_csv('vader_high_error.txt', sep='\t', index=False)
        logging.info("Saved high-error cases for VADER to vader_high_error.txt")

        logging.info("Pre-trained sentiment models comparison completed!")
    except Exception as e:
        logging.error(f"Error during sentiment model comparison: {e}")


    # Step 13: Sentiment Prediction Using Extracted Features
    try:
        logging.info("Training sentiment classification models...")
        train_sentiment_model_with_word2vec(df, 'cleaned_text', 'sentiment')  # Word2Vec Model
        train_sentiment_model_with_bert(df, 'cleaned_text', 'sentiment')  # BERT Model
        logging.info("Sentiment prediction model training completed!")
    except Exception as e:
        logging.error(f"Error during sentiment prediction: {e}")

    # Step 14: Topic Distributions Across Political Parties and Speakers
    try:
        # Assuming df is already available and contains topic modeling results
        logging.info("Starting the topic analysis...")
        df = pd.read_pickle("dataframe.pkl")
        # print(df)
        analyze_topic_distribution_with_representation(df)

        logging.info("Main script completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        
    # Step 15: Topic Representation and Interpretation:
    
    try:
        # df = pd.read_pickle("dataframe.pkl")
        # Interpret the topics and automate labeling
        
        # xxxx No Need to run this again if Task 5 is already run xxxxxxx
        # bertopic_model, topics, probs = train_bertopic_model(df['speech'], "progress/bertopic_checkpoint.pkl", min_topic_size=15)
        # df['bertopic_topic'] = topics
        
        # Interpret topics with predefined labels NEED TO BE ADDED AS PER SUBJECT
        predefined_labels = {
            'Economy': ['finance', 'economy', 'tax', 'budget'],
            'Health': ['health', 'hospital', 'medicine', 'treatment'],
            'Environment': ['climate', 'pollution', 'environment', 'sustainability']
        }
        result = interpret_topics_with_experts_and_automation(df, bertopic_model,predefined_labels=predefined_labels)
        # Print the result (optional)
        # print(result)
        # Save the result to a CSV file
        result.to_csv("topic_interpretation_results.csv", index=False)
        # Optionally, save the result as a pickle file
        result.to_pickle("topic_interpretation_results.pkl")
        
    except Exception as e:
        logging.error(f"Error during topic interpretation: {e}")
        
    # Step 16: Explore LLM and Transformers

    try:
        logging.info("Exploring sentiment analysis with LLMs and transformers...")
        # Explicitly set device and manage memory better
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        # Set smaller batch size and add padding
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            batch_size=4,  # Reduced batch size
            padding=True,
            truncation=True,
            max_length=512  # Limit input length
        )
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 100
        results = []
        for i in range(0, len(df), chunk_size):
            chunk = df[TEXT_COLUMN].iloc[i:i+chunk_size].tolist()
            chunk_results = explore_llm_transformers(chunk, classifier)
            results.extend(chunk_results)
            
            # Clear memory after each chunk
            if device != "cpu":
                torch.cuda.empty_cache()
            gc.collect()
        
        df['llm_sentiment'] = results
        logging.info("LLM and transformer exploration completed!")
        
    except Exception as e:
        logging.error(f"Error during LLM exploration: {e}")
        logging.warning("Skipping LLM exploration due to error")
    finally:
        # Cleanup
        if device != "cpu":
            torch.cuda.empty_cache()
        gc.collect()

    # Final execution time
    execution_time = time.time() - start_time
    logging.info(f"Total execution time: {execution_time:.2f} seconds")

    # After processing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
