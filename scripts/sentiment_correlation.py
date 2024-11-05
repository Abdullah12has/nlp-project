import seaborn as sns
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Ensure logging is configured
logging.basicConfig(level=logging.INFO)

def encode_categorical_features(df, categorical_features):
    """Encode categorical features for correlation analysis."""
    for feature in categorical_features:
        if feature in df.columns and df[feature].notnull().all():
            df[feature] = pd.Categorical(df[feature]).codes
    return df
def plot_correlation_heatmap(df, sentiment_columns):
    correlation_matrix = df[sentiment_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Sentiment Scores')
    plt.show()

def correlate_sentiment_with_topics(df, sentiment_column='sentiment_confidence', topic_column='topic'):
    """
    Correlate sentiment scores with topics and visualize the results.
    
    Args:
        df (pd.DataFrame): DataFrame containing sentiment and topic information.
        sentiment_column (str): Name of the column with sentiment scores.
        topic_column (str): Name of the column with topic information.
    """
    # Grouping by topic and calculating mean sentiment
    sentiment_by_topic = df.groupby(topic_column)[sentiment_column].mean().reset_index()

    # Create a bar plot for mean sentiment score by topic
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sentiment_by_topic, x=topic_column, y=sentiment_column, palette='coolwarm')
    plt.title('Mean Sentiment Confidence by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Mean Sentiment Confidence')
    plt.xticks(rotation=45)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Add a line at 0 for reference
    plt.tight_layout()
    plt.show()

    # Optional: Analyze sentiment distribution for each topic
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=topic_column, y=sentiment_column, palette='coolwarm')
    plt.title('Sentiment Confidence Distribution by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Sentiment Confidence')
    plt.xticks(rotation=45)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Add a line at 0 for reference
    plt.tight_layout()
    plt.show()
    
def analyze_sentiment(df, text_column, sentiment_column='sentiment_score'):
    """
    Analyze sentiment using VADER and add sentiment scores to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing text data.
        text_column (str): Column name for preprocessed text.
        sentiment_column (str): Column name to store sentiment scores.
    
    Returns:
        pd.DataFrame: DataFrame with sentiment scores added.
    """
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Calculate sentiment scores
    df[sentiment_column] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
    
    return df
    

def perform_pca(df, n_components=2):
    """Perform PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    return principal_components

def plot_sentiment_distribution(df, feature, sentiment_label_column='sentiment'):
    """
    Plots the distribution of positive and negative sentiment for a given categorical feature.
    
    Args:
        df (pd.DataFrame): The input DataFrame with sentiment data.
        feature (str): The feature for which to plot the sentiment distribution.
        sentiment_label_column (str): The column containing positive/negative sentiment labels.
    """
    if feature in df.columns:
        # Plot sentiment count for each category
        sns.countplot(x=feature, hue=sentiment_label_column, data=df)
        plt.title(f'Sentiment Distribution by {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print(f"Feature '{feature}' not found in DataFrame.")
    
def clean_numeric_columns(df, numeric_columns):
    """Convert specified numeric columns to numeric, coercing errors to NaN."""
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def identify_non_numeric_values(df, columns):
    """Identify and log non-numeric values in specified columns."""
    for column in columns:
        if column in df.columns:
            non_numeric = df[~df[column].apply(lambda x: isinstance(x, (int, float)))][column]
            if not non_numeric.empty:
                logging.warning(f"Non-numeric values in column '{column}': {non_numeric.tolist()}")

def drop_low_variance_features(df, threshold=0.01):
    """Drop features with low variance."""
    variances = df.var(numeric_only=True)  # Ensure only numeric columns are considered
    low_variance_features = variances[variances < threshold].index
    logging.info(f"Dropping low variance features: {low_variance_features.tolist()}")
    df = df.drop(columns=low_variance_features)
    return df

def calculate_and_plot_correlations(df, features, sentiment_column, sentiment_label_column='sentiment'):
    """
    Calculates and plots the correlation between sentiment scores and specified features,
    with feature-level visualizations that include sentiment classification.
    
    Args:
        df (pd.DataFrame): The input DataFrame with sentiment data.
        features (list): A list of feature column names to correlate with sentiment scores.
        sentiment_column (str): The name of the column containing sentiment scores.
        sentiment_label_column (str): The name of the column containing positive/negative labels.
        
    Returns:
        pd.Series: The correlation results between sentiment scores and specified features.
    """
    correlations = {}

    for feature in features:
        if df[feature].dtype == 'object' or df[feature].nunique() < 10:  # Categorical data handling
            # Plot the distribution of sentiment confidence for each category split by sentiment label
            sns.boxplot(x=feature, y=sentiment_column, hue=sentiment_label_column, data=df)
            plt.title(f'Sentiment Confidence by {feature.capitalize()} and Sentiment Type')
            plt.xticks(rotation=45)
            plt.show()
            
            # Calculate the percentage of positive and negative sentiment for each category
            sentiment_distribution = df.groupby([feature, sentiment_label_column]).size().unstack(fill_value=0)
            sentiment_distribution = (sentiment_distribution.T / sentiment_distribution.sum(axis=1)).T * 100
            sentiment_distribution.plot(kind='bar', stacked=True)
            plt.title(f'Sentiment Distribution by {feature.capitalize()}')
            plt.ylabel('Percentage (%)')
            plt.show()
            
            correlations[feature] = df.groupby(feature)[sentiment_column].mean()

        else:  # Numerical data handling
            # Calculate and display correlation
            correlation = df[feature].corr(df[sentiment_column])
            correlations[feature] = correlation
            
            # Plot scatterplot with color-coded sentiment
            sns.scatterplot(x=feature, y=sentiment_column, hue=sentiment_label_column, data=df)
            plt.title(f'Correlation between {feature.capitalize()} and Sentiment Confidence')
            plt.show()
    
    # Display correlation summary
    correlation_series = pd.Series(correlations, name='Correlation')
    print("Correlation results:")
    print(correlation_series)
    
    return correlation_series