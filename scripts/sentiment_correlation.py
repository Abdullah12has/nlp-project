import seaborn as sns
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Ensure logging is configured
logging.basicConfig(level=logging.INFO)

def encode_categorical_features(df, categorical_features):
    """Encode categorical features for correlation analysis."""
    for feature in categorical_features:
        if feature in df.columns and df[feature].notnull().all():
            df[feature] = pd.Categorical(df[feature]).codes
    return df

def plot_sentiment_correlation_heatmap(df):
    """
    Plots a correlation heatmap for the top 50 values of various sentiment columns and 'sentiment_confidence'.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the sentiment columns and 'sentiment_confidence'.
    """
    # List of required columns
    sentiment_columns = [
        'afinn_sentiment', 'bing_sentiment', 'nrc_sentiment', 
        'sentiword_sentiment', 'hu_sentiment', 'sentiment_confidence'
    ]
    
    # Ensure all required columns exist in the DataFrame
    missing_cols = [col for col in sentiment_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")
    
    # Select top 50 rows based on 'sentiment_confidence' (or another column if preferred)
    top_50_df = df.nlargest(50, 'sentiment_confidence')
    
    # Calculate the correlation matrix for the top 50 rows
    correlation_matrix = top_50_df[sentiment_columns].corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))  # Adjust figure size for readability
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap ')
    plt.show()




def correlate_sentiment_with_topics(df, sentiment_column='sentiment_score', topic_column='topic'):
    """
    Correlate sentiment scores with topics and visualize the results.
    
    Args:
        df (pd.DataFrame): DataFrame containing sentiment and topic information.
        sentiment_column (str): Name of the column with sentiment scores.
        topic_column (str): Name of the column with topic information.
    """
    # Grouping by topic and calculating mean sentiment
    sentiment_by_topic = df.groupby(topic_column)[sentiment_column].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sentiment_by_topic, x=topic_column, y=sentiment_column)
    plt.title('Mean Sentiment Score by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Mean Sentiment Score')
    plt.xticks(rotation=45)
    plt.show()
    

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