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
def plot_correlation_heatmap(df, sentiment_columns):
    correlation_matrix = df[sentiment_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Sentiment Scores')
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

def plot_sentiment_distribution(df, feature, sentiment_column='sentiment'):
    """Plot distribution of sentiment scores for different categories of a feature."""
    plt.figure(figsize=(14, 7))

    if pd.api.types.is_numeric_dtype(df[feature]):
        # Plot distribution for numerical features
        sns.histplot(data=df, x=feature, hue=sentiment_column, kde=True)
        plt.title(f'Sentiment Distribution by {feature}')
    else:
        # Plot distribution for categorical features
        sns.countplot(data=df, x=feature, hue=sentiment_column)
        plt.title(f'Sentiment Counts by {feature}')

    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
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

def calculate_and_plot_correlations(df, feature_list, sentiment_column='sentiment_confidence'):
    """Calculate correlations between specified features and sentiment column, and plot results."""
    correlation_results = {}

    # Drop NaN values to avoid calculation errors
    df = df.dropna(subset=feature_list + [sentiment_column])

    # Drop low variance features
    # df = drop_low_variance_features(df)

    logging.info("Calculating correlations...")

    for feature in feature_list:
        if feature in df.columns:
            # Check if the feature has variance
            if df[feature].nunique() > 1:  # Ensure there's more than one unique value
                if pd.api.types.is_numeric_dtype(df[feature]):
                    # Direct correlation for numerical data
                    correlation_results[feature] = df[sentiment_column].corr(df[feature])
                else:
                    # Encode categorical features and calculate correlation
                    encoded_feature = pd.Categorical(df[feature]).codes
                    correlation_results[feature] = df[sentiment_column].corr(pd.Series(encoded_feature))
            else:
                logging.warning(f"Feature '{feature}' has insufficient variance for correlation.")
                correlation_results[feature] = np.nan  # No correlation possible
        else:
            logging.warning(f"Feature '{feature}' is not present in the DataFrame.")

    logging.info(f"Correlations calculated: {correlation_results}")

    # Plot correlation results as a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(correlation_results.keys(), correlation_results.values(), color='skyblue')
    plt.title('Correlation of Features with Sentiment Confidence')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return correlation_results
