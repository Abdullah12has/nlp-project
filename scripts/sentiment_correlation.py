import seaborn as sns
import matplotlib.pyplot as plt
import logging

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


def calculate_and_plot_correlations(df, feature_list, sentiment_column):
    """Calculate correlations between specified features and sentiment column, and plot results."""
    correlation_results = {}
    
    # Calculate correlations
    logging.info("Calculating correlations...")
    for feature in feature_list:
        if feature in df.columns:
            correlation_results[feature] = df[sentiment_column].corr(df[feature])
    logging.info(f"Correlations calculated: {correlation_results}")

    # Plot correlation results
    plt.figure(figsize=(10, 6))
    plt.bar(correlation_results.keys(), correlation_results.values())
    plt.title('Correlation with Sentiment Score')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return correlation_results
# Example usage:
# plot_correlation_heatmap(df, ['afinn_sentiment', 'jockers_sentiment', 'nrc_sentiment'])
