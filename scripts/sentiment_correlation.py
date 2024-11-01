import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df, sentiment_columns):
    correlation_matrix = df[sentiment_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Sentiment Scores')
    plt.show()

# Example usage:
# plot_correlation_heatmap(df, ['afinn_sentiment', 'jockers_sentiment', 'nrc_sentiment'])
