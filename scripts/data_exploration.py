import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_feature_distribution(data):
    # Convert columns to datetime if needed
    data['speech_date'] = pd.to_datetime(data['speech_date'], errors='coerce')
    data['house_start_date'] = pd.to_datetime(data['house_start_date'], errors='coerce')
    data['date_of_birth'] = pd.to_datetime(data['date_of_birth'], errors='coerce')

    # Sentiment fields for analysis
    sentiment_fields = ['afinn_sentiment', 'bing_sentiment', 'nrc_sentiment', 'sentiword_sentiment', 'hu_sentiment']

    # 1. Distribution of Speeches by Year
    plt.figure(figsize=(10, 6))
    sns.histplot(data['year'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Speeches by Year')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.show()

    # 2. Gender Distribution
    plt.figure(figsize=(6, 6))
    gender_counts = data['gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff6666'])
    plt.title('Gender Distribution')
    plt.show()

    # 3. Distribution by Party Group
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, y='party_group', order=data['party_group'].value_counts().index, palette='viridis')
    plt.title('Distribution by Party Group')
    plt.xlabel('Count')
    plt.ylabel('Party Group')
    plt.show()

    # 4. Speech Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['word_count'], bins=30, kde=True)
    plt.title('Distribution of Speech Length (Word Count)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.show()

    # 5. Sentiment Score Distributions
    for sentiment in sentiment_fields:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[sentiment].dropna(), kde=True, bins=20)
        plt.title(f'{sentiment} Distribution')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.show()

    # 6. Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['age'].dropna(), bins=30, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    # 7. Sentiment by Party Group
    for sentiment in sentiment_fields:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x='party_group', y=sentiment, palette='viridis')
        plt.xticks(rotation=45)
        plt.title(f'{sentiment} by Party Group')
        plt.xlabel('Party Group')
        plt.ylabel('Sentiment Score')
        plt.show()

    # 8. Speech Length by Party Group
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='party_group', y='word_count', palette='pastel')
    plt.title('Speech Length by Party Group')
    plt.xlabel('Party Group')
    plt.ylabel('Word Count')
    plt.xticks(rotation=45)
    plt.show()

    # 9. Gender and Sentiment Comparison
    for sentiment in sentiment_fields:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x='gender', y=sentiment, palette='coolwarm')
        plt.title(f'{sentiment} by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Sentiment Score')
        plt.show()

    # 10. Age and Sentiment Relationship
    for sentiment in sentiment_fields:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='age', y=sentiment, hue='gender', palette='Set1', alpha=0.6)
        plt.title(f'Age vs. {sentiment}')
        plt.xlabel('Age')
        plt.ylabel('Sentiment Score')
        plt.show()
    

# Example usage:
# df = load_data('data/data.csv')
# plot_feature_distribution(df, 'year')
