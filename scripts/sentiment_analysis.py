from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def classify_sentiment(df, score_column):
    df['sentiment'] = df[score_column].apply(lambda x: 'positive' if x > 0 else 'negative')
    return df

def generate_wordcloud(df, sentiment):
    words = ' '.join(df[df['sentiment'] == sentiment]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Most Common Words in {sentiment} Speeches')
    plt.show()

def ngram_analysis(df, sentiment, n=2):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(n, n))
    sentiment_texts = df[df['sentiment'] == sentiment]['text']
    ngrams = vectorizer.fit_transform(sentiment_texts)
    ngram_counts = Counter(ngrams.toarray().sum(axis=0))
    ngram_features = vectorizer.get_feature_names_out()
    top_ngrams = sorted(zip(ngram_features, ngram_counts.values()), key=lambda x: x[1], reverse=True)[:10]
    for ngram, count in top_ngrams:
        print(f"{ngram}: {count}")

# Example usage:
# df = classify_sentiment(df, 'sentiment_score')
# generate_wordcloud(df, 'positive')
# ngram_analysis(df, 'positive', 2)
