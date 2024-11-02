# sentiment_analysis.py

from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def classify_sentiment(df, score_column):
    """
    Classifies the sentiment of the text based on the score column.
    Assumes positive if score > 0, negative otherwise.
    """
    df['sentiment'] = df[score_column].apply(lambda x: 'positive' if x > 0 else 'negative')
    return df

def generate_wordcloud(df, sentiment):
    """
    Generates and displays a word cloud for the specified sentiment.
    """
    words = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])  # Use 'cleaned_text' for preprocessed text
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Most Common Words in {sentiment} Speeches')
    plt.show()

def ngram_analysis(df, sentiment, n=2):
    """
    Performs n-gram analysis for the specified sentiment.
    """
    vectorizer = CountVectorizer(ngram_range=(n, n))
    sentiment_texts = df[df['sentiment'] == sentiment]['cleaned_text']  # Use 'cleaned_text'
    ngrams = vectorizer.fit_transform(sentiment_texts)
    ngram_counts = Counter(ngrams.toarray().sum(axis=0))
    ngram_features = vectorizer.get_feature_names_out()
    top_ngrams = sorted(zip(ngram_features, ngram_counts.values()), key=lambda x: x[1], reverse=True)[:10]
    for ngram, count in top_ngrams:
        print(f"{ngram}: {count}")

def get_word_vectors(texts, model):
    """
    Generate average word vectors for each text using the Word2Vec model.
    """
    vectors = []
    for text in texts:
        words = text.split()
        vecs = [model.wv[word] for word in words if word in model.wv]
        vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size))
    return np.array(vectors)

def train_sentiment_model_with_word2vec(df, text_column, label_column):
    """
    Trains a sentiment classification model using Word2Vec embeddings.
    """
    # Load a pre-trained Word2Vec model (adjust the path as needed)
    w2v_model = Word2Vec.load("models\GoogleNews-vectors-negative300.bin")  

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[text_column], df[label_column], test_size=0.3, random_state=42)
    
    # Generate word vectors for training and testing sets
    X_train_vec = get_word_vectors(X_train, w2v_model)
    X_test_vec = get_word_vectors(X_test, w2v_model)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)

    # Print accuracy and classification report
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Example usage:
# df = pd.read_csv("path/to/your/data/senti_df.csv")  # Load your DataFrame
# df = classify_sentiment(df, 'sentiment_score')
# generate_wordcloud(df, 'positive')
# ngram_analysis(df, 'positive', 2)
# train_sentiment_model_with_word2vec(df, 'cleaned_text', 'sentiment')
