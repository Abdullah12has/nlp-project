from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

def classify_sentiment(df, score_column):
    """Classifies the sentiment of the text based on the score column."""
    df['sentiment'] = df[score_column].apply(lambda x: 'positive' if x > 0 else 'negative')
    return df

def generate_wordcloud(df, sentiment):
    """Generates and displays a word cloud for the specified sentiment."""
    words = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Most Common Words in {sentiment.capitalize()} Speeches')
    plt.show()

def ngram_analysis(df, sentiment, n=2):
    """Performs n-gram analysis for the specified sentiment."""
    vectorizer = CountVectorizer(ngram_range=(n, n))
    sentiment_texts = df[df['sentiment'] == sentiment]['cleaned_text']
    ngrams = vectorizer.fit_transform(sentiment_texts)
    ngram_counts = Counter(ngrams.toarray().sum(axis=0))
    ngram_features = vectorizer.get_feature_names_out()
    top_ngrams = sorted(zip(ngram_features, ngram_counts.values()), key=lambda x: x[1], reverse=True)[:10]
    for ngram, count in top_ngrams:
        print(f"{ngram}: {count}")

def get_word_vectors(texts, model):
    """Generate average word vectors for each text using the Word2Vec model."""
    vectors = []
    for text in texts:
        words = text.split()
        vecs = [model.wv[word] for word in words if word in model.wv]
        vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size))
    return np.array(vectors)

def train_sentiment_model_with_word2vec(df, text_column, label_column):
    """Trains a sentiment classification model using Word2Vec embeddings."""
    w2v_model = Word2Vec.load("models/GoogleNews-vectors-negative300.bin")  
    X_train, X_test, y_train, y_test = train_test_split(df[text_column], df[label_column], test_size=0.3, random_state=42)
    X_train_vec = get_word_vectors(X_train, w2v_model)
    X_test_vec = get_word_vectors(X_test, w2v_model)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
def plot_most_common_words(df, sentiment):
    """Plot the most common words for positive or negative speeches."""
    text_data = df[df['sentiment'] == sentiment]['cleaned_text']
    all_words = ' '.join(text_data)
    words = all_words.split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(10)

    words, counts = zip(*most_common)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.title(f'Most Common Words in {sentiment.capitalize()} Speeches')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
    
def calculate_correlation(df, feature_list, sentiment_column):
    """Calculate correlations and plot distributions."""
    correlations = df[feature_list + [sentiment_column]].corr()

    for feature in feature_list:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x=feature, hue=sentiment_column, multiple="stack", bins=30)
        plt.title(f'Distribution of {sentiment_column} by {feature}')
        plt.show()

    return correlations

def train_sentiment_model_with_bert(df, text_column, label_column):
    """Trains a sentiment classification model using BERT."""
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Tokenize the input text
    encodings = tokenizer(df[text_column].tolist(), truncation=True, padding=True, max_length=128)
    
    # Create PyTorch dataset
    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)

    # Create dataset
    labels = df[label_column].apply(lambda x: 1 if x == 'positive' else 0).tolist()
    dataset = SentimentDataset(encodings, labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()

    # Save the trained model
    model.save_pretrained("sentiment_model")

# Example usage:
# df = pd.read_csv("path/to/your/data/senti_df.csv")  # Load your DataFrame
# df = classify_sentiment(df, 'sentiment_score')
# generate_wordcloud(df, 'positive')
# ngram_analysis(df, 'positive', 2)
# train_sentiment_model_with_word2vec(df, 'cleaned_text', 'sentiment')
# train_sentiment_model_with_bert(df, 'cleaned_text', 'sentiment')
