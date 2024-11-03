from collections import Counter
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import KeyedVectors
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import warnings

# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def preprocess_text(text):
    """Tokenization and truncation."""
    inputs = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    return inputs

def classify_sentiment(df):
    """
    Classifies sentiment by combining multiple sentiment scores.
    """
    try:
        # Create copies of the input data to avoid warnings
        df = df.copy()
        
        sentiment_columns = [
            'afinn_sentiment',
            'bing_sentiment', 
            'nrc_sentiment',
            'sentiword_sentiment',
            'hu_sentiment'
        ]
        
        sd_columns = [
            'afinn_sd', 
            'bing_sd', 
            'nrc_sd', 
            'sentiword_sd', 
            'hu_sd'
        ]
        
        # Verify all required columns exist
        missing_cols = [col for col in sentiment_columns + sd_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # 1. Normalize scores to same scale (-1 to 1)
        normalized_scores = pd.DataFrame(index=df.index)
        for col in sentiment_columns:
            max_abs = df[col].abs().max()
            if max_abs != 0:
                normalized_scores[f'{col}_norm'] = df[col] / max_abs
            else:
                normalized_scores[f'{col}_norm'] = df[col]
        
        # 2. Calculate weights
        weights = pd.DataFrame(index=df.index)
        for score_col, sd_col in zip(sentiment_columns, sd_columns):
            sd_values = df[sd_col].fillna(df[sd_col].mean())
            weights[score_col] = 1 / (sd_values + 1e-6)
        
        # 3. Calculate weighted average sentiment
        final_score = pd.Series(0, index=df.index)
        total_weight = pd.Series(0, index=df.index)
        
        for col in sentiment_columns:
            norm_col = f'{col}_norm'
            weight = weights[col]
            final_score += normalized_scores[norm_col] * weight
            total_weight += weight
        
        final_score = final_score / total_weight
        
        # 4. Calculate agreement
        binary_sentiments = pd.DataFrame(index=df.index)
        for col in sentiment_columns:
            binary_sentiments[col] = np.where(df[col] > 0, 1, 0)
        agreement = (binary_sentiments.mean(axis=1) * 100).round(2)
        
        # 5. Add results to dataframe
        df['sentiment'] = np.where(final_score > 0, 'positive', 'negative')
        df['sentiment_confidence'] = final_score.abs()
        df['sentiment_agreement'] = agreement
        
        # 6. Create summary
        summary = {
            'total_speeches': len(df),
            'positive_speeches': (df['sentiment'] == 'positive').sum(),
            'negative_speeches': (df['sentiment'] == 'negative').sum(),
            'avg_confidence': df['sentiment_confidence'].mean(),
            'avg_agreement': agreement.mean(),
            'high_confidence_predictions': (df['sentiment_confidence'] > 0.7).sum()
        }
        
        return df, summary
        
    except Exception as e:
        print(f"Detailed error in sentiment classification: {str(e)}")
        return e

def generate_wordcloud(df, sentiment):
    """Generates and displays a word cloud for the specified sentiment."""
    words = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Most Common Words in {sentiment.capitalize()} Speeches')
    plt.show()

def ngram_analysis(df, sentiment, n=2, feature=None):
    """Performs n-gram analysis for the specified sentiment, optionally by a feature."""
    if feature:
        sentiment_texts = df[(df['sentiment'] == sentiment) & (df[feature].notnull())]['cleaned_text']
    else:
        sentiment_texts = df[df['sentiment'] == sentiment]['cleaned_text']
        
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams = vectorizer.fit_transform(sentiment_texts)
    ngram_counts = Counter(ngrams.toarray().sum(axis=0))
    ngram_features = vectorizer.get_feature_names_out()
    top_ngrams = sorted(zip(ngram_features, ngram_counts.values()), key=lambda x: x[1], reverse=True)[:10]
    for ngram, count in top_ngrams:
        print(f"{ngram}: {count}")
    return top_ngrams

def get_word_vectors(texts, model):
    """Generate average word vectors for each text using the Word2Vec model."""
    vectors = []
    for text in texts:
        words = text.split()
        vecs = [model[word] for word in words if word in model]  # Access vectors directly from KeyedVectors
        vectors.append(np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size))
    return np.array(vectors)

def train_sentiment_model_with_word2vec(df, text_column, label_column):
    """Trains a sentiment classification model using Word2Vec embeddings."""
    w2v_model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)    
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
    """Trains a sentiment classification model using DistilBert."""
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', num_labels=2)

    encodings = tokenizer(df[text_column].tolist(), truncation=True, padding=True, max_length=512)
    
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

    labels = df[label_column].apply(lambda x: 1 if x == 'positive' else 0).tolist()
    dataset = SentimentDataset(encodings, labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    model.save_pretrained("sentiment_model")

def generate_ngram_wordcloud(ngrams, title, filename=None):
    """
    Generate and save/display wordcloud for n-grams
    
    Args:
        ngrams: Counter object of n-grams and their frequencies
        title: Title for the wordcloud plot
        filename: Optional filename to save the plot
    """
    # Convert n-grams to space-separated strings for wordcloud
    text = {' '.join(k): v for k, v in ngrams.items()}
    
    # Create and generate a word cloud image
    wordcloud = WordCloud(
        width=1600, 
        height=800,
        background_color='white',
        max_words=50,
        collocations=False  # Important for n-grams
    ).generate_from_frequencies(text)
    
    # Display the word cloud
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20, pad=20)
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def analyze_and_visualize_ngrams(df, sentiment_type='positive', n_range=(2,3)):
    """
    Analyze and visualize n-grams for given sentiment type
    
    Args:
        df: DataFrame with speeches and sentiment
        sentiment_type: 'positive' or 'negative'
        n_range: tuple of (min_n, max_n) for n-gram analysis
    """
    results = {}
    
    for n in range(n_range[0], n_range[1] + 1):
        # Get n-grams
        ngrams = ngram_analysis(df, sentiment_type, n)
        results[n] = ngrams
        
        # Generate wordcloud
        title = f"Most Common {n}-grams in {sentiment_type.title()} Speeches"
        filename = f"wordcloud_{n}gram_{sentiment_type}.png"
        generate_ngram_wordcloud(ngrams, title, filename)
        
        # Also print top n-grams
        print(f"\nTop 10 {n}-grams for {sentiment_type} speeches:")
        for gram, count in ngrams.most_common(10):
            print(f"{''.join(gram)}: {count}")
    
    return results


def plot_ngram_analysis(ngram_data, title):
    """Plots the most common n-grams from the given ngram_data."""
    ngrams, counts = zip(*ngram_data)  # Unzip the ngrams and counts
    plt.figure(figsize=(10, 6))
    plt.barh(ngrams, counts, color='skyblue')
    plt.gca().invert_yaxis()  # Reverse the order so the highest frequency is at the top
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('N-grams')
    plt.show()



def filtered_ngram_analysis(df, sentiment, n=2, feature=None, filter_value=None, top_n=10):
    """
    Performs n-gram analysis on a filtered subset of data based on a specified feature and filter value.
    
    Parameters:
    - df: DataFrame containing the data
    - sentiment: The sentiment type ('positive' or 'negative') to filter by
    - n: The n-gram size (e.g., 2 for bigrams, 3 for trigrams)
    - feature: Optional feature to filter by (e.g., 'gender', 'party_group')
    - filter_value: The specific value of the feature to filter on (e.g., 'Male' for gender)
    - top_n: The number of top n-grams to return and plot
    
    Returns:
    - top_ngrams: List of top n-grams and their counts
    """
    try:
        # Filter by sentiment and the specified feature value
        if feature and filter_value:
          
            filtered_df = df[(df['sentiment'] == sentiment) & (df[feature] == filter_value)]
            logging.debug(f"Filtered DataFrame size: {filtered_df.shape[0]} rows")
        else:
            filtered_df = df[df['sentiment'] == sentiment]
            logging.debug(f"Filtered DataFrame size without feature filter: {filtered_df.shape[0]} rows")
        
        # Check if filtered DataFrame is empty
        if filtered_df.empty:
            logging.warning(f"No data found for sentiment='{sentiment}' with {feature}='{filter_value}'")
            return []
        
        # Perform n-gram analysis on the filtered data
        top_ngrams = ngram_analysis(filtered_df, sentiment, n)
        
        # Check if ngram_analysis returned empty results
        if not top_ngrams:
            logging.warning(f"No n-grams found for sentiment='{sentiment}' with {feature}='{filter_value}'")
            return []
        
        # Plot the results
        title = f"Top {n}-grams in {sentiment.capitalize()} Speeches ({feature} = {filter_value})" if feature else f"Top {n}-grams in {sentiment.capitalize()} Speeches"
        plot_ngram_analysis(top_ngrams, title)
        
        return top_ngrams
    
    except Exception as e:
        logging.error(f"Error in filtered_ngram_analysis: {e}")
        return []
    


def plot_most_common_words_with_filter(df, sentiment, feature=None, filter_value=None):
    """
    Plot the most common words for positive or negative speeches,
    optionally filtered by a specified feature and filter value.
    
    Parameters:
    - df: DataFrame containing the data
    - sentiment: The sentiment type ('positive' or 'negative') to filter by
    - feature: Optional feature to filter by (e.g., 'gender', 'party_group')
    - filter_value: The specific value of the feature to filter on (e.g., 'Male' for gender)
    """
    # Filter by sentiment and optional feature and filter value
    if feature and filter_value:
        filtered_df = df[(df['sentiment'] == sentiment) & (df[feature] == filter_value)]
    else:
        filtered_df = df[df['sentiment'] == sentiment]

    # Check if there's data to plot
    if filtered_df.empty:
        print(f"No data found for sentiment='{sentiment}' with {feature}='{filter_value}'")
        return

    # Concatenate all text in the filtered data and calculate word frequency
    text_data = filtered_df['cleaned_text']
    all_words = ' '.join(text_data)
    words = all_words.split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(10)

    # Prepare data for plotting
    words, counts = zip(*most_common)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    title = f"Most Common Words in {sentiment.capitalize()} Speeches. {feature} ({filter_value})"
    if feature and filter_value:
        title += f" ({feature} = {filter_value})"
    plt.title(title)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
