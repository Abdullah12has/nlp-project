import logging
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os


def get_most_probable_topic(lda_model, doc, dictionary):
    """Get the most probable topic for a document."""
    bow = dictionary.doc2bow(doc.split())
    topics = lda_model.get_document_topics(bow)
    return max(topics, key=lambda x: x[1])[0]

def extract_topic_names(lda_model, num_words=5):
    """Extracts topic names based on the top words for each topic."""
    topic_names = {}
    for topic_id in range(lda_model.num_topics):
        # Get top words for the topic
        top_words = lda_model.show_topic(topic_id, topn=num_words)
        # Join the words to create a name
        topic_names[topic_id] = " ".join([word for word, prob in top_words])
    return topic_names

def train_and_visualize_lda(df, text_column='cleaned_text', time_column='year', num_topics=5, passes=5, output_dir='outputT6'):
    """
    Train a dynamic LDA model and visualize topics over time.
    
    Parameters:
    - df: DataFrame containing text data.
    - text_column: Name of the column with cleaned text.
    - time_column: Name of the column with time data (e.g., year).
    - num_topics: Number of topics for the LDA model.
    - passes: Number of passes for LDA training.
    - output_dir: Directory to save the visualization output.

    Returns:
    - lda_model: Trained LDA model.
    - df: DataFrame with topic assignments added.
    - topic_names: Dictionary mapping topic IDs to representative topic names.
    """
    # Preprocess text for LDA, removing empty or null values
    df = df.dropna(subset=[text_column])
    texts = df[text_column].apply(lambda x: x.split())
    
    # Prepare dictionary and corpus
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    
    # Visualize LDA topics using pyLDAvis
    os.makedirs(output_dir, exist_ok=True)
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, os.path.join(output_dir, 'lda_dynamic_visualization.html'))
    
    # Assign topics to each document
    df['lda_topic'] = df[text_column].apply(lambda doc: get_most_probable_topic(lda_model, doc, dictionary))
    
    # Extract topic names for each topic
    topic_names = extract_topic_names(lda_model)
    
    return lda_model, df, topic_names


def train_and_visualize_bertopic(df, text_column='cleaned_text', time_column='year', min_topic_size=5, output_dir='outputT6'):
    """
    Train a BERTopic model using year-based time evolution and visualize topic trends over time.
    
    Parameters:
    - df: DataFrame with text and time data.
    - text_column: Column containing text data.
    - time_column: Column containing yearly time data.
    - min_topic_size: Minimum size of topics.
    - output_dir: Directory to save the visualization output.
    
    Returns:
    - bertopic_model: Trained BERTopic model.
    - df: DataFrame with topic assignments added.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare timestamps and text data for BERTopic
    timestamps = df[time_column].to_list()
    documents = df[text_column].tolist()

    # Initialize and train the BERTopic model with minimum topic size
    bertopic_model = BERTopic(min_topic_size=min_topic_size, verbose=True)
    topics, probs = bertopic_model.fit_transform(documents)

    # Create topics over time
    topics_over_time = bertopic_model.topics_over_time(documents, timestamps)

    # Assign topics to the DataFrame
    df['bertopic_topic'] = topics

    # Visualize topic trends over time
    topic_trends_fig = bertopic_model.visualize_topics_over_time(topics_over_time)
    topic_trends_fig.write_html(os.path.join(output_dir, 'bertopic_topic_trends_over_time.html'))

    with open(os.path.join(output_dir, 'bertopic_topics.txt'), 'w') as file:
        for topic_id in bertopic_model.get_topic_info().Topic:
            if topic_id != -1:  # Exclude outliers
                topic_words = bertopic_model.get_topic(topic_id)
                topic_words_str = ", ".join([word for word, _ in topic_words])
                file.write(f"Topic {topic_id}: {topic_words_str}\n\n")
    
    return bertopic_model, df





def get_most_probable_topic(lda_model, document, dictionary):
    """
    Assign the most probable topic to a document.
    
    Parameters:
    - lda_model: Trained LDA model.
    - document: List of tokens in a document.
    - dictionary: Gensim dictionary for the corpus.
    
    Returns:
    - The most probable topic index for the document.
    """
    bow = dictionary.doc2bow(document.split())
    topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0.0)
    return max(topic_distribution, key=lambda x: x[1])[0]