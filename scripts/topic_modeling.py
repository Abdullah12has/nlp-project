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



def save_checkpoint(model, dictionary, corpus, num_topics, coherence, checkpoint_path):
    """Save checkpoint for the model."""
    checkpoint = {
        'model': model,
        'dictionary': dictionary,
        'corpus': corpus,
        'num_topics': num_topics,
        'coherence': coherence
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(checkpoint_path):
    """Load checkpoint if available."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None

def extract_topics(lda_model, num_words=10):
    """Extract topics from LDA model as lists of words with their probabilities."""
    topics = []
    for topic_id, topic in lda_model.show_topics(formatted=False, num_words=num_words):
        topic_words = {word: prob for word, prob in topic}
        topics.append((topic_id, topic_words))
    return topics

def train_lda_model(
    df: pd.DataFrame, 
    text_column: str, 
    min_topics: int = 2, 
    max_topics: int = 10, 
    passes: int = 20,
    checkpoint_path="progress/lda_checkpoint.pkl"
):
    """
    Train an optimized LDA model with coherence-based hyperparameter tuning 
    and generate visualization with pyLDAvis, using savepoints.

    Args:
        df (pd.DataFrame): DataFrame containing text data.
        text_column (str): Column name for preprocessed text.
        min_topics (int): Minimum number of topics for tuning.
        max_topics (int): Maximum number of topics for tuning.
        passes (int): Number of passes through the corpus during training.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        tuple: (best_model, dictionary, corpus, vis, topics) with best model, dictionary, 
               corpus, visualization object, and list of topics.
    """
    
    # Prepare the data
    texts = df[text_column].apply(lambda x: x.split())
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Load checkpoint if available
    checkpoint = load_checkpoint(checkpoint_path)
    best_model = checkpoint['model'] if checkpoint else None
    best_coherence = checkpoint['coherence'] if checkpoint else 0
    best_num_topics = checkpoint['num_topics'] if checkpoint else 0

    # Hyperparameter tuning for LDA
    for num_topics in range(min_topics, max_topics + 1):
        if checkpoint and num_topics <= best_num_topics:
            # Skip already trained models if resuming from checkpoint
            continue

        lda_model = LdaModel(
            corpus=corpus, 
            num_topics=num_topics, 
            id2word=dictionary, 
            passes=passes, 
            random_state=42
        )
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()

        if coherence > best_coherence:
            best_coherence = coherence
            best_model = lda_model
            best_num_topics = num_topics
            save_checkpoint(best_model, dictionary, corpus, best_num_topics, best_coherence, checkpoint_path)

    print(f"Best LDA model with {best_num_topics} topics and coherence score of {best_coherence:.4f}")

    # Extract topics
    topics = extract_topics(best_model)

    # Visualization
    vis = None
    try:
        vis = pyLDAvis.gensim_models.prepare(best_model, corpus, dictionary)
    except Exception as e:
        print(f"Visualization failed: {e}")

    return best_model, dictionary, corpus, vis, topics





def save_bertopic_checkpoint(model, path="progress/bertopic_checkpoint.pkl"):
    """Save the BERTopic model to a checkpoint."""
    model.save(path)
    logging.info(f"BERTopic model checkpoint saved at {path}")

def load_bertopic_checkpoint(path="progress/bertopic_checkpoint.pkl"):
    """Load BERTopic model checkpoint if available."""
    if os.path.exists(f"{path}.pkl"):
        model = BERTopic.load(path)
        logging.info(f"Loaded BERTopic model checkpoint from {path}")
        return model
    return None


def train_bertopic_model(documents, checkpoint_path="progress/bertopic_checkpoint.pkl", min_topic_size=10):
    """
    Train a BERTopic model with savepoint functionality.
    
    Args:
        documents (list of str): The documents to be used for topic modeling.
        checkpoint_path (str): Path to save/load model checkpoints.
        min_topic_size (int): Minimum topic size for BERTopic model.
    
    Returns:
        tuple: (model, topics, probabilities)
    """
    # Load checkpoint if available
    topic_model = load_bertopic_checkpoint(checkpoint_path)
    
    # Train if no checkpoint exists
    if not topic_model:
        logging.info("Initializing and training BERTopic model...")
        topic_model = BERTopic(min_topic_size=min_topic_size, verbose=True)  # Set other hyperparameters as needed
        topics, probs = topic_model.fit_transform(documents)
        save_bertopic_checkpoint(topic_model, checkpoint_path)
    else:
        logging.info("BERTopic model checkpoint found, skipping training.")
        topics, probs = topic_model.transform(documents)  # Transform if already trained

    return topic_model, topics, probs

def analyze_topic_distribution_with_representation(df, topic_column='topic', group_columns=['party_group', 'speaker'], topic_model=None):
    """Analyze topic distribution across specified groups."""
    for group_col in group_columns:
        if group_col in df.columns:
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, x=topic_column, hue=group_col)
            plt.title(f'Topic Distribution by {group_col.capitalize()}')
            plt.xticks(rotation=45)
            plt.legend(title=group_col)
            plt.tight_layout()
            plt.show()

    # Display topics with their top words
    if topic_model:
        num_topics = len(topic_model.get_topics())  # Get the number of topics
        for topic in range(num_topics):
            words = topic_model.get_topic(topic)
            if words:  # Check if the topic has any words
                print(f"Topic {topic}: {', '.join([word for word, _ in words])}")
            
    

def topic_evolution_over_time(df, topic_column='topic', time_column='year'):
    """Analyzes how topics evolve over time."""
    # Group by time and topic, counting occurrences
    topic_trends = df.groupby([time_column, topic_column]).size().unstack(fill_value=0)

    # Plotting the evolution of topics over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=topic_trends.T)
    plt.title('Topic Evolution Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Speeches')
    plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualize_topic_trends(df, topic_column, time_column):
    """Visualize topic trends over time."""
    topic_counts = df.groupby([time_column, topic_column]).size().unstack(fill_value=0)

    plt.figure(figsize=(15, 8))
    for topic in topic_counts.columns:
        plt.plot(topic_counts.index, topic_counts[topic], label=f'Topic {topic}')

    plt.title('Topic Trends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Speeches')
    plt.legend(title='Topics')
    plt.tight_layout()
    plt.show()

def train_bertopic_model_over_time(documents, checkpoint_path="progress/bertopic_checkpoint.pkl", min_topic_size=10):
    """Train a BERTopic model with savepoint functionality."""
    topic_model = load_bertopic_checkpoint(checkpoint_path)
    
    if not topic_model:
        logging.info("Initializing and training BERTopic model...")
        topic_model = BERTopic(min_topic_size=min_topic_size, verbose=True)
        topics, probs = topic_model.fit_transform(documents)
        save_bertopic_checkpoint(topic_model, checkpoint_path)
    else:
        logging.info("BERTopic model checkpoint found, skipping training.")
        topics, probs = topic_model.transform(documents)

    return topic_model, topics, probs

def visualize_topic_trends_over_time(topic_trends, title='Topic Trends Over Time'):
    """Visualize the topic trends over time."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=topic_trends)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Number of Documents')
    plt.legend(title='Topics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_topic_evolution(df, topic_column='topic', time_column='year'):
    """Analyze topic distribution over time."""
    topic_counts = df.groupby([time_column, topic_column]).size().reset_index(name='counts')
    topic_trends = topic_counts.pivot(index=time_column, columns=topic_column, values='counts').fillna(0)
    return topic_trends

def train_dynamic_lda_model(df, text_column, time_column, num_topics=5, passes=15):
    """Train a dynamic LDA model."""
    texts = df[text_column].apply(lambda x: x.split())
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    return lda_model, vis

def get_lda_topic_assignments(lda_model, documents):
    """
    Assign the most probable topic to each document based on an LDA model.

    Parameters:
    - lda_model: Trained LDA model.
    - documents: List or series of preprocessed documents.

    Returns:
    - A list of topic assignments for each document.
    """
    topic_assignments = []
    for doc in documents:
        bow = lda_model.id2word.doc2bow(doc.split())  # Convert document to bag-of-words format
        topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0.0)
        most_probable_topic = max(topic_distribution, key=lambda x: x[1])[0]  # Select topic with highest probability
        topic_assignments.append(most_probable_topic)
    
    return topic_assignments