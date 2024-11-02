from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_lda_model(df, text_column):
    """Train an LDA model with hyperparameter tuning and visualization."""
    # Prepare the data
    texts = df[text_column].apply(lambda x: x.split())
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Hyperparameter tuning for LDA
    best_model = None
    best_coherence = 0

    for num_topics in range(2, 10):  # Adjust range as needed
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()

        if coherence > best_coherence:
            best_coherence = coherence
            best_model = lda_model

    print(f"Best LDA model with coherence score of {best_coherence:.4f}")

    # Visualization
    vis = pyLDAvis.gensim_models.prepare(best_model, corpus, dictionary)
    return best_model, dictionary, corpus, vis

def train_bertopic_model(documents):
    # Initialize the model without unsupported arguments
    topic_model = BERTopic()  # Remove n_topics if unsupported
    topics, probs = topic_model.fit_transform(documents)
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
        for topic in range(topic_model.num_topics):
            words = topic_model.get_topic(topic)
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
    topic_counts = df.groupby([time_column, topic_column]).size().unstack().fillna(0)

    plt.figure(figsize=(15, 8))
    for topic in topic_counts.columns:
        plt.plot(topic_counts.index, topic_counts[topic], label=f'Topic {topic}')

    plt.title('Topic Trends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Speeches')
    plt.legend(title='Topics')
    plt.tight_layout()
    plt.show()
