# topic_modeling.py
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns

def train_lda_model(df, text_column):
    # Prepare data for LDA
    texts = [text.split() for text in df[text_column]]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary, passes=15)
    return lda_model, dictionary, corpus

def train_bertopic_model(df, text_column):
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(df[text_column])
    return topic_model

def analyze_topic_distribution_with_representation(df, topic_column='topic', group_columns=['party_group', 'speaker'], topic_model=None):
    # Analyze distribution
    for group_col in group_columns:
        if group_col in df.columns:
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, x=topic_column, hue=group_col)
            plt.title(f'Topic Distribution by {group_col.capitalize()}')
            plt.xticks(rotation=45)
            plt.show()

    # Display topics with their top words
    if topic_model:
        for topic in range(topic_model.num_topics):
            words = topic_model.get_topic(topic)
            print(f"Topic {topic}: {', '.join([word for word, _ in words])}")
            
            
def topic_evolution_over_time(df, topic_column='topic', time_column='year'):
    """
    Analyzes how topics evolve over time.
    
    Args:
        df (pd.DataFrame): DataFrame containing topic and time information.
        topic_column (str): Name of the column with topic information.
        time_column (str): Name of the column with time information (e.g., year).
    """
    # Group by time and topic, counting occurrences
    topic_trends = df.groupby([time_column, topic_column]).size().unstack(fill_value=0)
    
    # Plotting the evolution of topics over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=topic_trends.T)
    plt.title('Topic Evolution Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Speeches')
    plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# Example usage:
# lda_model, dictionary, corpus = train_lda_model(df, 'text')
# topic_model = train_bertopic_model(df, 'text')
