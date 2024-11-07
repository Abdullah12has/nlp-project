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
from sklearn.metrics.pairwise import cosine_similarity


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

def analyze_topic_distribution_with_representation(df, top_topic_count=10, num_parties=10, num_speakers=10):
    """
    Analyze topic distributions across political parties and speakers and generate visualizations.
    
    Args:
        df (DataFrame): The DataFrame containing the topics, party, and speaker data.
        top_topic_count (int or None): The number of top topics to display in the graph legend. 
                                      If None, show all topics in the legend.
        num_parties (int): The number of top political parties to display.
        num_speakers (int): The number of top speakers to display.
    """
    try:
        logging.info("Analyzing topic distributions across political parties and speakers...")

        # Ensure topic column exists
        if 'bertopic_topic' not in df.columns:
            logging.error("Topic column not found in the dataframe.")
            raise ValueError("Topic column must exist in the dataframe. Please ensure topic modeling has been performed.")

        # Remove topic -1 from the data (outlier or unassigned topic)
        df = df[df['bertopic_topic'] != -1]

        # Party-based topic distribution
        party_topic_distribution = df.groupby('party')['bertopic_topic'].value_counts(normalize=True).unstack().fillna(0)
        logging.info(f"Topic distribution across political parties:\n{party_topic_distribution}")

        # Speaker-based topic distribution
        speaker_topic_distribution = df.groupby('proper_name')['bertopic_topic'].value_counts(normalize=True).unstack().fillna(0)
        logging.info(f"Topic distribution across speakers:\n{speaker_topic_distribution}")

        # Get top topics for each party
        if top_topic_count is not None and top_topic_count > 0:
            # Filter and select only the top N topics for parties
            party_top_topics = party_topic_distribution.apply(lambda x: x.nlargest(top_topic_count), axis=1)
            logging.info(f"Top {top_topic_count} topics for each political party:\n{party_top_topics}")
        else:
            party_top_topics = party_topic_distribution

        # Get top topics for each speaker
        if top_topic_count is not None and top_topic_count > 0:
            # Filter and select only the top N topics for speakers
            speaker_top_topics = speaker_topic_distribution.apply(lambda x: x.nlargest(top_topic_count), axis=1)
            logging.info(f"Top {top_topic_count} topics for each speaker:\n{speaker_top_topics}")
        else:
            speaker_top_topics = speaker_topic_distribution

        # Remove rows/columns with all NaN values
        party_top_topics = party_top_topics.dropna(how='all', axis=0)
        speaker_top_topics = speaker_top_topics.dropna(how='all', axis=0)

        # Get the top N political parties and speakers based on the distribution
        top_parties = party_top_topics.sum(axis=1).nlargest(num_parties).index
        party_top_topics = party_top_topics.loc[top_parties]

        top_speakers = speaker_top_topics.sum(axis=1).nlargest(num_speakers).index
        speaker_top_topics = speaker_topic_distribution.loc[top_speakers]

        # Generate a custom color palette for topics, ensuring each topic has a unique color
        # For Party Graph: Calculate unique topics in party distribution
        if top_topic_count is None:
            unique_topics_party = party_top_topics.columns
        else:
            unique_topics_party = party_top_topics.columns[:top_topic_count]

        # For Speaker Graph: Calculate unique topics in speaker distribution
        if top_topic_count is None:
            unique_topics_speaker = speaker_top_topics.columns
        else:
            unique_topics_speaker = speaker_top_topics.columns[:top_topic_count]

        # Ensure color consistency across both graphs (same number of topics in both)
        num_topics_party = len(unique_topics_party)
        num_topics_speaker = len(unique_topics_speaker)

        # Using a colormap that can handle the maximum number of topics across both graphs
        max_topics = max(num_topics_party, num_topics_speaker)
        color_palette = plt.cm.get_cmap('tab20', max_topics)  # Adjust colormap if necessary
        colors = [color_palette(i) for i in range(max_topics)]

        # Visualization - Topic distribution across political parties
        ax_party = party_top_topics.plot(kind='bar', stacked=True, figsize=(12, 8), color=colors[:num_topics_party])
        plt.title(f'Topic Distribution Across Top {num_parties} Political Parties')
        plt.xlabel('Political Party')
        plt.ylabel('Topic Proportion')
        plt.tight_layout()

        # Get the handles and labels for the legend
        handles, labels = ax_party.get_legend_handles_labels()

        # Display only the top 'top_topic_count' topics in the legend
        if top_topic_count is not None:
            ax_party.legend(handles[:top_topic_count], labels[:top_topic_count], title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        else:
            # Display all topics in the legend if top_topic_count is None
            ax_party.legend(handles, labels, title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

        plt.savefig("graphs/topic_distribution_parties.png", bbox_inches='tight')
        plt.show()

        # Visualization - Topic distribution across speakers (Limit to top N speakers)
        if not speaker_top_topics.empty:
            ax_speaker = speaker_top_topics.plot(kind='bar', stacked=True, figsize=(12, 8), color=colors[:num_topics_speaker])
            plt.title(f'Topic Distribution Across Top {num_speakers} Speakers')
            plt.xlabel('Speaker')
            plt.ylabel('Topic Proportion')
            plt.tight_layout()

            # Get the handles and labels for the legend
            handles, labels = ax_speaker.get_legend_handles_labels()

            # Display only the top 'top_topic_count' topics in the legend
            if top_topic_count is not None:
                ax_speaker.legend(handles[:top_topic_count], labels[:top_topic_count], title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            else:
                # Display all topics in the legend if top_topic_count is None
                ax_speaker.legend(handles, labels, title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

            plt.savefig("graphs/topic_distribution_speakers.png", bbox_inches='tight')
            plt.show()
        else:
            logging.warning("No topic distribution to visualize for speakers.")

        logging.info(f"Topic distribution analysis across political parties and speakers completed!")

    except Exception as e:
        logging.error(f"Error during topic distribution analysis: {e}")
            
    

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

def visualize_topic_trends(df, topic_column, time_column, topic_names=None):
    """
    Visualize topic trends over time with topic names.
    
    Parameters:
    - df: DataFrame with topics and time columns.
    - topic_column: Column containing topic IDs.
    - time_column: Column containing the time (e.g., year).
    - topic_names: Dictionary mapping topic IDs to topic names (e.g., {0: "Economy", 1: "Health"}).
    """
    
    # Count the occurrences of each topic per time unit
    topic_counts = df.groupby([time_column, topic_column]).size().unstack(fill_value=0)

    plt.figure(figsize=(15, 8))
    
    # Plot each topic trend
    for topic in topic_counts.columns:
        # Use topic name if available; otherwise, use topic ID
        label = topic_names.get(topic, f'Topic {topic}') if topic_names else f'Topic {topic}'
        plt.plot(topic_counts.index, topic_counts[topic], label=label)

    # Plot formatting
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

def visualize_topic_trends_over_time(topic_trends, title='Topic Trends Over Time', save_path='data/topic_trends_over_time.png'):
    """Save the topic trends over time plot to a file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=topic_trends)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Number of Documents')
    plt.legend(title='Topics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()  # Close the plot to prevent it from displaying
    print(f"Plot saved to '{save_path}'")

    
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


def extract_representative_words(df, model, n_words=10):
    """
    Extract the top representative words for each topic directly from the BERTopic model.
    
    :param df: DataFrame containing topics assigned by the model (e.g., 'bertopic_topic' column)
    :param model: Fitted BERTopic model
    :param n_words: Number of representative words to extract per topic
    :return: DataFrame with topic number and top words
    """
    topic_words = []
    
    # Ensure 'bertopic_topic' is in the dataframe (topics assigned to each document)
    if 'bertopic_topic' not in df.columns:
        raise ValueError("DataFrame must contain a 'bertopic_topic' column")
    
    # Extract the topics using BERTopic model
    topics = model.get_topics()
    
    # For each topic, get the top n_words (represented by words with highest probabilities)
    for topic_num, words in topics.items():
        top_words = [word for word, _ in sorted(words, key=lambda x: x[1], reverse=True)[:n_words]]
        topic_words.append((topic_num, top_words))
    
    return pd.DataFrame(topic_words, columns=['Topic', 'Top Words'])

def automate_topic_labeling(representative_words, predefined_labels):
    """
    Automate labeling of topics by matching representative words to predefined labels.
    
    :param representative_words: DataFrame containing topics and their representative words
    :param predefined_labels: Dictionary with predefined labels for topics (optional)
    :return: DataFrame with topics labeled based on similarity to predefined labels
    """
    # if predefined_labels is None:
    #     predefined_labels = {
    #         'Economy': ['finance', 'economy', 'tax', 'budget'],
    #         'Health': ['health', 'hospital', 'medicine', 'treatment'],
    #         'Environment': ['climate', 'pollution', 'environment', 'sustainability']
    #     }
    
    labeled_topics = []
    
    for _, row in representative_words.iterrows():
        topic = row['Topic']
        top_words = row['Top Words']
        
        # Join the topic's top words into a single string for similarity comparison
        topic_vec = ' '.join(top_words)
        
        # Initialize variables to track the most similar label
        max_similarity = 0
        assigned_label = 'Unknown'
        
        # Compare with each label's predefined keywords
        for label, keywords in predefined_labels.items():
            keyword_vec = ' '.join(keywords)
            
            # Vectorize the keyword and topic vectors
            topic_vector = CountVectorizer().fit_transform([keyword_vec, topic_vec]).toarray()
            
            # Compute the cosine similarity between the topic and the label keywords
            similarity = cosine_similarity([topic_vector[0]], [topic_vector[1]])[0][0]
            
            # If the similarity is the highest so far, assign this label to the topic
            if similarity > max_similarity:
                max_similarity = similarity
                assigned_label = label
        
        labeled_topics.append((topic, assigned_label))
    
    return pd.DataFrame(labeled_topics, columns=['Topic', 'Label'])

def interpret_topics_with_experts_and_automation(df, model, predefined_labels):
    """
    Integrates both expert interpretation and automated labeling for topics.
    
    :param df: DataFrame containing the documents and the assigned topics (e.g., 'bertopic_topic')
    :param model: Fitted BERTopic model
    :param predefined_labels: Optional predefined labels to classify topics
    """
    # Step 1: Extract representative words based on BERTopic model
    representative_words = extract_representative_words(df, model)
    
    # Step 2: Automate labeling based on predefined labels (optional)
    automated_labels = automate_topic_labeling(representative_words, predefined_labels=predefined_labels)
    
    # Step 3: Get topic names (the actual labels from the BERTopic model)
    topic_names = model.get_topic_info()
    
    # Map the topic number to the actual topic name
    representative_words['Topic Name'] = representative_words['Topic'].map(topic_names.set_index('Topic')['Name'])
    
    # Step 4: Combine the results with topic names and automated labels
    result = pd.merge(representative_words, automated_labels, on='Topic')
    
    # Optionally, you can add domain expert interpretation here (e.g., manual input)
    
    return result