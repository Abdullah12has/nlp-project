from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim_models

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

# Example usage:
# lda_model, dictionary, corpus = train_lda_model(df, 'text')
# topic_model = train_bertopic_model(df, 'text')
