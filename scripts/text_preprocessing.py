import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure required NLTK packages are downloaded only if necessary
def download_nltk_resources():
    try:
        # Check for stopwords and WordNet lemmatizer data
        stopwords.words('english')
        
    except LookupError:
        nltk.download('stopwords')
        nltk.download('vader_lexicon')
        
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('vader_lexicon')

# only download when not there.
download_nltk_resources()


class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Tokenize and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words]
        return ', '.join(tokens)

# Example usage:
# preprocessor = TextPreprocessor()
# cleaned_text = preprocessor.clean_text("Sample text with punctuation!")
