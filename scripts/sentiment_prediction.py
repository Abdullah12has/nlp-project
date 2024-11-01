from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_sentiment_model(df, text_column, label_column):
    X_train, X_test, y_train, y_test = train_test_split(df[text_column], df[label_column], test_size=0.3, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Example usage:
# train_sentiment_model(df, 'text', 'sentiment')
