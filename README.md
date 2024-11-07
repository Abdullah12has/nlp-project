Hansard UK Parliamentary Debates NLP Analysis
=============================================

Project Overview
----------------

This project explores themes in UK parliamentary debates over time, analyzing emotional patterns in policy discussions and identifying differences in topic focus across political parties or speakers. The dataset contains parliamentary debate transcripts, which are analyzed using topic modeling and sentiment analysis techniques.

## 1. Text Preprocessing

Perform standard text preprocessing tasks:
- Remove stop words, punctuation, and special characters.
- Lowercase the text.
- Tokenize the text.
- Apply stemming or lemmatization.

## 2. Initial Data Exploration

Explore the distribution of key features:
- `Speech_date`, `year`, `time`, `gender`, and `party_group`.
- Visualize these distributions to gain insights into data patterns.

## 3. Speech Word Frequency Analysis

1. **Sentiment Classification**: Define a function to classify each speech as positive or negative based on sentiment scores.
2. **Most Common Words**: Generate a word frequency count for both positive and negative speeches:
   - Visualize frequent words using word clouds and bar charts.
   - Separate visualizations for positive and negative speeches.
3. **N-gram Analysis**:
   - Analyze bi-grams and trigrams to identify common phrases in the speeches.
   - Conduct separate analyses for positive and negative speeches to reveal phrases indicating strong sentiment.
4. **Feature-Based Analysis**: Repeat the Most Common Words and N-gram Analysis for `party_group` and `gender` features.

## 4. Correlation Between Features and Sentiment

Calculate the correlation between sentiment scores and features (`Speech_date`, `year`, `time`, `gender`, and `party_group`) to analyze sentiment trends:
- Identify if certain features (e.g., male, Labour) tend to be associated with positive or negative sentiment.
- Visualize distribution plots for positive and negative speeches to explore potential patterns.

## 5. Correlation Heatmap

Draw a correlation heatmap for sentiment scores from different models:
- `afinn_sentiment`, `jockers_sentiment`, `nrc_sentiment`, `huliu_sentiment`, and `rheault_sentiment`.
- Analyze the correlation results for insights into model alignment.

## 6. Topic Modeling with LDA and BERTopic

1. **Topic Modeling**: Implement topic modeling using LDA and BERTopic.
2. **Hyperparameter Optimization**: Optimize hyperparameters for both models using coherence scores (e.g., Cv measure) for optimal topic extraction.
3. **Visualization**: Use visualization tools like `pyLDAvis` and BERTopic’s built-in functions for interactive topic exploration.

## 7. Topic Evolution Over Time

Track the evolution of topics over time:
- Use Dynamic Topic Modeling (LDA) and BERTopic’s time-based analysis.
- Visualize topic trends to study policy shifts over time.

## 8. Sentiment Correlation with Topics

Perform sentiment analysis on debate transcripts using VADER or TextBlob, with potential fine-tuning for parliamentary language:
- Correlate sentiment trends with identified topics.
- Highlight emotional patterns and contextual nuances (e.g., sarcasm, negation).

## 9. Comparison of Pre-Trained Sentiment Models with Ground Truth Labels

1. **Sentiment Analysis**: Use pre-trained models (VADER, TextBlob) to analyze speech sentiment.
2. **Evaluation**: Compare sentiment scores with ground truth scores in the dataset.
3. **Error Analysis**: Calculate evaluation metrics (MSE, RMSE) and examine cases where pre-trained models significantly diverge from ground truth (e.g., sarcasm, ambiguous language).

## 10. Sentiment Prediction Using Extracted Features

Convert speech text into numerical representations for machine learning models:
1. Use word embeddings (Word2Vec, GloVe, BERT) for contextual and semantic capture.
2. Build a sentiment classification model to predict whether a speech is positive or negative.
3. Train the model using extracted features and evaluate on a test dataset.
4. Compare performance across classifiers (Logistic Regression, SVM, Random Forest, and deep learning models such as LSTM or CNN).

## 11. Topic Distributions Across Political Parties and Speakers

Analyze and visualize topic distributions across political parties and speakers:
- Reveal trends in political discourse and differences in focus among parties or speakers.

## 12. Topic Representation and Interpretation

Work with domain experts for topic interpretation, aided by automated labeling techniques:
- Systematic labeling for large datasets to assign meaningful labels to each topic based on representative words.

## 13. Advanced NLP and LLM Techniques and Suggestions

Explore additional NLP techniques or state-of-the-art models:
- Experiment with transformer-based models such as RoBERTa or GPT.
- Compare their performance to previous methods (BERT, traditional embeddings).

## 14. Literature Review and Discussion

Identify relevant literature to support findings:
- Discuss strengths and weaknesses of the data processing pipeline.

---



File Structure
--------------

*   **main\_analysis.ipynb**: The main Jupyter Notebook for running the complete analysis.
    
*   **scripts/**: Contains reusable Python scripts for different tasks:
    
    *   text\_preprocessing.py
        
    *   data\_exploration.py
        
    *   sentiment\_analysis.py
        
    *   sentiment\_correlation.py
        
    *   topic\_modeling.py
        
    *   sentiment\_prediction.py
        
*   **data/**: Directory containing the CSV data file (senti\_df.csv).
    
*   **outputs/**: Directory to save generated plots and models.
    
*   **requirements.txt**: Python package requirements for the project.
    

Installation and Setup
----------------------

1.  bashCopy codegit clone https://github.com/username/hansard-nlp-analysis.gitcd hansard-nlp-analysis
    
2.  bashCopy codepython -m venv envsource env/bin/activate # On Windows: env\\Scripts\\activate
    
3.  bashCopy codepip install -r requirements.txt
    
4.  bashCopy codejupyter notebook main\_analysis.ipynb
    

How to Run
----------

1.  Open main\_analysis.ipynb and run each cell sequentially to perform the analysis.
    
2.  Ensure that data/senti\_df.csv is in the correct directory.
    

Project Tasks
-------------

1.  **Text Preprocessing**: Cleaning and normalizing the speech text.
    
2.  **Data Exploration**: Visualizing distributions of features such as date, year, gender, and party group.
    
3.  **Sentiment Analysis**: Classifying speeches and visualizing word frequencies.
    
4.  **N-gram Analysis**: Identifying common bi-grams and tri-grams.
    
5.  **Correlation Analysis**: Plotting correlation heatmaps for different sentiment scores.
    
6.  **Topic Modeling**: Implementing LDA and BERTopic for topic extraction.
    
7.  **Sentiment Prediction**: Training models using extracted features for sentiment classification.
    
8.  **Visualization and Interpretation**: Generating insights through plots and model outputs.
    

Requirements
------------

*   Python 3.8+
    
*   Jupyter Notebook
    
*   Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, gensim, BERTopic, etc.
    

Refer to requirements.txt for the complete list.

Results and Observations
-----------------------

*   Detailed results, including visualizations and sentiment trends, can be found within the notebook output.
