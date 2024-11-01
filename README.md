Hansard UK Parliamentary Debates NLP Analysis
=============================================

Project Overview
----------------

This project explores UK parliamentary debates using NLP techniques to perform topic modeling, sentiment analysis, and trend analysis. The analysis covers aspects such as identifying themes, analyzing emotional patterns in policy discussions, and comparing the topic focus across political groups and speakers.

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
------------------------

*   Detailed results, including visualizations and sentiment trends, can be found within the notebook output.