# Disaster Response Pipeline Project

### Project description
In the Project, data set containing real messages that were sent during disaster events is used to build Machine learning model to categorize events so that one can send the messages to an appropriate disaster relief agency.

Dataset used: disaster data from Figure Eight(https://www.figure-eight.com/)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Main files
1. `process_data.py`
    - Load the CSV files.
    - Merge the messages & categories df. 
    - Process the categories to a format which is better suitable for processing.
    - Clean the data ( Remove Duplicates ).
    - Save the DataFrame into SQLite db.
    
2. `train_classifier.py`
    - Load and split the data from the SQLite DB into test and train sets.
    - The script uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text.
    - Use GridSearch to find the best parameters of a `RandomForestClassifier`.
    - Use the best parameters found above to train the model.
    - Measure & display the performance of the trained model on the test set. 
    - Save the model as a Pickle file. 
