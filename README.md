# Disaster Response Pipeline Project

This is a project that will try to predict what type of disaster a certain text is classified as. The model that will predict what type of disaster a text is based on is trained on labeled data.

The experimenting such as loading, cleaning, finding, model and hyperparameters are performed separately in two notebooks.

The files included in this project are the following:
- ETL Pipeline Preparation.ipynb, notebook file with ETL exploration.
- ML Pipeline Preparation.ipynb, notebook file with machine learning experimentation.
- README.md, this file with a short summary of the project.
- run.py, main python file that runs the flask application.
- disaster_categories.csv, csv file with all the disaster categories.
- disaster_messages.csv, csv file with the disaster texts.
- process_data.py, python file that will do the data loading and cleaning.
- train_classifier.py, python file that will train a random forest model with a cross validation approach.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
