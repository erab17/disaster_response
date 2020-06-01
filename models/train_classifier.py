import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(["wordnet", "punkt", "stopwords"])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """ Load data from database, perform simple data cleaning and
    split data into X and Y data."""
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Disaster_Data", engine)
    #There is a 2 that shouldnt be there, replacing it with the mode
    df["related"] = np.where(df["related"]==2, 0, 1)
    #df.drop("child_alone", axis=1, inplace=True)
    Y = df.iloc[:,4:]
    X = df["message"]
    return X, Y, Y.columns


def tokenize(text):
    """ Takes input text, lower it, remove everything except letters,
    keep only useful words, remove stopwords and lemmatize words and
    returns it."""
    #Normalize, Remove punctuation characters and lower text
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    #Tokenize
    text = word_tokenize(text)
    #Remove stop words
    words = [w for w in text if w not in stopwords.words("english")]
    # Lemmetize, reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    """ Build pipeline of countvectrorizer, tfidf and a randomforest classifier.
    Create gridsearchcv with a few parameters and returns this model."""
    pipeline5 = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))#n_estimators=100, min_samples_split=4)))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
            }
    cv = GridSearchCV(pipeline5, param_grid=parameters)#, verbose=2)
    return cv
        


def evaluate_model(model, X_test, Y_test, category_names):
    """ Predict based on the testset, prints a classification report
    with classification accuracy for the different disasters."""
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """ Save model to the specified filepath"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()