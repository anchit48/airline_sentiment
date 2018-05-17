""" Rubikloud take home problem """
import luigi
import pandas as pd
import numpy as np
import ast
import pickle

from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def closest_node(node, nodes):
    """
    Euclidean distance logic
    """
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    output_file = luigi.Parameter(default='clean_data.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        tweets_df = pd.read_csv(self.tweet_file, encoding = "ISO-8859-1")
        
        #Dropping NA and [0.0, 0.0] values
        tweets_clean_df = tweets_df[['airline_sentiment', 'tweet_coord']].dropna()
        tweets_clean_df = tweets_clean_df.loc[tweets_clean_df['tweet_coord'] != '[0.0, 0.0]']
        tweets_clean_df.to_csv(self.output().path)


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    def requires(self):
        return CleanDataTask(tweet_file=self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        tweets_clean_df = pd.read_csv(self.input().open('r'), encoding = "ISO-8859-1")
        cities_df = pd.read_csv(self.cities_file, encoding = "ISO-8859-1")

        # Replacing sentiment class with labels
        labels = {'negative': 0, 'neutral': 1, 'positive': 2}
        tweets_clean_df.replace({'airline_sentiment': labels}, inplace=True)       

        # Will output a pandas Series of the closest cities
        tweets_clean_df['closest_city'] = tweets_clean_df['tweet_coord'].map(ast.literal_eval).map(lambda coords: closest_node(coords, cities_df[['latitude', 'longitude']])).map(lambda ind: cities_df['name'][ind])

        # One-Hot Encoding the closest cities

        # I do not use a sparse matrix for enconding for its difficulty to be integrated into a pandas column, more efficient ways could be applied here
        # I fit and transform only according to the cities that show up in the tweets dataset, to save space
        le = LabelEncoder()
        cities_df['label'] = le.fit_transform(cities_df['name'])
        
        ohe = OneHotEncoder(sparse=False)#.fit(cities_df['label'].values.reshape(-1, 1))
        
        tweets_clean_df['closest_city_OHE'] = ohe.fit_transform(le.transform(tweets_clean_df['closest_city']).reshape(-1, 1)).tolist()
        
        # Formatting them to a dataframe with X, y, and write
        features_df = tweets_clean_df.rename(columns = {'closest_city_OHE': 'X', 'airline_sentiment': 'y'})[['X', 'y']]
        features_df.to_csv(self.output().path)



class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    output_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        return TrainingDataTask(tweet_file=self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        features_df = pd.read_csv(self.input().open('r'), index_col=0)

        # Transform string representation of list to list
        features_df['X'] = features_df['X'].map(ast.literal_eval)

        # I do a training and testing split here, though I do not explicitly use the test set
        X_train, X_test, y_train, y_test = train_test_split(sparse.csr_matrix(features_df['X'].tolist()), features_df['y'], test_size=0.3, random_state=42)

        # Grid Search on Random Forest
        params = {'n_estimators':  [10, 50, 100],
                    'max_depth': [None, 5, 10],
                    'max_features': ['sqrt', 'auto'],
                    'class_weight': ['balanced', 'balanced_subsample']
                     }

        rfc_GS = GridSearchCV(RandomForestClassifier(), params, verbose=1, n_jobs=4)
        rfc_GS.fit(X_train, y_train)

        # Printing metrics
        print('\n\n')
        print('Baseline Train Score (0 Prediction): {:0.4f}'.format(accuracy_score(y_train, np.zeros(X_train.shape[0]))))
        print('Baseline Test Score (0 Prediction): {:0.4f}'.format(accuracy_score(y_test, np.zeros(X_test.shape[0]))))
        print('F1 Test Score (Macro Avg): %0.4f' % f1_score(y_test, np.zeros(X_test.shape[0]), average='macro'))
        print("\n---- Baseline Classification Report (Test) ---\n")
        print(classification_report(y_test,  np.zeros(X_test.shape[0])))

        print('Train score: %0.4f' % rfc_GS.best_estimator_.score(X_train, y_train))
        print('Test score: %0.4f' % rfc_GS.best_estimator_.score(X_test, y_test))
        print('F1 Score (Macro Avg): %0.4f' % f1_score(y_test, rfc_GS.best_estimator_.predict(X_test), average='macro'))
        print("\n---- Classification Report (Test) ---\n")
        print(classification_report(y_test, rfc_GS.best_estimator_.predict(X_test)))
        print('\n\n')

        #Save the classifier
        #pickle.dump(rfc_GS.best_estimator_, open(self.output().path, 'wb'))
        with open(self.output().path, 'wb') as f:
            pickle.dump(rfc_GS.best_estimator_, f) 


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    output_file = luigi.Parameter(default='scores.csv')
    # Adding Cities file in addition
    cities_file = luigi.Parameter(default='cities.csv')

    def requires(self):
        return TrainModelTask(tweet_file=self.tweet_file), TrainingDataTask(tweet_file=self.tweet_file), CleanDataTask(tweet_file=self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        loaded_model = pickle.load(open(self.input()[0].path, 'rb'))
        features_df = pd.read_csv(self.input()[1].open('r'), index_col=0)
        tweets_clean_df = pd.read_csv(self.input()[2].open('r'), index_col=0, encoding = "ISO-8859-1")
        cities_df = pd.read_csv(self.cities_file, encoding = "ISO-8859-1")

        # Applying Euclidean distance logic
        tweets_clean_df['closest_city'] = tweets_clean_df['tweet_coord'].map(ast.literal_eval).map(lambda coords: closest_node(coords, cities_df[['latitude', 'longitude']])).map(lambda ind: cities_df['name'][ind])

        # Predict probabilities
        probs = loaded_model.predict_proba(sparse.csr_matrix(features_df['X'].map(ast.literal_eval).tolist()))

        scores_df = pd.DataFrame(tweets_clean_df['closest_city']).rename(columns = {'closest_city': 'city name'})
        scores_df['negative probability'] = probs[:, 0]
        scores_df['neutral probability'] = probs[:, 1] 
        scores_df['positive probability'] = probs[:, 2]

        # Store the result in 'score.csv' 
        scores_df.to_csv(self.output().path, encoding = "ISO-8859-1")


if __name__ == "__main__":
    # Adding task name to make it run from Python Shell
    luigi.run(['ScoreTask', '--workers', '1', '--local-scheduler'])
