#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille

import time
import warnings
import sklearn.exceptions
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore",
                        category=sklearn.exceptions.UndefinedMetricWarning)


class Classifier():
    """
    Classification task using a particular method

    Parameters
    ----------
    * dataset: input dataset dict including dataset['train'] pandas DataFrame,
        dataset['test'] pandas DataFrame and dataset['target'] pandas
        DataSeries obtained from train_test_split function of Reader class

    * strategy: str, default = 'NB' ; The choice for the classification method:
    'NB', 'LDA', 'CART' and 'MNB'

    * target: str, name of the target variable encoded as int64 from
        dataset['target'] pandas DataSeries

    * k_folds: int, default = 10, number of folds for cross-validation

    * verbose: Boolean,  default = 'False' otherwise display the list of
        duplicate rows that have been removed.
    """

    def __init__(self, dataset, target, strategy='NB', k_folds=10,
                 verbose=False):

        self.dataset = dataset

        self.target = target

        self.strategy = strategy

        self.k_folds = k_folds

        self.verbose = verbose

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'target': self.target,

                'k_folds': self.k_folds,

                'verbose': self.verbose}

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for clusterer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`classifier.get_params().keys()`")
            else:

                setattr(self, k, v)

    def LDA_classification(self, dataset, target):
        # quality metrics : accuracy
        # requires no missing values
        k = self.k_folds

        X_train = dataset['train'].select_dtypes(['number']).dropna()

        if (len(X_train.columns) <= 1) or (len(X_train) < k):

            print('Error: Need at least one continous variable and',
                  k, 'observations for classification')

            accuracy = None

        else:

            y_train = dataset['target'].loc[X_train.index]

            X_test = dataset['test'].select_dtypes(['number']).dropna()

            if (isinstance(self.dataset['target_test'], dict)):

                y_test = dataset['target'].loc[X_test.index]

            else:

                y_test = dataset['target_test']

            if target in X_train.columns.values:

                X_train = X_train.drop([target], 1)

            if target in X_test.columns.values:

                X_test = X_test.drop([target], 1)

            if (dataset['target'].nunique() < k):

                k = dataset['target'].nunique()

            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

            params = {}

            model = LinearDiscriminantAnalysis(n_components=2)

            gs = GridSearchCV(
                model, cv=skf, param_grid=params, scoring='accuracy')

            gs.fit(X_train, y_train)

            results = gs.cv_results_

            if self.verbose:

                print(results)

            clf = gs.best_estimator_

            best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]

            accuracy = results['mean_test_score'][best_index]

            if target in X_test.columns.values:

                accuracy = clf.score(X_test, y_test)

                y_true, y_pred = y_test, clf.predict(X_test)

                if self.verbose:

                    print("Detailed classification report:")

                    print()

                    print("The model is trained on the full development set. ")

                    print("The scores are computed on the full "
                          "evaluation set.")

                    print()

                    print("Labels in y_test that don't appear in y_pred "
                          "causing ill-defined recall, precision, or "
                          "F1 metrics "
                          "in the classification "
                          "report:",  set(y_test) - set(y_pred))

                    print(classification_report(y_true, y_pred))

            print()

            print("Accuracy of LDA result for", self.k_folds,
                  "cross-validation :", accuracy)

            print()

        return accuracy

    def CART_classification(self, dataset, target):
        # quality metrics : accuracy
        # requires no missing values
        NUM_TRIALS = 10

        k = self.k_folds

        X_train = dataset['train'].select_dtypes(['number']).dropna()

        if (len(X_train.columns) <= 1) or (len(X_train) < k):

            print('Error: Need at least one continous variable and',
                  k, 'observations for classification')

            accuracy = None

        else:

            y_train = dataset['target'].loc[X_train.index]

            X_test = dataset['test'].select_dtypes(['number']).dropna()

            if (isinstance(self.dataset['target_test'], dict)):

                y_test = dataset['target'].loc[X_test.index]

            else:

                y_test = dataset['target_test']

            if target in X_train.columns.values:

                X_train = X_train.drop([target], 1)

            if target in X_test.columns.values:

                X_test = X_test.drop([target], 1)

            # if (dataset['target'].nunique() < k):

                # k = dataset['target'].nunique()
            # skf = StratifiedKFold(n_splits=k)
            params = {'max_depth': [3, 5, 7, 9, 10]}

            non_nested_scores = np.zeros(NUM_TRIALS)

            nested_scores = np.zeros(NUM_TRIALS)

            for i in range(1, NUM_TRIALS):

                inner_cv = KFold(n_splits=k, shuffle=True, random_state=i)

                outer_cv = KFold(n_splits=k, shuffle=True, random_state=i)

                model = DecisionTreeClassifier(random_state=i)

                gs = GridSearchCV(model, cv=inner_cv,
                                  param_grid=params, scoring='accuracy')

                gs.fit(X_train, y_train)

                non_nested_scores[i] = gs.best_score_

                nested_score = cross_val_score(
                    gs, X=X_train, y=y_train, cv=outer_cv)

                nested_scores[i] = nested_score.mean()

            clf = gs.best_estimator_

            results = gs.cv_results_

            if self.verbose:

                print(results)

            best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]

            accuracy = results['mean_test_score'][best_index]

            if target in X_test.columns.values:

                accuracy = clf.score(X_test, y_test)

                y_true, y_pred = y_test, clf.predict(X_test)

                if self.verbose:

                    print("Detailed classification report:")

                    print()

                    print("The model is trained on the full development set.")

                    print("The scores are computed on the "
                          "full evaluation set.")

                    print()

                    print(classification_report(y_true, y_pred))

                    print("Labels in y_test that don't appear in y_pred "
                          "causing ill-defined recall,  "
                          "precision, or F1 metrics in the "
                          "classification report:",  set(y_test) - set(y_pred))

            print("Avg accuracy of CART classification for",
                  k, "cross-validation :", accuracy)

            print()

        return accuracy

    def NB_classification(self, dataset, target):  # quality metrics : accuracy
        # requires no missing values
        k = self.k_folds

        X_train = dataset['train'].select_dtypes(['number']).dropna()

        if (len(X_train.columns) <= 1) or (len(X_train) < k):

            print('Error: Need at least one continous variable and',
                  k, 'observations for classification')

            accuracy = None

        else:

            y_train = dataset['target'].loc[X_train.index]

            X_test = dataset['test'].select_dtypes(['number']).dropna()

            if (isinstance(self.dataset['target_test'], dict)):

                y_test = dataset['target'].loc[X_test.index]

            else:

                y_test = dataset['target_test']

            if target in X_train.columns.values:

                X_train = X_train.drop([target], 1)

            if target in X_test.columns.values:

                X_test = X_test.drop([target], 1)

            # if (dataset['target'].nunique() < k):

                # k = dataset['target'].nunique()

            skf = StratifiedKFold(n_splits=k)

            params = {}

            model = GaussianNB()

            gs = GridSearchCV(
                model, cv=skf, param_grid=params, scoring='accuracy')

            gs.fit(X_train, y_train)

            clf = gs.best_estimator_

            results = gs.cv_results_

            if self.verbose:

                print(results)

            best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]

            accuracy = results['mean_test_score'][best_index]

            if target in X_test.columns.values:

                accuracy = clf.score(X_test, y_test)

                y_true, y_pred = y_test, clf.predict(X_test)

                if self.verbose:

                    print("Detailed classification report:")

                    print()

                    print("The model is trained on the full development set.")

                    print("The scores are computed on the full "
                          "evaluation set.")

                    print()

                    print(classification_report(y_true, y_pred))

                    print("Labels in y_test that don't appear in y_pred "
                          "causing ill-defined recall, precision,"
                          " or F1 metrics in the "
                          "classification report:",  set(y_test) - set(y_pred))

            print("Accuracy of Naive Naive Bayes classification for",
                  k, "cross-validation :", accuracy)

            print()

        return accuracy

    def MNB_classification(self, dataset, target):  # quality metrics: accuracy
        # requires no missing values
        k = self.k_folds

        X_train = dataset['train'].select_dtypes(['number']).dropna()

        if (len(X_train.columns) <= 1) or (len(X_train) < k):

            print('Error: Need at least one continous variable and',
                  k, 'observations for classification')

            accuracy = None

        else:

            y_train = dataset['target'].loc[X_train.index]

            X_test = dataset['test'].select_dtypes(['number']).dropna()

            if (isinstance(self.dataset['target_test'], dict)):

                y_test = dataset['target'].loc[X_test.index]

            else:

                y_test = dataset['target_test']

            if target in X_train.columns.values:

                X_train = X_train.drop([target], 1)

            if target in X_test.columns.values:

                X_test = X_test.drop([target], 1)

            # if (dataset['target'].nunique() < k):

                # k = dataset['target'].nunique()

            skf = StratifiedKFold(n_splits=k)

            params = {"alpha": np.arange(0.001, 1, 0.01)}

            model = MultinomialNB()

            gs = GridSearchCV(
                model, cv=skf, param_grid=params, scoring='accuracy')

            gs.fit(X_train, y_train)

            clf = gs.best_estimator_

            results = gs.cv_results_

            best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]

            accuracy = results['mean_test_score'][best_index]

            if target in X_test.columns.values:

                accuracy = clf.score(X_test, y_test)

                y_true, y_pred = y_test, clf.predict(X_test)

                if self.verbose:

                    print()

                    print("Detailed classification report:")

                    print()

                    print("The model is trained on the full development set.")

                    print("The scores are computed on the full "
                          "evaluation set.")

                    print()

                    print(classification_report(y_true, y_pred))

                    print("Labels in y_test that don't appear in"
                          " y_pred causing "
                          "ill-defined recall, precision, or F1 metrics "
                          "in the "
                          "classification report:",  set(y_test) - set(y_pred))

            if self.verbose:

                print("Best alpha:", gs.best_params_)

                print("Accuracy scores on development set:")

                means = gs.cv_results_['mean_test_score']

                stds = gs.cv_results_['std_test_score']

                for mean, std, params in zip(means, stds,
                                             gs.cv_results_['params']):

                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2,
                          params))

            print("Accuracy of Multinomial Naive Bayes classification for",
                  k, "cross-validation : %0.3f" % accuracy)

            print()

        return accuracy

    def transform(self):

        start_time = time.time()

        d = self.dataset

        if self.target == d['target'].name:

            print()

            print(">>Classification task")

            if (self.strategy == "LDA"):

                dn = self.LDA_classification(dataset=d, target=self.target)

            elif (self.strategy == "CART"):

                dn = self.CART_classification(dataset=d, target=self.target)

            elif (self.strategy == "NB"):

                dn = self.NB_classification(dataset=d, target=self.target)

            elif (self.strategy == "MNB"):

                dn = self.MNB_classification(dataset=d, target=self.target)

            else:

                raise ValueError(
                    "The classification function should be LDA, CART, NB "
                    "or MNB.")

            print("Classification done -- CPU time: %s seconds" %
                  (time.time() - start_time))

        else:

            raise ValueError("Target variable invalid.")

        return {'quality_metric': dn}
