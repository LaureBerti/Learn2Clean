#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille

import warnings
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class Imputer():
    """
    Replace or remove the missing values using a particular strategy

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * strategy: str, default = 'DROP'
        The choice for the feature selection strategy:
            - 'EM': only for numerical variables; imputation based on
                expectation maximization
            - 'MICE': only for numerical variables  missing at random (MAR);
                Multivariate Imputation by Chained Equations
            - 'KNN', only for numerical variables; k-nearest neighbor
                imputation (k=4) which weights samples using the mean squared
                difference on features for which two rows both have observed
                data
            - 'RAND', 'MF': both for numerical and categorical variables;
                replace missing values by randomly selected value in the
                variable domain or by the most frequent value in the variable
                domain respectively
            - 'MEAN', 'MEDIAN': only for numerical variables; replace missing
                values by mean or median of the numerical variable respectvely
            - or 'DROP' remove the row with at least one missing value

    * verbose: Boolean,  default = 'False' otherwise display about imputation

    * threshold: float, default =  None

    * exclude: str, default = 'None' name of variable to be excluded
        from imputation
    """

    def __init__(self, dataset, strategy='DROP', verbose=False,
                 exclude=None, threshold=None):

        self.dataset = dataset

        self.strategy = strategy

        self.verbose = verbose

        self.threshold = threshold

        self.exclude = exclude  # to implement

    def get_params(self, deep=True):

        return {'strategy': self.strategy,
                'verbose': self.verbose,
                'exclude': self.exclude,
                'threshold': self.threshold}

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`imputer.get_params().keys()`")

            else:

                setattr(self, k, v)

    # Handling Missing values

    def mean_imputation(self, dataset):
        # for numerical data
        # replace missing numerical values by the mean of
        # the corresponding variable

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = dataset.select_dtypes(['number'])

            for i in X.columns:

                X[i] = X[i].fillna(int(X[i].mean()))

            Z = dataset.select_dtypes(exclude=['number'])

            df = pd.DataFrame.from_records(
                X, columns=dataset.select_dtypes(['number']).columns)

            df = pd.concat([X, Z], axis=1)

        else:

            pass

        return df

    def median_imputation(self, dataset):
        # only for numerical data
        # replace missing numerical values by the median
        # of the corresponding variable

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = dataset.select_dtypes(['number'])

            for i in X.columns:

                X[i] = X[i].fillna(int(X[i].median()))

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(
                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df

    def NaN_drop(self, dataset):
        # for both categorical and numerical data
        # drop observations with missing values

        print("Dataset size reduced from", len(
            dataset), "to", len(dataset.dropna()))

        return dataset.dropna()

    def MF_most_frequent_imputation(self, dataset):
        # for both categorical and numerical data
        # replace missing values by the most frequent value
        # of the corresponding variable

        for i in dataset.columns:

            mfv = dataset[i].value_counts().idxmax()

            dataset[i] = dataset[i].replace(np.nan, mfv)

            if self.verbose:

                print("Most frequent value for ", i, "is:", mfv)

        return dataset

    def NaN_random_replace(self, dataset):
        # for both categorical and numerical data
        # replace missing data with a random observation with data

        M = len(dataset.index)

        N = len(dataset.columns)

        ran = pd.DataFrame(np.random.randn(
            M, N), columns=dataset.columns, index=dataset.index)

        dataset.update(ran)

        return dataset

    def KNN_imputation(self, dataset, k=4):
        # only for numerical values
        # Nearest neighbor imputations which weights samples
        # using the mean squared difference on features for which two
        # rows both have observed data.

        from fancyimpute import KNN

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = dataset.select_dtypes(['number'])

            for i in X.columns:

                X[i] = KNN(k=k, verbose=False).fit_transform(X)

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(
                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df

    def MICE_imputation(self, dataset):
        # only for numerical values
        # Multivariate Imputation by Chained Equations only suitable
        # for Missing At Random (MAR),
        # which means that the probability that a value is missing
        # depends only on observed values and not on unobserved values

        import impyute as imp

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = imp.mice(dataset.select_dtypes(['number']).iloc[:, :].values)

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(
                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df

    def EM_imputation(self, dataset):
        # only for numerical values
        # imputes given data using expectation maximization.
        # E-step: Calculates the expected complete data log
        # likelihood ratio.

        import impyute as imp

        df = dataset

        if dataset.select_dtypes(['number']).isnull().sum().sum() > 0:

            X = imp.em(dataset.select_dtypes(['number']).iloc[:, :].values)

            Z = dataset.select_dtypes(include=['object'])

            df = pd.DataFrame.from_records(

                X, columns=dataset.select_dtypes(['number']).columns)

            df = df.join(Z)

        else:

            pass

        return df

    def transform(self):

        start_time = time.time()

        print(">>Imputation ")

        impd = self.dataset

        for key in ['train', 'test']:

            if (not isinstance(self.dataset[key], dict)):

                d = self.dataset[key].copy()

                print("* For", key, "dataset")

                total_missing_before = d.isnull().sum().sum()

                Num_missing_before = d.select_dtypes(
                    include=['number']).isnull().sum().sum()

                NNum_missing_before = d.select_dtypes(
                    exclude=['number']).isnull().sum().sum()

                print("Before imputation:")

                if total_missing_before == 0:

                    print("No missing values in the given data")

                else:

                    print("Total", total_missing_before, "missing values in",
                          d.columns[d.isnull().any()].tolist())

                    if Num_missing_before > 0:

                        print("-", Num_missing_before,
                              "numerical missing values in",
                              d.select_dtypes(['number']).
                              columns[d.select_dtypes(['number']).
                                      isnull().any()].tolist())

                    if NNum_missing_before > 0:

                        print("-", NNum_missing_before,
                              "non-numerical missing values in",
                              d.select_dtypes(['object']).
                              columns[d.select_dtypes(['object']).
                                      isnull().any()].tolist())

                    if (self.strategy == "EM"):

                        dn = self.EM_imputation(d)

                    elif (self.strategy == "MICE"):

                        dn = self.MICE_imputation(d)

                    elif (self.strategy == "KNN"):

                        dn = self.KNN_imputation(d)

                    elif (self.strategy == "RAND"):

                        dn = self.NaN_random_replace(d)

                    elif (self.strategy == "MF"):

                        dn = self.MF_most_frequent_imputation(d)

                    elif (self.strategy == "MEAN"):

                        dn = self.mean_imputation(d)

                    elif (self.strategy == "MEDIAN"):

                        dn = self.median_imputation(d)

                    elif (self.strategy == "DROP"):

                        dn = self.NaN_drop(d)
                    else:

                        raise ValueError("Strategy invalid. Please "
                                         "choose between "
                                         "'EM', 'MICE', 'KNN', 'RAND', 'MF', "
                                         "'MEAN', 'MEDIAN', or 'DROP'")

                    impd[key] = dn

                    print("After imputation:")

                    print("Total", impd[key].isnull(
                    ).sum().sum(), "missing values")

                    print("-", impd[key].select_dtypes(include=['number']
                                                       ).isnull().sum().sum(),
                          "numerical missing values")

                    print("-", impd[key].select_dtypes(exclude=['number']
                                                       ).isnull().sum().sum(),
                          "non-numerical missing values")

            else:

                print("No", key, "dataset, no imputation")

        print("Imputation done -- CPU time: %s seconds" %
              (time.time() - start_time))

        print()

        return impd
