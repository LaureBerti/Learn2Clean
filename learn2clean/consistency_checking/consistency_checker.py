#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille <laure.berti@ird.fr>

import warnings
import time
from ast import literal_eval
import re
import pandas as pd
pd.options.mode.chained_assignment = None


def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else literal_eval(val)


def constraint_discovery(dataset, file_name):
    # function to discover constraints from the input dataset
    # and generate the corresponding file
    from tdda.constraints import discover_df
    constraints = discover_df(dataset)
    fn = './save/' + file_name + '_constraints.tdda'
    with open(fn, 'w') as f:
        f.write(constraints.to_json())


def pattern_discovery(dataset, file_name):
    # function to discover patterns from the input dataset and
    # generate the corresponding file
    from tdda import rexpy

    import collections

    from functools import reduce

    # dataset = dataset.select_dtypes(['object'])
    patterns = collections.defaultdict(list)

    list_pattern = list()

    listp = []

    fn = './save/'+file_name + '_patterns.txt'
    # file = open(fn,"w")

    for c in dataset.columns.values:

        corpus = dataset[c].unique().astype('str').tolist()

        results = rexpy.extract(corpus)

        patterns[c].append(results)

        for i in range(0, len(patterns)):

            lp = list(patterns.items())[i][1]

            lp = reduce(lambda x, y: x+y, lp)

            p = reduce(lambda x, y: x+y, lp)

            list_pattern = (c, i, "'"+str(p)+"'")

            listp.append(list_pattern)

            # file.writelines(str(list_pattern)+ '\n')

    p = pd.DataFrame(listp, columns=['col', 'num', 'pattern'])

    p.to_csv(fn, header=('col', 'num', 'pattern'), index=False, sep=';')

    return p


class Consistency_checker():
    """
    Identify and remove rows that do not violate constraints
    or patterns specified in a file .tdda or .txt
    for the strategy 'CC' or 'PC' respectively

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * strategy : str, default = 'CC'
        The choice for consistency checking strategy :
            - 'CC': checks whether the data satisfy the constraints
                specified in 'file_name'_constraint.tdda
            - 'PC': checks whether the data satisfy the patterns
                specified in 'file_name'_patterns.txt

    * file_name: str, prefix of the constraint/pattern file name to be
        used to check violations

    * verbose: Boolean, default = 'False' to list the violations

    * exclude: str, default = 'None' name of variable to be excluded
        from consistency checking
    * threshold: float, default =  None
    """

    def __init__(self, dataset, file_name,  strategy='CC',
                 verbose=False, threshold=None, exclude=None):

        self.dataset = dataset

        self.file_name = file_name

        self.strategy = strategy

        self.verbose = verbose

        self.threshold = threshold  # to implement

        self.exclude = exclude  # to implement

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'file_name': self.file_name,

                'verbose': self.verbose,

                'threshold': self.threshold,

                'exclude': self.exclude}

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`consistency_checker.get_params().keys()`")

            else:

                setattr(self, k, v)

    def CC_constraint_checking(self, dataset, file_name, verbose):

        from tdda.constraints import detect_df, verify_df
        # first define your constraints in a specific tdda
        # file associated to the dataset

        detection_df = 0

        d = detect_df(dataset,  'save/'+file_name + '_constraints.tdda')

        print("Constraints from the file:", file_name + '_constraints.tdda')

        detection_df = d.detected()

        df = dataset

        v = verify_df(dataset, 'save/'+file_name+'_constraints.tdda')

        print('Constraints passing: %d\n' % v.passes)

        print('Constraints failing: %d\n' % v.failures)

        if detection_df is not None:

            if verbose:

                print(str(v))

                print(v.to_frame())

                print('Row index with constraint failure:\n')

                print(list(detection_df.index))

        # return the dataset with inconsistent tuples removed
            to_keep = set(dataset.index) - set(detection_df.index)

        # return the dataset with inconsistent tuples removed
        if len(list(to_keep)) > 0:

            df = dataset.loc[list(to_keep)]

        else:

            df = dataset

        return df

    def PC_pattern_checking(self, dataset, file_name, verbose):
        # to do : handle complicated cases with mutliple patterns for
        # the same variable
        df = dataset.copy()

        fn = 'save/'+file_name + '_patterns.txt'

        # converters=dict.fromkeys('pattern', literal_converter))
        p = pd.read_csv(fn, sep=";")

        dataset = dataset.select_dtypes(['object'])

        to_drop = []

        to_keep = set(dataset.index)

        if verbose:

            print("Patterns:")

            print(p)

            print()

        for index, row in p.iterrows():

            c = row['col']

            n = row['num']

            pt = row['pattern']

            # vn = c+'_'+n+'_violation'
            pte = re.compile(eval(pt)) if pt else False

            a = 1

            for i in list(dataset.columns.values):

                if i == c:

                    z = dataset[c].str.contains(
                        pte, regex=True).sum()-len(dataset)

                    # check = 0 when a pattern is verified on variable i
                    # otherwise check = 1

                    check = dataset[dataset[c].str.contains(pte,
                                    regex=True) == 0].index

                    # several patterns may exist for one variable
                    # if a = 0, this means that at least one pattern i
                    a *= z

                    if z < 0:

                        print("Number of pattern violations on variable '",
                              c, "'for pattern#", n, ":", abs(z))
                        if verbose:
                            print("Indexes of rows to be removed:",
                                  list(to_drop))

                    if a == 0:

                        print("No violation on variable '", c,
                              "' for pattern#", n, "as", pt)

                        to_drop = []

                    else:

                        to_drop = set(to_drop) | set(check)

                else:

                    pass

            to_keep = set(to_keep) - set(to_drop)

        if (len(list(set(dataset.index))) - len(list(to_keep)) == 0):

            df = dataset

        else:

            df = dataset.loc[list(to_keep)]

        if df.empty:

            print('No record from the dataset satisfied the patterns!')

            print('Will return empty dataset - please change our patterns')

        return df

    def transform(self):

        ccd = self.dataset

        start_time = time.time()

        print(">>Consistency checking")

        for key in ['train', 'test']:

            print("* For", key, "dataset")

            if (not isinstance(self.dataset[key], dict)):

                dn = self.dataset[key]

                if (self.strategy == "CC"):

                    dn = self.CC_constraint_checking(dataset=dn,
                                                     file_name=self.file_name,
                                                     verbose=self.verbose)

                elif (self.strategy == "PC"):

                    dn = self.PC_pattern_checking(dataset=dn,
                                                  file_name=self.file_name,
                                                  verbose=self.verbose)

                else:

                    raise ValueError("Strategy invalid. Please choose between "
                                     "'CC'  or 'PC' ")

                ccd[key] = dn

        print("Consistency checking done -- CPU time: %s seconds" %
              (time.time() - start_time))

        return ccd
