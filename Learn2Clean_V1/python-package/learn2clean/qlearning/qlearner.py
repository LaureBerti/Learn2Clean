#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille <laure.berti@ird.fr>

import warnings
import time
import numpy as np
import re
import random
from random import randint

from ..normalization.normalizer import Normalizer
from ..duplicate_detection.duplicate_detector import Duplicate_detector
from ..outlier_detection.outlier_detector import Outlier_detector
from ..consistency_checking.consistency_checker import Consistency_checker
from ..imputation.imputer import Imputer
from ..feature_selection.feature_selector import Feature_selector
from ..regression.regressor import Regressor
from ..clustering.clusterer import Clusterer
from ..classification.classifier import Classifier


def update_q(q, r, state, next_state, action, beta, gamma):

    rsa = r[state, action]

    qsa = q[state, action]

    new_q = qsa + beta * (rsa + gamma * max(q[next_state, :]) - qsa)

    q[state, action] = new_q
    # renormalize row to be between 0 and 1
    rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])

    q[state][q[state] > 0] = rn

    return r[state, action]


def remove_adjacent(nums):

    previous = ''

    for i in nums[:]:  # using the copy of nums

        if i == previous:

            nums.remove(i)

        else:

            previous = i

    return nums


class Qlearner():
    """
    Learn2clean class with Qlearning for data preparation plus random cleaning
    and no-preparation functions

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test'] pandas
        DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * goal: str, default = 'HCA' to define the ML method and task:
        classification, clustering or regression
        The choice for the goal :
        - 'NB', 'LDA', 'CART' and 'MNB' for classification
        - 'HCA' or 'KMEANS' for clustering
        - 'MARS, 'LASSO or 'OLS'  for regression

    * target_goal: str, name of the target variable encoded as int64 from
        dataset['target'] pandas DataSeries

    * target_prepare: str, name of the variable that should not be excluded
        from data preparation
    * verbose: Boolean,  default = 'False'
    """

    def __init__(self, dataset, goal, target_goal, target_prepare,
                 verbose=False, file_name=None, threshold=None):

        self.dataset = dataset

        self.goal = goal

        self.target_goal = target_goal

        self.target_prepare = target_prepare

        self.verbose = verbose

        self.file_name = file_name

        self.threshold = threshold  # TODO: handle an array of thresholds

    def get_params(self, deep=True):

        return {'goal': self.goal,

                'target_goal': self.target_goal,

                'target_prepare': self.target_prepare,

                'verbose': self.verbose,

                'file_name': self.file_name,

                'threshold': self.threshold

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`qlearner.get_params().keys()`")

            else:

                setattr(self, k, v)

    def Initialization_Reward_Matrix(self, dataset):
        """ defines the reward/connection graph between 18 preprocessing
            methods and 1 ML model : 19x19 matrix if missing values
         4 (MICE EM KNN MF) for imputation
         3 (DS MM ZS) for normalization
         4 (MR WR LC TB) for feature selection
         3 (ZSB LOF IQR) for outlier detection
         2 (CC PC) for inconsistency checking
         2 (AD ED) for duplication detection
         1 (LASSO or OLS or MARS) regression or (HCA or KMEANS) for clustering
         or (CART or LDA or NB) for classification
         """

        if dataset['train'].copy().isnull().sum().sum() > 0:

            r = np.array([
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],

                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0,  0,  0,  0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 -1],

                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],

                [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0,
                 0, 100],
                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0,
                 0, 100],
                [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0,
                 0, 100],

                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, 0, 0, -1, -1,
                 0, 0, 100],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, 0, 0, -1, -1,
                 0, 0, 100],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1]]).astype("float32")

            n_actions = 19

            n_states = 19

            check_missing = True

        else:  # no imputation needed
            """defines the reward/connection graph between 18 preprocessing
            methods and 1 ML model : 14x14 if no missing values
            3 (DS MM ZS) for normalization
            3 (WR LC TB) for feature selection
            3 (ZSB LOF IQR) for outlier detection
            2 (CC PC) for inconsistency checking
            2 (AD ED) for duplication detection
            1 (LASSO or OLS or MARS) regression or (HCA or KMEANS)
            for clustering or (CART or LDA or NB) for classification
            """

            r = np.array([
                          [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1],
                          [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1],
                          [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1],

                          [0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],
                          [0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],
                          [0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, -1],

                          [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 100],
                          [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 100],
                          [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 100],

                          [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, 100],
                          [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, 100],

                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],

                          [-1, -1, -1, -1, -1, -1, -1, -1, -1,  -1, -1, -1, -1,
                           -1]]).astype("float32")

            n_actions = 14

            n_states = 14

            check_missing = False

        q = np.zeros_like(r)

        # we prevent the transition from any ML model LASSO OLSR
        # MARS HCA KMEAN CART LDA NB (last rows) to preprocessing
        r = r[~np.all(r == -1, axis=1)]

        if self.verbose:

            print("Reward matrix")

            print(r)

        return q, r, n_actions, n_states, check_missing

    def pipeline(self, dataset, actions_list, target_goal,
                 target_prepare, check_missing):

        dataset = dataset.copy()

        goals_name = ["LASSO", "OLS", "MARS",
                      "HCA", "KMEANS", "CART", "LDA", "NB"]

        res = None

        if check_missing:

            actions_name = ["MICE", "EM", "KNN", "MF",
                            "DS", "MM", "ZS",
                            "MR", "WR", "LC", "Tree",
                            "ZSB", "LOF", "IQR",
                            "CC", "PC",
                            "ED", "AD"]

            L2C_class = [Imputer, Imputer, Imputer, Imputer,
                         Normalizer, Normalizer, Normalizer,
                         Feature_selector, Feature_selector, Feature_selector,
                         Feature_selector,
                         Outlier_detector, Outlier_detector, Outlier_detector,
                         Consistency_checker, Consistency_checker,
                         Duplicate_detector, Duplicate_detector,
                         Regressor, Regressor, Regressor,
                         Clusterer, Clusterer,
                         Classifier, Classifier, Classifier]

        else:

            actions_name = ["DS", "MM", "ZS",
                            "WR", "LC", "Tree",
                            "ZSB", "LOF", "IQR",  "CC",
                            "PC", "ED", "AD"]

            L2C_class = [Normalizer, Normalizer, Normalizer,
                         Feature_selector, Feature_selector, Feature_selector,
                         Outlier_detector, Outlier_detector, Outlier_detector,
                         Consistency_checker, Consistency_checker,
                         Duplicate_detector, Duplicate_detector,
                         Regressor, Regressor, Regressor,
                         Clusterer, Clusterer,
                         Classifier, Classifier, Classifier]

        print()

        print("Start pipeline")

        print("-------------")

        start_time = time.time()

        n = None

        for a in actions_list:

            if not check_missing:

                if a in range(0, 5):
                    # normalization 0-2, and feature selection 3-5

                    n = L2C_class[a](dataset=dataset, strategy=actions_name[a],
                                     exclude=self.target_prepare,
                                     verbose=self.verbose).transform()

                if a in (6, 7, 8, 11, 12):
                    # 6-8: Outlier detection
                    # and 11-12: duplicate detection

                    n = L2C_class[a](dataset=dataset, strategy=actions_name[a],
                                     verbose=self.verbose).transform()

                if a in (9, 10):
                    # Consistency checking

                    n = L2C_class[a](dataset=dataset, strategy=actions_name[a],
                                     file_name=self.file_name,
                                     verbose=self.verbose).transform()

                if a in (13, 14, 15, 18, 19, 20):
                    # 3 regression and 3 classification methods

                    res = L2C_class[a](dataset=dataset,
                                       strategy=goals_name[a -
                                                           len(actions_name)],
                                       target=self.target_goal,
                                       verbose=self.verbose).transform()

                if a in (16, 17):
                    # clustering

                    res = L2C_class[a](dataset=dataset,
                                       strategy=goals_name[a -
                                                           len(actions_name)],
                                       verbose=self.verbose).transform()

            else:

                if (len(dataset['train'].dropna()) == 0) or \
                        (len(dataset['test'].dropna()) == 0):

                    pass

                else:

                    if a in (0, 1, 2, 3):
                        # 0-3: Imputation

                        n = L2C_class[a](dataset=dataset,
                                         strategy=actions_name[a],
                                         verbose=self.verbose).transform()

                    if a in range(4, 10):
                        # normalization and feature selection
                        n = L2C_class[a](dataset=dataset,
                                         strategy=actions_name[a],
                                         exclude=self.target_prepare,
                                         verbose=self.verbose).transform()

                    if a in (11, 12, 13, 16, 17):
                        # 11-13: Outlier detection
                        # and 16-17: duplicate detection

                        n = L2C_class[a](dataset=dataset,
                                         strategy=actions_name[a],
                                         verbose=self.verbose).transform()

                    if a in (14, 15):
                        # Consistency checking

                        n = L2C_class[a](dataset=dataset,
                                         strategy=actions_name[a],
                                         file_name=self.file_name,
                                         verbose=self.verbose).transform()

                    if a in (18, 19, 20, 23, 24, 25):
                        # 3 regression and 3 classification methods

                        a_new = a-len(actions_name)
                        res = L2C_class[a](dataset=dataset,
                                           strategy=goals_name[a_new],
                                           target=self.target_goal,
                                           verbose=self.verbose).transform()

                    if a in (21, 22):
                        # clustering
                        a_new = a-len(actions_name)
                        res = L2C_class[a](dataset=dataset,
                                           strategy=goals_name[a_new],
                                           verbose=self.verbose).transform()

        t = time.time() - start_time

        print("End Pipeline CPU time: %s seconds" % (time.time() - start_time))

        return n, res, t

    def show_traverse(self, dataset, q, g, target1, target2, check_missing):
        # show all the greedy traversals

        if check_missing:

            methods, goals = ["MICE", "EM", "KNN", "MF",
                              "DS", "MM", "ZS", "MR", "WR", "LC", "Tree",
                              "ZSB", "LOF", "IQR", "CC", "PC",
                              "ED", "AD"], ["LASSO", "OLS", "MARS",
                                            "HCA", "KMEANS", "CART",
                                            "LDA", "NB"]

        else:

            methods, goals = ["DS", "MM", "ZS",
                              "WR", "LC", "Tree",
                              "ZSB", "LOF", "IQR",
                              "CC", "PC",
                              "ED", "AD"], ["LASSO", "OLS", "MARS",
                                            "HCA", "KMEANS",
                                            "CART", "LDA", "NB"]

        n_states = len(methods) + 1

        methods.append(str(goals[g]))

        strategy = []

        actions_strategy = []

        for i in range(len(q)-1):

            actions_list = []

            current_state = i

            current_state_name = methods[i]
            # traverse = "%i -> " % current_state

            traverse_name = "%s -> " % current_state_name

            n_steps = 0

            while current_state != n_states-1 and n_steps < 20:

                actions_list.append(current_state)

                next_state = np.argmax(q[current_state])

                current_state = next_state

                current_state_name = methods[next_state]
                # traverse += "%i -> " % current_state

                traverse_name += "%s -> " % current_state_name

                actions_list.append(next_state)

                n_steps = n_steps + 1

                actions_list = remove_adjacent(actions_list)

            if not check_missing:

                traverse_name = traverse_name[:-4]

                del actions_list[-1]
                actions_list.append(g+len(methods)-1)

            else:

                del actions_list[-1]

                actions_list.append(g+len(methods)-1)

                traverse_name = traverse_name[:-4]

            print("\n\nStrategy#", i, ": Greedy traversal for "
                  "starting state %s" % methods[i])

            print(traverse_name)

            # print(actions_list)

            actions_strategy.append(traverse_name)

            strategy.append(self.pipeline(dataset, actions_list, target1,
                                          target2, check_missing)[1])

        strategy.append(self.pipeline(dataset, [g+len(methods)-1], target1,
                                      target2, check_missing)[1])

        print()

        print("==== Recap ====\n")

        print("List of strategies tried by Learn2Clean:")

        print(actions_strategy)

        print('\nList of corresponding quality metrics ****\n',
              strategy)

        print()

        return actions_strategy, strategy

    def learn2clean(self):

        goals = ["LASSO", "OLS", "MARS", "HCA", "KMEANS", "CART", "LDA", "NB"]

        if self.goal not in goals:

            raise ValueError("Goal invalid. Please choose between "
                             "'LASSO', 'OLS', 'MARS', for regression "
                             "'HCA' or 'KMEANS' for clustering "
                             "'CART', 'LDA', or 'NB' for classification.")

        else:

            g = goals.index(self.goal)

        if self.target_goal != self.dataset['target'].name:

            raise ValueError("Target variable invalid.")

        else:

            pass

        start_l2c = time.time()

        print("Start Learn2Clean")

        gamma = 0.8

        beta = 1.

        n_episodes = 1E3

        epsilon = 0.05

        random_state = np.random.RandomState(1999)

        q, r, n_actions, n_states, \
            check_missing = self.Initialization_Reward_Matrix(self.dataset)

        for e in range(int(n_episodes)):

            states = list(range(n_states))

            random_state.shuffle(states)

            current_state = states[0]

            goal = False

            if e % int(n_episodes / 10.) == 0 and e > 0:

                pass

            while (not goal) and (current_state != n_states-1):

                # epsilon greedy
                valid_moves = r[current_state] >= 0

                if random_state.rand() < epsilon:

                    actions = np.array(list(range(n_actions)))

                    actions = actions[valid_moves]

                    if type(actions) is int:

                        actions = [actions]

                    random_state.shuffle(actions)

                    action = actions[0]

                    next_state = action

                else:

                    if np.sum(q[current_state]) > 0:

                        action = np.argmax(q[current_state])

                    else:

                        actions = np.array(list(range(n_actions)))

                        actions = actions[valid_moves]

                        random_state.shuffle(actions)

                        action = actions[0]

                    next_state = action

                reward = update_q(q, r, current_state,
                                  next_state, action, beta, gamma)

                if reward > 1:

                    goal = True

                np.delete(states, current_state)

                current_state = next_state

        if self.verbose:

            print("Q-value matrix\n", q)

        print("Learn2Clean - Pipeline construction -- CPU time: %s seconds"
              % (time.time() - start_l2c))

        metrics_name = ["MSE", "MSE", "MSE", "silhouette",
                        "silhouette", "accuracy", "accuracy", "accuracy"]

        print("=== Start Pipeline Execution ===")

        start_pipexec = time.time()

        result_list = self.show_traverse(self.dataset, q, g, self.target_goal,
                                         self.target_prepare, check_missing)

        quality_metric_list = []

        if result_list[1]:

            for dic in range(len(result_list[1])):

                for key, val in result_list[1][dic].items():

                    if key == 'quality_metric':

                        quality_metric_list.append(val)

            if g in range(0, 2):

                result = min(x for x in quality_metric_list if x is not None)

                result_l = quality_metric_list.index(result)

                result_list[0].append(goals[g])

                print("Strategy", result_list[0][result_l], 'for minimal MSE ',
                      result, 'for', goals[g])

                print()

            else:

                result = max(x for x in quality_metric_list if x is not None)

                result_l = quality_metric_list.index(result)

                result_list[0].append(goals[g])

                print("Strategy", result_list[0][result_l], 'for maximal',
                      metrics_name[g], ':', result, 'for', goals[g])

                print()

        else:

            result = None

            result_l = None

        t = time.time() - start_pipexec

        print("=== End of Learn2Clean - Pipeline execution "
              "-- CPU time: %s seconds" % t)

        print()

        if result_l is not None:

            rr = (self.file_name, "learn2clean", goals[g], self.target_goal,
                  self.target_prepare, result_list[0][result_l],
                  metrics_name[g], result, t)

        else:

            rr = (self.file_name, "learn2clean", goals[g], self.target_goal,
                  self.target_prepare, None, metrics_name[g], result, t)

        print("**** Best strategy ****")

        print(rr)

        with open('./save/'+self.file_name+'_results.txt',
                  mode='a') as rr_file:

            print("{}".format(rr), file=rr_file)

    def random_cleaning(self, dataset_name):

        random.seed(time.clock())

        # d.drop('Id', axis=1)
        check_missing = self.dataset['train'].isnull().sum().sum() > 0

        if check_missing:

            methods = ["-", "MICE", "EM", "KNN", "MF", "-", "DS", "MM", "ZS",
                       "-", "MR", "WR", "LC", "Tree",
                       "-", "ZSB", "LOF", "IQR",
                       "-",  "-", "-",
                       "-",  "ED", "-"]

            rand_actions_list = [randint(0, 3), randint(4, 8), randint(9, 13),
                                 randint(14, 17), randint(18, 20),
                                 randint(21, 23)]

        else:

            methods = ["-", "DS", "MM", "ZS",
                       "-",  "WR", "LC", "Tree",
                       "-",  "ZSB", "LOF", "IQR",
                       "-", "-", "-",
                       "-",  "ED", "-"]

            rand_actions_list = [randint(0, 3), randint(4, 7), randint(8, 11),
                                 randint(12, 14), randint(15, 17)]

        goals = ["LASSO", "OLS", "MARS", "HCA", "KMEANS", "CART", "LDA", "NB"]

        metrics_name = ["MSE", "MSE", "MSE", "silhouette",
                        "silhouette", "accuracy",
                        "accuracy", "accuracy"]

        if self.goal not in goals:
            raise ValueError("Goal invalid. Please choose between "
                             "'LASSO', 'OLS', 'MARS', for regression "
                             "'HCA' or 'KMEANS' for clustering "
                             "'CART', 'LDA', or 'NB' for classification.")

        else:

            g = goals.index(self.goal)

        traverse_name = methods[rand_actions_list[0]] + " -> "

        for i in range(1, len(rand_actions_list)):

            traverse_name += "%s -> " % methods[rand_actions_list[i]]

        traverse_name = re.sub('- -> ', '', traverse_name) + goals[g]

        name_list = re.sub(' -> ', ',', traverse_name).split(",")

        print()

        print()

        print("--------------------------")

        print("Random cleaning strategy:\n", traverse_name)

        print("--------------------------")

        if check_missing:

            rand_actions_list[len(rand_actions_list)-1] = g+len(methods)-6

            methods = ["MICE", "EM", "KNN", "MF",
                       "DS", "MM", "ZS",
                       "MR", "WR", "LC", "Tree",
                       "ZSB", "LOF", "IQR", "CC", "PC", "ED", "AD"]

            new_list = []

            for i in range(len(name_list)-1):

                m = methods.index(name_list[i])

                new_list.append(m)

            new_list.append(g+len(methods))

        else:

            rand_actions_list[len(rand_actions_list)-1] = g+len(methods)-5

            methods = ["DS", "MM", "ZS", "WR", "LC",
                       "Tree", "ZSB", "LOF", "IQR",
                       "CC", "PC", "ED", "AD"]
            new_list = []

            for i in range(len(name_list)-1):

                m = methods.index(name_list[i])

                new_list.append(m)

            new_list.append(g+len(methods))

        # print("New list", new_list)
        p = self.pipeline(self.dataset, new_list, self.target_goal,
                          self.target_prepare, check_missing)

        rr = (dataset_name, "random", goals[g], self.target_goal,
              self.target_prepare, traverse_name, metrics_name[g], p[1:])

        print(rr)

        if p[1] is not None:

            with open('./save/'+dataset_name+'_results.txt',
                      mode='a') as rr_file:

                print("{}".format(rr), file=rr_file)

        return p[1]

    def no_prep(self, dataset_name):

        goals = ["LASSO", "OLS", "MARS", "HCA", "KMEANS", "CART", "LDA", "NB"]

        metrics_name = ["MSE", "MSE", "MSE", "silhouette", "silhouette",
                        "accuracy", "accuracy", "accuracy"]

        if self.goal not in goals:

            raise ValueError("Goal invalid. Please choose between "
                             "'LASSO', 'OLS', 'MARS', for regression "
                             "'HCA' or 'KMEANS' for clustering "
                             "'CART', 'LDA', or 'NB' for classification.")

        else:

            g = goals.index(self.goal)

        check_missing = self.dataset['train'].isnull().sum().sum() > 0

        if check_missing:
            len_m = 18

        else:
            len_m = 13

        p = self.pipeline(self.dataset, [g+len_m], self.target_goal,
                          self.target_prepare, check_missing)

        rr = (dataset_name, "no-prep", goals[g], self.target_goal,
              self.target_prepare, goals[g], metrics_name[g], p[1:])

        if p[1] is not None:

            with open('./save/'+dataset_name+'_results.txt',
                      mode='a') as rr_file:

                print("{}".format(rr), file=rr_file)
