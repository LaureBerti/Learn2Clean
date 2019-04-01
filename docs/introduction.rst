Getting started
===============

Learn2Clean main package contains the following sub-packages for data preprocessing: **loading**, **normalization**, **feature-selection**, **outlier-detection**, **duplicate-detection**, **consistency-checking**,  **imputation** and **qlearning**. And ML packages for **clustering**, **classification**, and **regression**.


**Here are a few lines to import Learn2Clean:**

.. code-block:: python 

  import learn2clean.normalization.normalizer as nl 
  import learn2clean.feature_selection.feature_selector as fs
  import learn2clean.duplicate_detection.duplicate_detector as dd
  import learn2clean.outlier_detection.outlier_detector as od
  import learn2clean.imputation.imputer as imp
  import learn2clean.classification.classifier as cl


**Then, you need to give :** 

* the list of paths to your train datasets and test datasets
* the name of the target you try to predict (classification or regression)
* or you can submit only one dataset and Leanr2Clean will split it into train and test datasets

.. code-block:: python 

   paths = ["<file_1>.csv", "<file_2>.csv", ..., "<file_n>.csv"] #to modify
   target_name = "<my_target>" #to modify


**Now, you can write your own pipeline**

... to preprocess your files : 

.. code-block:: python 

   data = rd.Reader(sep=',',verbose = True, encoding = True).train_test_split(paths, target_name) #reading
   # replace numerical missing values by median
   d1 = imp.Imputer(dataset = data, strategy = 'MEDIAN', verbose = False).transform()
   # decimal scaling for numerical variables
   d2 = nl.Normalizer(dataset = d1, strategy = 'DS', exclude = None, verbose = False).transform()
   # eliminate 20 LOF outliers
   d3 = od.Outlier_detector(dataset = d2, strategy = 'LOF', threshold= 0.2, verbose= False).transform()
   # classify with LDA
   cl.Classifier(dataset = d3, strategy = 'LDA', target = target_name, verbose = True).transform()


... or to test the automatic optimization the preprocessing pipeline by Learn2Clean based on Qlearning:


.. code-block:: python 

   import learn2clean.qlearning.qlearner as ql
   l2c_classif=ql.Qlearner(dataset = data, goal = 'LDA',target_goal = target_name,
      target_prepare=None, file_name = 'results_file_name', verbose = False)
   l2c_classif.learn2clean()

... finally, Learn2Clean will select the best preprocessing strategy for the given ML task and its by-default quality metric.

**That's all !** You can have a look at the Jupyter notebook examples in the folder "examples" and also the folder "save" where you can find :

* the results in 'results_file_name' including the best processing strategy found by Learn2Clean,
* the results of random preprocessing and no-preprocessing,
* and the discovered patterns and constraints from the data by the Consistency_checker
