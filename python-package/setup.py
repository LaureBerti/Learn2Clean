#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pip
from setuptools import setup
from setuptools.command.install import install

requirements = [
    "fancyimpute == 0.4.2",
    "impyute == 0.0.7",
    "jellyfish == 0.7.1",
    "joblib == 0.13.1",
    "matplotlib == 2.2.2",
    "numpy == 1.14.3",
    "pandas == 0.23.0",
    "py_stringmatching == 0.4.0",
    "py_stringsimjoin == 0.1.0",
    "scipy == 1.1.0",
    "seaborn == 0.8.1",
    "sklearn == 0.0",
    "sklearn_contrib_py_earth == 0.1.0",
    "statsmodels == 0.9.0",
    "tdda == 1.0.13"
]


class OverrideInstallCommand(install):
    def run(self):
        # Install all requirements
        failed = []

        # for req in requirements:
            # if pip.main(["install", req]) == 1:
            #    failed.append(req)

        if len(failed) > 0:
            print("")
            print("Error installing the following packages:")
            print(str(failed))
            print("Please install them manually")
            print("")
            raise OSError("Aborting")

        # install Learn2Clean
        install.run(self)


with open("README.rst","r") as readme_file:
    readme = readme_file.read()


setup(
    name='learn2clean',
    version='0.2.1',
    description="Python Library for Data Preprocessing with Reinforcement "
    "Learning.",
    long_description=readme,
    author="Laure BERTI-EQUILLE",
    author_email='laure.berti@ird.fr',
    url='https://github.com/LaureBerti/learn2clean',
    packages=[
        'learn2clean',
        'learn2clean.loading',
        'learn2clean.normalization',
        'learn2clean.consistency_checking',
        'learn2clean.duplicate_detection',
        'learn2clean.imputation',
        'learn2clean.outlier_detection',
        'learn2clean.feature_selection',
        'learn2clean.qlearning',
        'learn2clean.classification',
        'learn2clean.regression',
        'learn2clean.clustering'],
    package_dir={
        'learn2clean': 'learn2clean',
        'learn2clean.loading': 'learn2clean/loading',
        'learn2clean.normalization': 'learn2clean/normalization',
        'learn2clean.consistency_checking': 'learn2clean/consistency_checking',
        'learn2clean.duplicate_detection': 'learn2clean/duplicate_detection',
        'learn2clean.imputation': 'learn2clean/imputation',
        'learn2clean.outlier_detection': 'learn2clean/outlier_detection',
        'learn2clean.feature_selection': 'learn2clean/feature_selection',
        'learn2clean.qlearning': 'learn2clean/qlearning',
        'learn2clean.classification': 'learn2clean/classification',
        'learn2clean.regression': 'learn2clean/regression',
        'learn2clean.clustering': 'learn2clean/clustering'},
    include_package_data=True,
    cmdclass={
        'install': OverrideInstallCommand},
    install_requires=requirements,
    zip_safe=False,
    license='BSD-3',
    keywords='learn2clean data preprocessing pipeline',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Database',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.6'],
    test_suite='tests',
    tests_require=requirements)
