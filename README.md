# Flight Delay Prediction

This repository contains code for the final project of Stanford's CS221 (Artificial Intelligence: Principles and Techniques) on predicting flight delays, applying a variety of machine learning models (Logistic Regression, SVM, Decision Tree, Random Forest, & Bayesian Networks).

## Dataset

The Kaggle Flight Delays dataset can be found [here](https://www.kaggle.com/usdot/flight-delays#flights.csv). We augmented that dataset with data from the FAA about particular airplanes as well as weather data at the time of flight departure via the DarkSky API.

## Required Packages/Software
* Python 2
* darksky
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn

## Running Code
Once the Kaggle dataset has been downloaded and the required packages have been installed, running the get_airplane_data.ipynb and getWeatherData.ipynb notebooks in the data-processing folder will pull the auxiliary data from the FAA and DarkSky, respectively. You will need to create a DarkSky account in order to extract the weather data for all of the flights. From there, running the data-merge-split.ipynb in the data-processing folder will merge the 3 datasets.

From there, you can run any of the model notebooks in the models folder to train a model and evaluate its performance.

## Authors

* **Nicholas Benavides**
* **Katherine Erdman**