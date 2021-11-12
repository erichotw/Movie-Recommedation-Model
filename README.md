# Movie Recommedation Model

Dataset source : https://www.kaggle.com/netflix-inc/netflix-prize-data

## Introduction

This project is building a KNN model based on the dataset from Kaggle.

### modelBuild.py

In this file, the program will read all files in the folder **"TrainingData"**.

After reading all the input data, the program will build a model by using Scikit Learn Library.

An pickle file will be generated to save the model.

If you use all original data files in Kaggle to build the model :

1. It will require up to 20 minutes to read, process and build the model.

2. The pickle file size can up to several GBs.

### predict.py

In this file, the program will read all files in folder **"TestingData"** and pickle model produced by **modelBuild.py**.

After the program fininsh predicting, it will write the result into a text file.

If the model predict the rating will >= 3, then the label will be **"Recommended"** while others will be **"Not Recommended"**.
