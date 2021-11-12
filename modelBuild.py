from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from os import walk, path

# creating filename list with their path

# relative path from current directory
trainingRelativePath = "TrainingData/"
# absolute path to current directory
trainingScriptPath = path.dirname(__file__)
# full path to the target path
trainingFilePath = path.join(trainingScriptPath, trainingRelativePath)
# the list use to save filenames
trainingFiles = []
tempF = "tempData.csv"

# list all filenames of training data
for dirPath, _, fileName in walk(trainingFilePath):
    trainingFiles.extend([dirPath + temp for temp in fileName])
    break

# open a temporary csv file to save all processed input csv file
if path.isfile("tempData.csv"):
    tempF = input(
        "Default file name for temporary csv exists, please enter a non-exist one: ") + ".csv"
    tempCSV = open(tempF, mode="w")
else:
    tempF = "tempData.csv"
    tempCSV = open("tempData.csv", mode="w")

for txt in trainingFiles:
    with open(txt) as fp:
        # start time counting to calculate the time requires to read each file
        start = datetime.now()
        print("Starting reading file : {0}!".format(txt))
        for lines in fp:
            # eliminate space and line change
            lines = lines.strip()
            # if the line end with ":" means the line denote the movie id
            if ":" in lines:
                movieId = int(lines.replace(":", ""))
            # reading each row of data and write into the csv file
            else:
                userid, rating, date = lines.split(",")
                if int(rating) >= 3:
                    rating = "1"
                else:
                    rating = "0"
                year, month, day = date.split("-")
                row = str(movieId) + "," + userid + "," + \
                    rating + "," + year + "," + month + "\n"
                tempCSV.write(row)
        # end of reading this file
        print("File : {0} is being read! Used Time : {1}".format(
            txt, datetime.now() - start))

tempCSV.close()


print("\nProceessing data with proper data structure")
# build pandas data frame
pdData = pd.read_csv(tempF, sep=",", names=[
                     "MovieID", "UserID", "Ratings", "Year", "Month"])
#pdData = pd.DataFrame(data, columns=["MovieID", "UserID", "Ratings", "Year", "Month"])

# use numpy to transform to array
dataPart = np.asanyarray(pdData[["MovieID", "UserID", "Year", "Month"]])
labelPart = np.asanyarray(pdData["Ratings"])

print("Data processing is done")
print("The total number of data entries is : {0}\n".format(pdData.shape[0]))

# split data
dataForTrain, dataForTest, labelForTrain, labelForTest = train_test_split(
    dataPart, labelPart, test_size=0.1)

"""
KNN part
"""

numOfNeighbors = 30

knnModel = KNeighborsClassifier(n_neighbors=numOfNeighbors)

# Training the model
print("Start to build KNN model")
start = datetime.now()
knnModel.fit(dataPart, labelPart)
print("KNN model building is done! Total time used: {0}".format(
    datetime.now() - start))

# Build a pickle file to indicated folder
print("Enter your desired file name of the KNN model: ")
inputFileName = input() + ".pkl"
print("Start to create the model file...")
start = datetime.now()
pickle.dump(knnModel, open(inputFileName, "wb"))
print("The pickle model is saved! Total time used: {0}\n".format(
    datetime.now() - start))

# calculate the accuracy
print("Start to calculate the accuracy")
accuracyKNN = knnModel.score(dataForTest, labelForTest)
print("Finished! Total time used: {0}".format(datetime.now() - start))

print("The accuracy of K Nearest Neighbor model with K = {0} is {1}\n".format(
    numOfNeighbors, accuracyKNN))
