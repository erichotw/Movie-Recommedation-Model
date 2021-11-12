from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import pickle
from os import walk, path

# creating filename list with their path

# relative path from current directory
testingRelativePath = "TestingData/"
# absolute path to current directory
testingScriptPath = path.dirname(__file__)
# full path to the target path
testingFilePath = path.join(testingScriptPath, testingRelativePath)
# the list use to save filenames
testingFiles = []
tempF = "testData.csv"

# list all filenames of training data
for dirPath, _, fileName in walk(testingFilePath):
    testingFiles.extend([dirPath + temp for temp in fileName])
    break

# open a temporary csv file to save all processed input csv file
if path.isfile("testData.csv"):
    tempF = input(
        "Default file name for temporary csv exists, please enter a non-exist one: ") + ".csv"
    tempCSV = open(tempF, mode="w")
else:
    tempF = "testData.csv"
    tempCSV = open("testData.csv", mode="w")

for txt in testingFiles:
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
                userid, date = lines.split(",")
                year, month, day = date.split("-")
                row = str(movieId) + "," + userid + \
                    "," + year + "," + month + "\n"
                tempCSV.write(row)
        # end of reading this file
        print("File : {0} is being read! Used Time : {1}".format(
            txt, datetime.now() - start))

tempCSV.close()


print("\nProceessing data with proper data structure")
# build pandas data frame
pdData = pd.read_csv(tempF, sep=",", names=[
                     "MovieID", "UserID", "Year", "Month"])

# use numpy to transform to array
#dataPart = np.asanyarray(pdData[["MovieID", "UserID", "Year", "Month"]])

print("Data processing is done")
print("The total number of data entries is : {0}\n".format(pdData.shape[0]))

# load the model and predict the results
print("Please enter the file name of the model")
filename = input(
    "Only provide the filename without entering extension, e.g. \".pkl\":\n") + ".pkl"
print("Start loading the model...")
start = datetime.now()
loadedModel = pickle.load(open(filename, "rb"))
print("Finished! Total time Used: {0}\n".format(datetime.now() - start))

print("Start predicting...")
start = datetime.now()
result = loadedModel.predict(pdData.values)
print("Finished! Total time used: {0}\n".format(datetime.now() - start))


print("Start to write result into text file...")
start = datetime.now()
cur = pdData.iat[0, 0]

resultFileWriter = open("Predict_Result.txt", "w")

resultFileWriter.write(str(cur) + ":\n")

for i in range(1, pdData.shape[0]):
    if cur == pdData.iat[i, 0]:
        if (result[i] == 1):
            row = "Recommended\n"
        else:
            row = "Not Recommended\n"
        resultFileWriter.write(row)
    else:
        resultFileWriter.write(str(pdData.iat[i, 0]) + ":\n")
        cur = pdData.iat[i, 0]

resultFileWriter.close()
print("Writing is done. Total time used: {0}".format(datetime.now() - start))
