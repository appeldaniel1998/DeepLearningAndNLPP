# Relevant imports
import os as os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

import Constants

# Image size
size = Constants.imageSize


def showPics(list, colOrGray):
    fig = plt.figure(figsize=(15, 6))
    for i in range(1, 5):
        rand = random.randint(0, 5)
        img = list[rand]
        img_array = cv2.imread(img, colOrGray)
        ax = fig.add_subplot(2, 4, i)
        ax.title.set_text(rand)
        plt.imshow(img_array, cmap='gray')
        plt.show()


def toDF(dfFrom, bwOrColor, dimentionX, dimentionY):
    dfTo = pd.DataFrame(columns=np.arange(size * size))
    for i in range(0, len(dfFrom)):
        imgPath = dfFrom[i]
        imgArray = cv2.imread(imgPath, bwOrColor)
        imgArray = cv2.resize(imgArray, (dimentionX, dimentionY))
        imgArray = imgArray.ravel()
        dfTo.loc[i] = imgArray
    return dfTo


if __name__ == '__main__':
    covidTestData = ['Data/test/COVID19/' + file_name for file_name in os.listdir('Data/test/COVID19/')]
    normalTestData = ['Data/test/NORMAL/' + file_name for file_name in os.listdir('Data/test/NORMAL/')]
    pneumoniaTestData = ['Data/test/PNEUMONIA/' + file_name for file_name in os.listdir('Data/test/PNEUMONIA/')]
    tuberculosisTestData = ['Data/test/TUBERCULOSIS/' + file_name for file_name in
                            os.listdir('Data/test/TUBERCULOSIS/')]

    covidTrainData = ['Data/train/COVID19/' + file_name for file_name in os.listdir('Data/train/COVID19/')]
    normalTrainData = ['Data/train/NORMAL/' + file_name for file_name in os.listdir('Data/train/NORMAL/')]
    pneumoniaTrainData = ['Data/train/PNEUMONIA/' + file_name for file_name in os.listdir('Data/train/PNEUMONIA/')]
    tuberculosisTrainData = ['Data/train/TUBERCULOSIS/' + file_name for file_name in
                             os.listdir('Data/train/TUBERCULOSIS/')]

    covidValidationData = ['Data/val/COVID19/' + file_name for file_name in os.listdir('Data/val/COVID19/')]
    normalValidationData = ['Data/val/NORMAL/' + file_name for file_name in os.listdir('Data/val/NORMAL/')]
    pneumoniaValidationData = ['Data/val/PNEUMONIA/' + file_name for file_name in os.listdir('Data/val/PNEUMONIA/')]
    tuberculosisValidationData = ['Data/val/TUBERCULOSIS/' + file_name for file_name in
                                  os.listdir('Data/val/TUBERCULOSIS/')]
    showPics(covidValidationData, cv2.IMREAD_COLOR)
    showPics(normalValidationData, cv2.IMREAD_COLOR)

    trainPneumoniaDF = toDF(pneumoniaTrainData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    trainTuberculosisDF = toDF(tuberculosisTrainData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    trainNormalDF = toDF(normalTrainData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    trainCovidDF = toDF(covidTrainData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    valNormalDF = toDF(normalValidationData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    valCovidDF = toDF(covidValidationData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    valPneumoniaDF = toDF(pneumoniaValidationData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    valTuberculosisDF = toDF(tuberculosisValidationData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    testCovidDF = toDF(covidTestData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    testNormalDF = toDF(normalTestData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    testPneumoniaDF = toDF(pneumoniaTestData, cv2.IMREAD_GRAYSCALE, size, size)
    # %%
    testTuberculosisDF = toDF(tuberculosisTestData, cv2.IMREAD_GRAYSCALE, size, size)


    # %%
    def addLabel(df, label):
        df['label'] = label


    addLabel(trainPneumoniaDF, "Pneumonia")
    addLabel(trainTuberculosisDF, "Tuberculosis")
    addLabel(trainNormalDF, "Normal")
    addLabel(trainCovidDF, "Covid")
    addLabel(valNormalDF, "Normal")
    addLabel(valCovidDF, "Covid")
    addLabel(valPneumoniaDF, "Pneumonia")
    addLabel(valTuberculosisDF, "Tuberculosis")
    addLabel(testCovidDF, "Covid")
    addLabel(testNormalDF, "Normal")
    addLabel(testPneumoniaDF, "Pneumonia")
    addLabel(testTuberculosisDF, "Tuberculosis")


    def mergeDf(lst):
        df = pd.concat(lst)
        df.index.name = 'i'  # renaming the index column so that the new index will not have the same name as the old
        df = df.reset_index()
        df = df.drop(['i'], axis=1)
        return df


    dfTest = mergeDf([testTuberculosisDF, testPneumoniaDF, testNormalDF, testCovidDF])
    dfrain = mergeDf(
        [valTuberculosisDF, valPneumoniaDF, valCovidDF, valNormalDF, trainCovidDF, trainNormalDF, trainTuberculosisDF,
         trainPneumoniaDF])
    dfrain.to_csv('Data/dfTrain.csv')
    dfTest.to_csv('Data/dfTest.csv')
    dfTrain = pd.read_csv('Data/dfTrain.csv')
    dfTrain = dfTrain.drop(['Unnamed: 0'], axis=1)
    dfTest = pd.read_csv('Data/dfTest.csv')
    dfTest = dfTest.drop(['Unnamed: 0'], axis=1)
