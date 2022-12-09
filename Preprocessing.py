# Relevant imports
import os as os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd


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


if __name__ == '__main__':
    covidTestData = ['Data/test/COVID19/' + file_name for file_name in os.listdir('Data/test/COVID19/')]
    normalTestData = ['Data/test/NORMAL/' + file_name for file_name in os.listdir('Data/test/NORMAL/')]
    pneumoniaTestData = ['Data/test/PNEUMONIA/' + file_name for file_name in os.listdir('Data/test/PNEUMONIA/')]
    tuberculosisTestData = ['Data/test/TUBERCULOSIS/' + file_name for file_name in os.listdir('Data/test/TUBERCULOSIS/')]

    covidTrainData = ['Data/train/COVID19/' + file_name for file_name in os.listdir('Data/train/COVID19/')]
    normalTrainData = ['Data/train/NORMAL/' + file_name for file_name in os.listdir('Data/train/NORMAL/')]
    pneumoniaTrainData = ['Data/train/PNEUMONIA/' + file_name for file_name in os.listdir('Data/train/PNEUMONIA/')]
    tuberculosisTrainData = ['Data/train/TUBERCULOSIS/' + file_name for file_name in os.listdir('Data/train/TUBERCULOSIS/')]

    covidValidationData = ['Data/val/COVID19/' + file_name for file_name in os.listdir('Data/val/COVID19/')]
    normalValidationData = ['Data/val/NORMAL/' + file_name for file_name in os.listdir('Data/val/NORMAL/')]
    pneumoniaValidationData = ['Data/val/PNEUMONIA/' + file_name for file_name in os.listdir('Data/val/PNEUMONIA/')]
    tuberculosisValidationData = ['Data/val/TUBERCULOSIS/' + file_name for file_name in os.listdir('Data/val/TUBERCULOSIS/')]
    showPics(covidValidationData, cv2.IMREAD_COLOR)
    showPics(normalValidationData, cv2.IMREAD_COLOR)
