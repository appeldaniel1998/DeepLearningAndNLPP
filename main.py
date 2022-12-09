from BatchGenerator import BatchGenerator
# Relevant imports
import os as os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.metrics import accuracy_score

tf.disable_v2_behavior()

def eval2():
    # delete the current graph
    tf.reset_default_graph()

    # import the graph from the file
    # imported_graph = tf.train.import_meta_graph('saved_variable.meta')

    categories = 4
    W = tf.Variable(tf.zeros([10000, categories]))
    b = tf.Variable(tf.zeros([categories]))

    # sess = tf.Session()
    # imported_graph.restore(sess, './saved_variable')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "./saved_variable")
        print("W : %s" % W.eval())
        print("b : %s" % b.eval())

    # create saver object

    # saver.restore(sess, './saved_variable')
    # numOfBatches = (len(dfTrainArray) // batchSize) + 2

    x = []
    y = []
    for i in range(len(dfTestArray)):
        x.append(dfTestArray[i].tolist().copy())
        y.append([dfTestArray[i][-1]])
        x[i].pop()

    for i in range(len(y)):
        if y[i][0] == 'Pneumonia':
            y[i] = [1, 0, 0, 0]
        elif y[i][0] == 'Tuberculosis':
            y[i] = [0, 1, 0, 0]
        elif y[i][0] == 'Covid':
            y[i] = [0, 0, 1, 0]
        elif y[i][0] == 'Normal':
            y[i] = [0, 0, 0, 1]

    sess.run(tf.global_variables_initializer())

    # categories = 4
    x_ = tf.placeholder(tf.float32, [None, len(x[0])])
    y_ = tf.placeholder(tf.float32, [None, categories])

    z = tf.matmul(x_, W) + b
    pred = tf.nn.softmax(tf.matmul(x_, W) + b)
    # sess.run(tf.global_variables_initializer())

    predictions = []
    for i in range(len(x)):
        predictions.append(np.matmul(x[i], sess.run(W)) + sess.run(b))
    print(predictions + "\n\n")
    print(accuracy_score(y, predictions))


def run(dfTrainArray, batchSize, bg):
    # delete the current graph
    tf.reset_default_graph()

    # import the graph from the file
    imported_graph = tf.train.import_meta_graph('saved_variable.meta')

    flag = True
    categories = 4
    W = tf.Variable(tf.zeros([10000, categories]))
    b = tf.Variable(tf.zeros([categories]))

    sess = tf.Session()
    imported_graph.restore(sess, './saved_variable')

    # create saver object
    saver = tf.train.Saver()
    numOfBatches = (len(dfTrainArray) // batchSize) + 2
    for k in range(numOfBatches):
        try:
            # create a single batch
            batch = bg.getRandomBatch()

            lst = []
            for i in range(len(batch)):
                lst.append(batch[i].tolist())

            x = []
            y = []
            for i in range(len(batch)):
                x.append(batch[i].tolist().copy())
                y.append([batch[i][-1]])
                x[i].pop()

            for i in range(len(y)):
                if y[i][0] == 'Pneumonia':
                    y[i] = [1, 0, 0, 0]
                elif y[i][0] == 'Tuberculosis':
                    y[i] = [0, 1, 0, 0]
                elif y[i][0] == 'Covid':
                    y[i] = [0, 0, 1, 0]
                elif y[i][0] == 'Normal':
                    y[i] = [0, 0, 0, 1]

            # categories = 4
            x_ = tf.placeholder(tf.float32, [None, len(x[0])])
            y_ = tf.placeholder(tf.float32, [None, categories])

            # W = tf.Variable(tf.zeros([len(x[0]), categories]))
            # b = tf.Variable(tf.zeros([categories]))
            z = tf.matmul(x_, W) + b
            pred = tf.nn.softmax(tf.matmul(x_, W) + b)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y_, z))
            update = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

            sess.run(tf.global_variables_initializer())
            for i in range(iterationsPerBatch):
                sess.run(update, feed_dict={x_: x, y_: y})  # BGD

                currLoss = loss.eval(session=sess, feed_dict={x_: x, y_: y})
                if i % 50 == 0:
                    print("iteration " + str(i) + " loss: " + str(currLoss))
                if currLoss < 1:
                    print("iteration " + str(i) + " loss: " + str(currLoss))
                    break
            # print('\n W:', sess.run(W)[:10], ' b:', sess.run(b)[:10], "\n")
            print("\n\nBatch " + str(k) + " Finished out of: " + str(numOfBatches) + "\n\n")

            # save the variable in the disk
            saved_path = saver.save(sess, './saved_variable')
        except:
            pass


def eval():
    # import the graph from the file
    imported_graph = tf.train.import_meta_graph('saved_variable.meta')

    categories = 4
    W = tf.Variable(tf.zeros([10000, categories]))
    b = tf.Variable(tf.zeros([categories]))



    sess = tf.Session()
    imported_graph.restore(sess, './saved_variable')
    sess.run(tf.global_variables_initializer())

    WNew, bNew = sess.run([W, b])
    print(0)


if __name__ == '__main__':
    # dfTrain = pd.read_csv('Data/dfTrain.csv')
    # dfTrain = dfTrain.drop(['Unnamed: 0'], axis=1)

    dfTest = pd.read_csv('Data/dfTest.csv')
    dfTest = dfTest.drop(['Unnamed: 0'], axis=1)

    # Dataframe to numpy arrays
    dfTestArray = dfTest.to_numpy()
    # dfTrainArray = dfTrain.to_numpy()

    batchSize = 200
    iterationsPerBatch = 500

    eval2()

    # for i in range(50):
    #     bg = BatchGenerator(batchSize, dfTrainArray)
    #     run(dfTrainArray, batchSize, bg)





    # for i in range(iterationsPerBatch):
    #     sess.run(update, feed_dict={x_: x, y_: y})  # BGD
    #
    #     currLoss = loss.eval(session=sess, feed_dict={x_: x, y_: y})
    #     if i % 50 == 0:
    #         print("iteration " + str(i) + " loss: " + str(currLoss))
    #     if currLoss < 1:
    #         print("iteration " + str(i) + " loss: " + str(currLoss))
    #         break
    # # print('\n W:', sess.run(W)[:10], ' b:', sess.run(b)[:10], "\n")
    # print("\n\nBatch " + str(k) + " Finished out of: " + str(numOfBatches) + "\n\n")
    #
    # # save the variable in the disk
    # saved_path = saver.save(sess, './saved_variable')



