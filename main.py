# Relevant imports
import pandas as pd
# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

import Constants
import Constants as constants
from BatchGenerator import BatchGenerator

# Exercise is strictly tf1.
tf.disable_v2_behavior()

features = constants.imageSize * constants.imageSize  # used in initTF!


def initTF(session):
    # delete the current graph
    # tf.reset_default_graph()

    # import the graph from the file
    # imported_graph = tf.train.import_meta_graph('saved_variable.meta') <<<<<<<<

    with tf.variable_scope(tf.get_variable_scope()):
        W = tf.get_variable(name="W", shape=[features, constants.categories],
                            initializer=tf.zeros_initializer, trainable=True)
        b = tf.get_variable(name="b", shape=[constants.categories],
                            initializer=tf.zeros_initializer, trainable=True)

    session.run(tf.global_variables_initializer())
    # imported_graph.restore(session, savePath)

    # create saver object
    saver = tf.train.Saver()
    try:
        saver.restore(session, constants.savePath)  # TODO: this might need to be explicit (see line 32 <<<)
    except:
        print("saver.restore failed to restore!")

    return W, b, saver


def labelToTensor(y) -> None:
    for i in range(len(y)):
        if y[i][0] == 'Pneumonia':
            y[i] = [1, 0, 0, 0]
        elif y[i][0] == 'Tuberculosis':
            y[i] = [0, 1, 0, 0]
        elif y[i][0] == 'Covid':
            y[i] = [0, 0, 1, 0]
        elif y[i][0] == 'Normal':
            y[i] = [0, 0, 0, 1]


def initGraphValues(batch):
    x = []
    y = []
    for i in range(len(batch)):
        x.append(batch[i].tolist().copy())
        y.append([batch[i][-1]])
        x[i].pop()

    labelToTensor(y)

    return x, y


def trainOperation(W, b, session, batch):
    # create a single batch
    inputData, desiredOutput = initGraphValues(batch)
    if len(inputData) == 0 or len(desiredOutput) == 0:
        return False

    x = tf.placeholder(tf.float32, [None, constants.imageSize * Constants.imageSize])
    y = tf.placeholder(tf.float32, [None, constants.categories])
    z = tf.matmul(x, W) + b
    # pred = tf.nn.softmax(z)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, z))
    # loss = tf.reduce_mean(y * tf.log(pred))
    update = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    printControl = constants.iterationsPerBatch / constants.iterationDivider
    for i in range(constants.iterationsPerBatch):
        session.run(update, feed_dict={x: inputData, y: desiredOutput})  # Batch Gradeient Descent

        currLoss = loss.eval(session=session, feed_dict={x: inputData, y: desiredOutput})
        if i % printControl == 0:
            print("iteration " + str(i) + " loss: " + str(currLoss))
        if currLoss < 1:
            print("iteration " + str(i) + " loss: " + str(currLoss))
            break
    # tf.saved_model.simple_save(session, constants.export_dir, inputs={"x": x, "y": y}, outputs={"z": z})
    return True


def batchRunner(session, train_dataframes, bg):
    """
    The method trains the model on a generated batch
    :param train_dataframes: dataframe array
    :param test_dataframes: dataframe array
    :param bg: batch generator
    :param training: True or False
    :return: None
    """

    (W, b, saver) = initTF(session)
    numOfBatches = (len(train_dataframes) // constants.batchSize) + 1

    for k in range(numOfBatches + 1):
        try:
            # Generate a batch
            batch = bg.getRandomBatch()
            if len(batch) == 0:
                print("Empty batch from generator - saving and stopping!")
                saver.save(session, constants.savePath)
                break

            # then run the model on it
            success = trainOperation(W, b, session, batch)

            if success:
                print("\n\nBatch " + str(k) + " Finished out of: " + str(numOfBatches) + "\n\n")
                saver.save(session, constants.savePath)
        except Exception as e:
            print("Exception in batchRunner! :\n\t", e)
            return


def testOperation(session, test_dataframes, bg):
    (W, b, saver) = initTF(session)
    # create a single batch
    batch = bg.getRandomBatch()
    inputData, desiredOutput = initGraphValues(batch)
    if len(inputData) == 0 or len(desiredOutput) == 0:
        return False

    x = tf.placeholder(tf.float32, [None, constants.imageSize * Constants.imageSize])
    y = tf.placeholder(tf.float32, [None, constants.categories])
    z = tf.matmul(x, W) + b
    pred = tf.nn.softmax(z)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, z))
    # loss = tf.reduce_mean(y * tf.log(pred))
    print("Loss:\n\t"+loss+"\n\n", "Prediction:\n\t"+pred+"\n")


if __name__ == '__main__':
    training = True

    with tf.Session() as session:
        # train_dataframes = pd.read_csv(constants.train16DfPath)
        test_dataframes = pd.read_csv(constants.test16DfPath)

        # train_dataframes = train_dataframes.drop(['Unnamed: 0'], axis=1)
        test_dataframes = test_dataframes.drop(['Unnamed: 0'], axis=1)
        # bg = BatchGenerator(constants.batchSize, train_dataframes)
        # batchRunner(session, train_dataframes, bg)  # train
        bg = BatchGenerator(constants.batchSize, test_dataframes)
        testOperation(session, test_dataframes, bg)  # test
