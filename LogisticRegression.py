# Name: Denim Datta

import decimal
import random
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


# Read the file and create dataset.
# ratio denotes the portion of dataset to be used as training data, rest are used as test data
def create_dataset(filename, ratio):
    dataset = []

    with open(filename, "r") as file:
        data = file.readlines()

    for line in data:
        values = line.rstrip().split(",")
        # dataset.append([float(values[0]), float(values[1]), float(values[2]), float(values[3]), int(values[4])])
        tmp = []
        for v in values[0:-1]:
            tmp.append(float(v))
        tmp.append(int(values[-1]))
        dataset.append(tmp)

    # ratio portion of the dataset is used as Training Data
    # rest is Test Data
    # trainlen = int(round(len(dataset) * ratio))
    traind, testd = fraction_training(dataset, ratio)

    return [traind, testd]


def fraction_training(trndata, fraction):
    tlen = int(round(len(trndata) * fraction))
    temp = list(trndata)
    sampletrndata = []
    while len(sampletrndata) < tlen:
        i = random.randrange(len(temp))
        sampletrndata.append(temp.pop(i))
    return [sampletrndata, temp]


def find_sigmoid(data, weight):
    # wSum = weight[0] + (weight[1] * data[0]) + (weight[2] * data[1]) + (weight[3] * data[2]) + (weight[4] * data[3])
    # exp = np.exp(-wSum)
    # sigmoid = (1.0 / (1 + exp))

    wSum = weight[0]
    for i in range(len(data) - 1):
        wSum += (weight[i+1] * data[i])
    sigmoid = expit(wSum)
    return sigmoid


def weight_vector_update(trndata, learning_rate, repeatation):
    # intialize weight vector (0, 0, 0, 0, 0)
    weight = [0] * len(trndata[0])
    
    # dweight will keep the d[l(w)]/d[wi]
    dweight = [0] * len(trndata[0])

    for repeat in range(repeatation):
        for data in trndata:
            sigmoid = find_sigmoid(data, weight)
            y_minus_p = data[-1] - sigmoid
            # w_update = find_sigmoid(data, weight)
            dweight[0] = dweight[0] + y_minus_p
            for i in range(len(data) - 1):
                dweight[i+1] = dweight[i+1] + (y_minus_p * data[i])
            # dweight[1] = dweight[1] + (y_minus_p * data[0])
            # dweight[2] = dweight[2] + (y_minus_p * data[1])
            # dweight[3] = dweight[3] + (y_minus_p * data[2])
            # dweight[4] = dweight[4] + (y_minus_p * data[3])

        for i in range(len(weight)):
            weight[i] = weight[i] + (learning_rate * dweight[i])

    return weight


def prediction(tstdata, weight_v):
    predict = []
    for index in range(len(tstdata)):
        predict_sigmoid = find_sigmoid(tstdata[index], weight_v).flatten('F')
        predict.append(round(predict_sigmoid[0]))

    return predict


def accuracy(tstdata, predicts):
    right = 0
    for i in range(len(tstdata)):
        if predicts[i] == tstdata[i][-1]:
            right += 1

    return (right / len(tstdata)) * 100.0


# Main Call #

def avg_accuracy():
    traindata, testdata = create_dataset("data/data_banknote_authentication.txt", 0.67)
    fractions = [0.01, 0.02, 0.05, 0.1, 0.625, 1]
    learning_rate = 0.1
    repeat = 25

    avg_accuracy_dict = {}
    for frac in fractions:
        accuracy_val = 0
        for count in range(5):
            sampletraining, tmp = fraction_training(traindata, frac)
            weight_vector = weight_vector_update(sampletraining, learning_rate, repeat)
            predicts = prediction(testdata, weight_vector)
            accuracy_val += accuracy(testdata, predicts)

        avg_accuracy_dict[frac] = accuracy_val / 5
        # print("[T] {} -- {}".format(frac, avg_accuracy_dict.get(frac)))

    return avg_accuracy_dict


lr_accuracy = avg_accuracy()
keys = []
vals = []
for k in lr_accuracy.keys():
    keys.append(k)
    vals.append(lr_accuracy.get(k))

x_array = np.array(keys)
y_array = np.array(vals)
plt.ylabel("Accuracy %")
plt.xlabel("Training data fraction")
plt.title("[Logistic Regression]\nAccuracy(%) Vs. Training Data Size(in fraction)")
plt.plot(x_array, y_array, 'bo-.')
plt.show()
