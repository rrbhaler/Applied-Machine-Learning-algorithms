# Name: Denim Datta

import math
import random
import numpy as np
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
    trainlen = int(round(len(dataset) * ratio))
    traind = dataset[:trainlen]
    testd = dataset[trainlen:]

    return [traind, testd]


def fraction_training(trndata, fraction):
    tlen = int(round(len(trndata) * fraction))
    temp = list(trndata)
    sampletrndata = []
    while len(sampletrndata) < tlen:
        i = random.randrange(len(temp))
        sampletrndata.append(temp.pop(i))
    return sampletrndata


# calculate mean (mu)
def mu(no):
    return sum(no) / len(no)


# calculate sigma (variance = sigma^2)
def sigma(no):
    mean = mu(no)
    var = sum([pow(x - mean, 2) for x in no]) / (len(no))
    return math.sqrt(var)


def split_target(dataset):
    splitdata = {}
    for index in range(len(dataset)):
        data = dataset[index]
        if data[4] not in splitdata:
            splitdata[data[-1]] = []
        splitdata[data[-1]].append(data[:-1])

    return splitdata


# calculate mu and sigma for all feature value given a class target
def estimate_features(dataset):
    split = split_target(dataset)
    musigma_dict = {}
    for target in split.keys():
        allfeatures = split.get(target)
        musigma_dict[target] = [(mu(feature), sigma(feature)) for feature in zip(*allfeatures)]

    return musigma_dict


# N(µi,k, σi,k) form
def normal_distrib(x, mu_, sigma_):
    if sigma_ == 0:
        return x
    denom_part = 1 / (sigma_ * math.sqrt(2 * math.pi))
    exp_part = math.exp(-(math.pow(x - mu_, 2) / (2 * math.pow(sigma_, 2))))
    return denom_part * exp_part


# calculate probability of both target given the feature
def calculate_probability(estimate, input_param):
    probability = {}
    for target in estimate.keys():
        estvalue = estimate.get(target)
        probability[target] = 1
        for index in range(4):
            mu_, sigma_ = estvalue[index]
            no = input_param[index]
            probability[target] *= normal_distrib(no, mu_, sigma_)

    return probability


def class_prediction(estimate, input_param):
    probabilities = calculate_probability(estimate, input_param)
    target_final = ""
    probab_final = -5

    for target in probabilities.keys():
        probab = probabilities.get(target)
        if probab > probab_final:
            probab_final = probab
            target_final = target

    return target_final


def prediction(estimate, tstdata):
    predict = []
    for index in range(len(tstdata)):
        predict.append(class_prediction(estimate, tstdata[index]))

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

    avg_accuracy_dict = {}
    for frac in fractions:
        accuracy_val = 0
        for count in range(5):
            sampletraining = fraction_training(traindata, frac)
            estimation = estimate_features(sampletraining)
            predicts = prediction(estimation, testdata)
            accuracy_val += accuracy(testdata, predicts)

        # print("[TEST] {}  --  {}".format(frac, accuracy_val / 5))
        avg_accuracy_dict[frac] = accuracy_val / 5

    return avg_accuracy_dict


nb_accuracy = avg_accuracy()
keys = []
vals = []
for k in nb_accuracy.keys():
    keys.append(k)
    vals.append(nb_accuracy.get(k))

x_array = np.array(keys)
y_array = np.array(vals)
plt.ylabel("Accuracy %")
plt.xlabel("Training data fraction")
plt.title("[Naive Bayes]\nAccuracy(%) Vs. Training Data Size(in fraction)")
plt.plot(x_array, y_array, 'bo-.')
plt.show()
