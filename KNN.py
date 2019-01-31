# Name: Denim Datta

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST


def read_data(_datadir):
    part = 0.1
    mndata = MNIST(_datadir)
    train_image_f, train_label_f = mndata.load_training()
    test_image_f, test_label_f = mndata.load_testing()

    _train_len = int(part * len(train_label_f))
    _test_len = int(part * len(test_label_f))
    _train_image = np.array(train_image_f[:_train_len])
    _train_label = np.array(train_label_f[:_train_len])
    _test_image = np.array(test_image_f[:_test_len])
    _test_label = np.array(test_label_f[:_test_len])

    return _train_image, _train_label, _test_image, _test_label


def all_distances(train_data, cur_data, len):
    dist_matrix = np.linalg.norm((cur_data - train_data), axis=1)
    sorted_index = np.argsort(dist_matrix)[:len]
    return sorted_index


def knn(nearest_list, trn_label, _k_):
    knn_list = []
    for index in range(_k_):
        knn_list.append(trn_label[nearest_list[index]])
    return knn_list


def predict(_knn_list):
    prediction = 0
    predict_dict = {}
    for label in _knn_list:
        if label in predict_dict:
            predict_dict[label] += 1
        else:
            predict_dict[label] = 1

    max_val = 0
    for key in predict_dict.keys():
        if predict_dict[key] > max_val:
            prediction = key
            max_val = predict_dict[key]

    return prediction


def accuracy(_actual_list, _prediction_list):
    correct = 0
    for index in range(len(_prediction_list)):
        if _actual_list[index] == _prediction_list[index]:
            correct += 1

    return round((correct / len(_prediction_list)) * 100.0, 3)


def error(_actual_list, _prediction_list):
    incorrect = 0
    for index in range(len(_prediction_list)):
        if _actual_list[index] != _prediction_list[index]:
            incorrect += 1

    return round((incorrect / len(_prediction_list)) * 100.0, 3)


def main():
    train_image, train_label, test_image, test_label = read_data('data')
    k_list = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    knn_error_test = []
    # knn_error_train = []
    # distance_list_train = []
    nearest_index_list_test = []
    number = len(test_image)
    # number = 10

    # for index in range(number):
    #     distance_list_train.append(all_distances(train_image, train_label, train_image[index]))

    for index in range(number):
        nearest_index_list_test.append(all_distances(train_image, test_image[index], max(k_list)))

    for K in k_list:
        # prediction_train = []
        prediction_test = []

        # for index in range(number):
        #     knn_list = knn(distance_list_train[index], K)
        #     prediction_train.append(predict(knn_list))

        for index in range(number):
            knn_list = knn(nearest_index_list_test[index], train_label, K)
            prediction_test.append(predict(knn_list))

        # knn_error_train.append(error(train_label, prediction_train))
        knn_error_test.append(error(test_label, prediction_test))

        # print(knn_error_train)
        print(knn_error_test)

    k_vals = np.array(k_list)
    # trn_error = np.array(knn_error_train)
    tst_error = np.array(knn_error_test)

    plt.ylim(0, int(math.ceil((max(tst_error) + 1) / 10.0)) * 10)
    plt.ylabel("Test Error")
    plt.xlabel("K Value")
    plt.title("Test Error Vs. K value")
    plt.plot(k_vals, tst_error, 'rv--')
    plt.show()


ts_time = time.time()
main()
te_time = time.time()
print("Time : {} Sec".format(te_time - ts_time))
