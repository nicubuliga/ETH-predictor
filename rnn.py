import numpy as np
import csv


def read_data():
    rows = []
    with open(r"data\ETHUSDT_day.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)

    test_data = np.asarray(rows[:31], dtype=object)
    train_data = np.asarray(rows[31:], dtype=object)

    return test_data[::-1], train_data[::-1]


def filter_data(set):
    return np.asarray([[float(x[3])] for x in set], dtype=object)


def normalize(set):
    min_value = np.min(set)
    max_value = np.max(set)

    for row in set:
        row[0] = (row[0] - min_value) / (max_value - min_value)

    return set


def train_model(train_set):
    x_train = []
    y_train = []

    for i in range(60, len(train_set)):
        x_train.append(train_set[i-60:i])
        y_train.append(train_set[i])


if __name__ == "__main__":
    test_set, train_set = read_data()
    scaled_test_set = normalize(filter_data(test_set))
    scaled_train_set = normalize(filter_data(train_set))
    # for x in train_set:
    #     print(x[3])
    # print(scaled_test_set)
    train_model(scaled_train_set)
