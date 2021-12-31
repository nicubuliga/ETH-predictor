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

    return test_data, train_data


def filter_data(set):
    return np.asarray([x[3] for x in set], dtype=object)


if __name__ == "__main__":
    test_set, train_set = read_data()
    test_set = filter_data(test_set)
    print(test_set)
