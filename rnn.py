import numpy as np
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model


def read_data():
    rows = []
    with open(r"data\ETHUSDT_day.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)

    test_data = np.asarray(rows[:31])
    train_data = np.asarray(rows[31:])

    return test_data[::-1], train_data[::-1]


def filter_data(nn_set):
    return np.asarray([[float(x[3])] for x in nn_set])


def normalize(nn_set):
    min_value = np.min(nn_set)
    max_value = np.max(nn_set)

    for row in nn_set:
        row[0] = (row[0] - min_value) / (max_value - min_value)

    return nn_set


def inverse_normalize(prices, train_set):
    min_value = np.min(train_set)
    max_value = np.max(train_set)

    output = []
    for price in prices:
        output.append(price[0] * (max_value - min_value) + min_value)

    return output


def train_model(train_set):
    x_train = []
    y_train = []

    for i in range(60, len(train_set)):
        x_train.append(train_set[i - 60:i])
        y_train.append(train_set[i][0])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    #  Train model
    model.fit(x=x_train, y=y_train, epochs=100, batch_size=32)

    model.save("trained_lstm")


def test_model(test_set, train_set, original_train_set):
    model = load_model("trained_lstm")
    full_set = np.concatenate((train_set[len(train_set) - 60:], test_set))

    x_test = []
    for i in range(60, len(full_set)):
        x_test.append(full_set[i - 60:i])

    x_test = np.asarray(x_test)
    predicted_prices = model.predict(x_test)
    predicted_prices = inverse_normalize(predicted_prices, original_train_set)
    return predicted_prices


def lstm_model():
    test_set, train_set = read_data()
    scaled_test_set = normalize(filter_data(test_set))
    scaled_train_set = normalize(filter_data(train_set))
    # train_model(scaled_train_set)
    predicted_prices = test_model(scaled_test_set, scaled_train_set, filter_data(train_set))
    for price in filter_data(test_set):
        print(price[0])


if __name__ == "__main__":
    lstm_model()
