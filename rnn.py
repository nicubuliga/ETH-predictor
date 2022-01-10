import numpy as np
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle

params = {
    "daily": {
        "nr": 60,
        "epochs": 100,
        "batch": 32
    },
    "hourly": {
        "nr": 504,
        "epochs": 20,
        "batch": 128
    },
    "minute": {
        "nr": 504,
        "epochs": 20,
        "batch": 128
    }
}


def read_data(nn_type, nr):
    rows = []
    with open("data\\ETHUSDT_{}.csv".format(nn_type), 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)

    test_data = np.asarray(rows[:nr])
    train_data = np.asarray(rows[nr:])

    return test_data[::-1], train_data[::-1]


def filter_data(nn_set):
    nr_days_for_sma = 9
    nr_days_for_12ema = 12
    nr_days_for_26ema = 26
    ema12 = 0
    ema26 = 0
    multiplier12 = float(2 / 13)
    multiplier26 = float(2 / 27)

    gain = [0]
    loss = [0]
    avg_gain_list = [0]
    avg_loss_list = [0]

    close_list = [float(nn_set[0][6])]
    table = np.array([[float(nn_set[0][3]), float(nn_set[0][6]), 0, 0, 0]])
    for i in range(1, len(nn_set)):
        # if i % 10000 == 0:
        #     print(i)
        if i < 14:
            close = float(nn_set[i][6])
            close_list.append(close)
            if len(close_list) == nr_days_for_sma:
                sma = float(sum(close_list) / nr_days_for_sma)
                sma = float(round(sma, 2))

                close_list.pop(0)
            else:
                sma = 0
            if i == 12:
                ema12 = close * multiplier12 + sma * (1 - multiplier12)
            if i > 12:
                ema12 = close * multiplier12 + ema12 * (1 - multiplier12)
            table = np.append(table, [[float(nn_set[i][3]), float(nn_set[i][6]), 0, sma, 0]], axis=0)

            last_close = float(nn_set[i - 1][6])
            diff = close - last_close
            if diff >= 0:
                gain.append(diff)
                loss.append(0)
                avg_gain_list.append(0)
                avg_loss_list.append(0)
            else:
                gain.append(0)
                loss.append(float(float(-1) * diff))
                avg_gain_list.append(0)
                avg_loss_list.append(0)
        else:
            close = float(nn_set[i][6])
            close_list.append(close)
            if len(close_list) == nr_days_for_sma:
                sma = float(sum(close_list) / nr_days_for_sma)
                sma = float(round(sma, 2))
                close_list.pop(0)
            else:
                sma = 0

            ema12 = close * multiplier12 + ema12 * (1 - multiplier12)
            if i == 26:
                ema26 = close * multiplier26 + sma * (1 - multiplier26)
            if i > 26:
                ema26 = close * multiplier26 + ema26 * (1 - multiplier26)
            macd = ema12 - ema26

            last_close = float(nn_set[i - 1][6])
            diff = close - last_close
            if diff >= 0:
                gain.append(diff)
                loss.append(0)
            else:
                gain.append(0)
                loss.append(float(float(-1) * diff))

            if i == 14:
                avg_gain = float(sum(gain) / 14)
                avg_loss = float(sum(loss) / 14)
            else:
                avg_gain = (avg_gain_list[i - 1] * 13 + gain[i]) / 14
                avg_loss = (avg_loss_list[i - 1] * 13 + loss[i]) / 14
            avg_gain_list.append(avg_gain)
            avg_loss_list.append(avg_loss)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
            else:
                rs = 0
            rsi = float(100 - (100 / (1 + rs)))

            table = np.append(table, [[float(nn_set[i][3]), float(nn_set[i][6]), rsi, sma, macd]], axis=0)
    return table
    # return np.asarray([[float(x[3]), float(x[6])] for x in nn_set])


def normalize(nn_set):
    for i in range(nn_set.shape[1]):
        # Find maximum and minimum values for normalization
        min_value = nn_set[0][i]
        max_value = nn_set[0][i]
        for row in nn_set:
            min_value = min(min_value, row[i])
            max_value = max(max_value, row[i])

        for row in nn_set:
            row[i] = (row[i] - min_value) / (max_value - min_value)

    return nn_set


def inverse_normalize(prices, train_set):
    min_value = np.min(train_set)
    max_value = np.max(train_set)

    output = []

    for price in prices:
        output.append(price[0] * (max_value - min_value) + min_value)

    return output


def train_model(train_set, nn_type):
    x_train = []
    y_train = []

    for i in range(params[nn_type]["nr"], len(train_set)):
        x_train.append(train_set[i - params[nn_type]["nr"]:i])
        y_train.append(train_set[i][0])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=80, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    #  Train model
    model.fit(x=x_train, y=y_train, epochs=params[nn_type]["epochs"], batch_size=params[nn_type]["batch"])

    model.save("trained_lstm_" + str(nn_type))


def test_model(test_set, train_set, original_train_set, nn_type):
    model = load_model("trained_lstm_" + str(nn_type))
    full_set = np.concatenate((train_set[len(train_set) - params[nn_type]["nr"]:], test_set))

    x_test = []
    for i in range(params[nn_type]["nr"], len(full_set)):
        x_test.append(full_set[i - params[nn_type]["nr"]:i])

    x_test = np.asarray(x_test)
    predicted_prices = model.predict(x_test)
    predicted_prices = inverse_normalize(predicted_prices, original_train_set)
    return predicted_prices


def only_open(arr):
    return np.asarray([[x[0]] for x in arr])


def show_results(output, target):
    plt.plot(only_open(target), color="red", label="Real ETH price")
    plt.title("ETH price prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.figure()
    plt.plot(output, color="blue", label="Predicted ETH price")
    plt.title("ETH price prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def save_train_set(train_set):
    with open("minute_train_set", "wb") as fd:
        pickle.dump(train_set, fd)


def get_train_set():
    with open("minute_train_set", "rb") as fd:
        data = pickle.load(fd)

    return data


def lstm_model(nn_type, nr):
    test_set, train_set = read_data(nn_type, nr)
    scaled_test_set = normalize(filter_data(test_set))
    scaled_train_set = normalize(filter_data(train_set))
    # train_model(scaled_train_set, nn_type)
    predicted_prices = test_model(scaled_test_set, scaled_train_set, filter_data(train_set), nn_type)
    show_results(predicted_prices, filter_data(test_set))


def get_params():
    print("Dataset types: daily, hourly, minute")
    return input("Enter dataset: "), int(input("Enter number: "))


if __name__ == "__main__":
    pass
