#  _*_ coding: utf-8 _*_
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
import time
import math
from math import sqrt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
Timestep = 3


def create_dataset(dataset, lock_back=1):
    Timestep = 3
    dataX, dataY = [], []
    for i in range(len(dataset)-lock_back-1):
        a = dataset[i:(lock_back-Timestep), :]
        dataX.append(a)
        dataY.append(dataset[i+Timestep:, :])
        print()
    return np.array(dataX), np.array(dataY)


BATCH_START = 128
INPUT_SIZE = 6
OUTPUT_SIZE = 6
# PRED_SIZE = 72  # 预测输出3天序列数据
CELL_SIZE = 128
EPOCH = 6


def get_batch(train_x, train_y, TIME_STEPS):
    data_len = len(train_x) - TIME_STEPS
    seq = []
    res = []
    for i in range(data_len):
        seq.append(train_x[i:i + TIME_STEPS])
        res.append(train_y[i:i + TIME_STEPS])  # 取后5组数据
    seq, res = np.array(seq), np.array(res)
    return seq, res


def f_data(train_1, train_2):
    TIME_STEPS = 24*6
    PRED_SIZE = 24*3  # 前三天数据预测后三天数据

    if train_1 == True:

        # load_dataset
        # 时间序列做为标签
        # 第一轮训练
        D1HP = pd.read_excel('/media/mgege007/Compitition/Math/data/data_2_knn_AQI.xlsx',
                             sheet_name='监测点C逐小时污染物浓度与气象一次预报数据', index_col="time", parse_dates=True)

        D1HP.set_axis(['temperature_2m', 'temperature_surface', 'specific_humidity', 'humidity', 'wind_speed_10m', 'wind_direction_10m', 'rainfall', 'cloudiness', 'height',
                       'pressure', 'sensible_heat', 'latent_heat', 'longwave', 'shortwave', 'solar_radiation', 'so2', 'no2', 'pm10', 'pm2.5', 'o3', 'co', 'aqi'], axis='columns', inplace=True)

        D1HP = D1HP[['so2', 'no2', 'pm10', 'pm2.5', 'o3', 'co', 'aqi']]
        values = D1HP.values
        values = values.astype('float32')

        # 对数据进行归一化处理, valeus.shape=(, 8),inversed_transform时也需要8列
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(values)
        # 划分训练集测试集
        # len = data.shape[0]
        # p = 0.2
        # train_size = int(len*(1-p))
        # train, test = data[0:train_size, :], data[train_size:, :]
        train = data  # 全用来训练模型
        trainX, trainY = get_batch(
            train[:-PRED_SIZE], train[PRED_SIZE:][:, 0:OUTPUT_SIZE+1], TIME_STEPS)  # 0:6
        # testX, testY = trainX[-1000:, :, :], trainY[-1000:, :, :]

        testX, testY = trainX[2000:5000, :, :], trainY[2000:5000, :, :]
        # 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
        model = Sequential()

        model.add(LSTM(units=128,  activation='relu', return_sequences=True,
                       input_shape=(TIME_STEPS, INPUT_SIZE))
                  )
        model.add(Dropout(0.2))
        model.add(LSTM(units=64,
                       activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32,
                       activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        #全连接，输出， add output layer
        model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

        model.summary()
        start = time.time()
        model.compile(metrics=['accuracy'], loss='mse', optimizer='adam')
        checkpoint = tf.keras.callbacks.ModelCheckpoint('./weight/C_weight_f.tf', save_format='tf',
                                                        verbose=0, save_best_only=True, save_weights_only=True)
        k = trainX.shape[0] % BATCH_START
        trainX, trainY = trainX[k:], trainY[k:]
        # model.load_weights('./weight/C_weight_f.tf')
        history = model.fit(trainX, trainY, batch_size=BATCH_START, epochs=2,
                            validation_split=0.1,
                            callbacks=[checkpoint],
                            verbose=1)
        plt.figure(2)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        print()
        y_out = model.predict(testX)
        #预测数据逆缩放 invert scaling for forecast
        y_p = []
        y_t = []
        for i in range(y_out.shape[0]):
            y_pre = scaler.inverse_transform(y_out[i, :, :])
            y_pmp = np.mean(y_pre, axis=0)
            # y_pmp = y_pre

            y_p.append(y_pmp)
            #真实数据逆缩放 invert scaling for actual
            y_true = scaler.inverse_transform(testY[i, :, :])
            y_tmp = np.mean(y_true, axis=0)
            # y_tmp = y_true
            y_t.append(y_tmp)

        #画出真实数据和预测数据
        y_pre = np.array(y_p).reshape(-1, OUTPUT_SIZE)
        y_true = np.array(y_t).reshape(-1, OUTPUT_SIZE)
        plt.figure(2)
        for i, index in enumerate(D1HP.columns):
            ax1 = plt.subplot(7, 1, i+1)
            plt.sca(ax1)
            plt.plot(y_pre[:, i], label="Pre-"+str(index))
            plt.plot(y_true[:, i], label="True-"+str(index))
            plt.legend()
        plt.show()
        RMSE = []
        MAPE = []
        MAE = []

        def MAPE_T(true, pred):
            diff = np.abs(np.array(true) - np.array(pred))
            return np.mean(diff / true)
        for i in range(OUTPUT_SIZE):
            RMSE_tmp = np.sqrt(mean_squared_error(y_true[:, i], y_pre[:, i]))
            mape = MAPE_T(y_true[:, i], y_pre[:, i])
            mae = mean_absolute_error(y_true[:, i], y_pre[:, i])
            RMSE.append(RMSE_tmp)
            MAPE.append(mape)
            MAE.append(mae)
        print(RMSE)
        print(MAPE)
        print(MAE)
        pre_result = data[-TIME_STEPS:, :].reshape(1, TIME_STEPS, -1)
        result = model.predict(pre_result)
        result = scaler.inverse_transform(result[0, :, :])
        df_result = pd.DataFrame(result)
        df_result.to_csv("3_CHP.csv", index=False)
        print()
    if train_2 == True:
        # 开始第二轮训练
        # load_dataset
        D1HA = pd.read_excel('/media/mgege007/Compitition/Math/data/data_2_knn_AQI.xlsx',
                             sheet_name='监测点C逐小时污染物浓度与气象实测数据', index_col="time", parse_dates=True)
        D1HA.set_axis(['so2', 'no2', 'pm10', 'pm2.5', 'o3', 'co', 'temperature',
                       'pressure', 'wind', 'direction', 'aqi'], axis='columns', inplace=True)

        D1HA = D1HA[['so2', 'no2', 'pm10', 'pm2.5', 'o3', 'co', 'aqi']]
        values = D1HA.values
        values = values.astype('float32')

        # 对数据进行归一化处理, valeus.shape=(, 8),inversed_transform时也需要8列
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(values)
        # len = data.shape[0]
        # p = 0.2
        # train_size = int(len*(1-p))
        # test_size = len - train_size
        # train, test = data[0:train_size, :], data[train_size:, :]
        train = data
        trainX, trainY = get_batch(
            train[:-PRED_SIZE], train[PRED_SIZE:][:, 0:OUTPUT_SIZE+1], TIME_STEPS)  # 0:9
        # testX, testY = get_batch(
        #     test[:-PRED_SIZE], test[PRED_SIZE:][:, 0:OUTPUT_SIZE+1], TIME_STEPS)  # 0:9
        testX, testY = trainX[2000:5000, :, :], trainY[2000:5000, :, :]
        # 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
        model = Sequential()

        model.add(LSTM(units=128,  activation='relu', return_sequences=True,
                       input_shape=(TIME_STEPS, INPUT_SIZE))
                  )
        model.add(Dropout(0.2))
        model.add(LSTM(units=64,
                       activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32,
                       activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        #全连接，输出， add output layer
        model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

        model.summary()
        start = time.time()
        model.compile(metrics=['accuracy'], loss='mse', optimizer='adam')
        checkpoint = tf.keras.callbacks.ModelCheckpoint('./weight/C_weight_s.tf', save_format='tf',
                                                        verbose=0, save_best_only=True, save_weights_only=True)
        k = trainX.shape[0] % BATCH_START
        trainX, trainY = trainX[k:], trainY[k:]
        # model.load_weights('./weight/C_weight_s.tf')
        history = model.fit(trainX, trainY, batch_size=BATCH_START, epochs=4,
                            validation_split=0.1,
                            callbacks=[checkpoint],
                            verbose=1)
        plt.figure(1)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        y_out = model.predict(testX)
        #预测数据逆缩放 invert scaling for forecast
        y_p = []
        y_t = []
        for i in range(y_out.shape[0]):
            y_pre = scaler.inverse_transform(y_out[i, :, :])
            y_pmp = np.mean(y_pre, axis=0)
            # y_pmp = y_pre

            y_p.append(y_pmp)
            #真实数据逆缩放 invert scaling for actual
            y_true = scaler.inverse_transform(testY[i, :, :])
            y_tmp = np.mean(y_true, axis=0)
            # y_tmp = y_true
            y_t.append(y_tmp)

        #画出真实数据和预测数据
        y_pre = np.array(y_p).reshape(-1, OUTPUT_SIZE)
        y_true = np.array(y_t).reshape(-1, OUTPUT_SIZE)
        plt.figure(2)
        for i, index in enumerate(D1HA.columns):

            ax1 = plt.subplot(7, 1, i+1)
            plt.sca(ax1)
            plt.plot(y_pre[:, i], label="Pre-"+str(index))
            plt.plot(y_true[:, i], label="True-"+str(index))
            plt.legend()
        plt.show()
        RMSE = []
        MAPE = []
        MAE = []

        def MAPE_T(true, pred):
            diff = np.abs(np.array(true) - np.array(pred))
            return np.mean(diff / true)
        for i in range(OUTPUT_SIZE):
            RMSE_tmp = np.sqrt(mean_squared_error(y_true[:, i], y_pre[:, i]))
            mape = MAPE_T(y_true[:, i], y_pre[:, i])
            mae = mean_absolute_error(y_true[:, i], y_pre[:, i])
            RMSE.append(RMSE_tmp)
            MAPE.append(mape)
            MAE.append(mae)
        print(RMSE)
        print(MAPE)
        print(MAE)
        pre_result = data[-TIME_STEPS:, :].reshape(1, TIME_STEPS, -1)
        result = model.predict(pre_result)
        result = scaler.inverse_transform(result[0, :, :])
        df_result = pd.DataFrame(result)
        df_result.to_csv("3_CHA.csv", index=False)
        print()


def pre_data():
    # load_dataset
    # # 时间序列做为标签
    TIME_STEPS = 3*2  # 时间步长
    PRED_SIZE = 3   # 前三天数据预测后三天数据

    D1DA = pd.read_excel('/media/mgege007/Compitition/Math/data/data_2_knn_AQI.xlsx',
                         sheet_name='监测点C逐日污染物浓度实测数据', index_col="time", parse_dates=True)
    D1DA.set_axis(['so2', 'no2', 'pm10', 'pm2.5', 'o3', 'co',
                   'aqi'], axis='columns', inplace=True)

    D1DA = D1DA[['so2', 'no2', 'pm10', 'pm2.5', 'o3', 'co', 'aqi']]

    values = D1DA.values
    values = values.astype('float32')

    # 对数据进行归一化处理, valeus.shape=(, 8),inversed_transform时也需要8列
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(values)
    # 不划分测试集
    # len = data.shape[0]
    # p = 0.2
    # train_size = int(len*(1-p))
    # test_size = len - train_size
    # train, test = data[0:train_size, :], data[train_size:, :]
    train = data
    trainX, trainY = get_batch(
        train[:-PRED_SIZE], train[PRED_SIZE:][:, 0:OUTPUT_SIZE+1], TIME_STEPS)  # 0:9
    # testX, testY = get_batch(
    #     test[:-PRED_SIZE], test[PRED_SIZE:][:, 0:OUTPUT_SIZE+1], TIME_STEPS)  # 0:9
    testX, testY = trainX[:, :, :], trainY[:, :, :]
    # 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
    model = Sequential()

    model.add(LSTM(units=128,  activation='relu', return_sequences=True,
                   input_shape=(TIME_STEPS, INPUT_SIZE))
              )
    model.add(Dropout(0.2))
    model.add(LSTM(units=64,
                   activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32,
                   activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    #全连接，输出， add output layer
    model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

    model.summary()
    start = time.time()
    model.compile(metrics=['accuracy'], loss='mse', optimizer='adam')
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./weight/C_weight_t.tf', save_format='tf',
                                                    verbose=0, save_best_only=True, save_weights_only=True)
    k = trainX.shape[0] % BATCH_START
    trainX, trainY = trainX[k:], trainY[k:]
    # model.load_weights('./weight/C_weight_t.tf')
    history = model.fit(trainX, trainY, batch_size=BATCH_START, epochs=EPOCH,
                        validation_split=0.1,
                        callbacks=[checkpoint],
                        verbose=1)
    plt.figure(1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    y_out = model.predict(testX)
    #预测数据逆缩放 invert scaling for forecast
    y_p = []
    y_t = []
    for i in range(y_out.shape[0]):
        y_pre = scaler.inverse_transform(y_out[i, :, :])
        y_pmp = np.mean(y_pre, axis=0)
        # y_pmp = y_pre

        y_p.append(y_pmp)
        #真实数据逆缩放 invert scaling for actual
        y_true = scaler.inverse_transform(testY[i, :, :])
        y_tmp = np.mean(y_true, axis=0)
        # y_tmp = y_true
        y_t.append(y_tmp)

    #画出真实数据和预测数据
    y_pre = np.array(y_p).reshape(-1, OUTPUT_SIZE)
    y_true = np.array(y_t).reshape(-1, OUTPUT_SIZE)
    plt.figure(2)
    for i, index in enumerate(D1DA.columns):
        # if i == 6:
        #     break
        ax1 = plt.subplot(7, 1, i+1)
        plt.sca(ax1)
        plt.plot(y_pre[:, i], label="Pre-"+str(index))
        plt.plot(y_true[:, i], label="True-"+str(index))
        plt.legend()
    plt.show()
    RMSE = []
    MAPE = []
    MAE = []

    def MAPE_T(true, pred):
        diff = np.abs(np.array(true) - np.array(pred))
        return np.mean(diff / true)
    for i in range(OUTPUT_SIZE):
        RMSE_tmp = np.sqrt(mean_squared_error(y_true[:, i], y_pre[:, i]))
        mape = MAPE_T(y_true[:, i], y_pre[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pre[:, i])
        RMSE.append(RMSE_tmp)
        MAPE.append(mape)
        MAE.append(mae)
    print(RMSE)
    print(MAPE)
    print(MAE)
    pre_result = data[-TIME_STEPS:, :].reshape(1, TIME_STEPS, -1)
    result = model.predict(pre_result)
    result = scaler.inverse_transform(result[0, :, :])
    df_result = pd.DataFrame(result)
    df_result.to_csv("3_CDA.csv", index=False)
    print()


if __name__ == '__main__':
    # f_data(False, True)
    pre_data()
