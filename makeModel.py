import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime, time

# Using GPU
# tensorflow가 GPU를 사용하게 함
gpus = tf.config.experimental.list_physical_devices('GPU')

# 테스트에 사용할 loss function 목록
lossF = ['mean_absolute_error',
         'mean_squared_error',
         'mean_squared_logarithmic_error']

# Training data filename
# Speed, Steering, 가속도계(x, y, z), 자이로스코프(pitch, roll, yaw) 값이 저장되어있는 csv 파일

training_filename = ['20_46_33.txt.csv',
                    '21_01_03.txt.csv',
                    '22_18_48.txt.csv',
                    '00_21_56.txt.csv',
                    '23_26_36.txt.csv',
                    '23_45_20.txt.csv',
                    '23_58_10.txt.csv']

validating_filename = ['22_49_29.txt.csv',
                       '00_21_56.txt.csv']


def trainingmodel(lossFun):
    # make model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128), input_shape=(16, 8)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(1, 128)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dense(2, activation='relu'))

    model.summary()

    # set model training process
    model.compile(loss=lossFun, optimizer='adam', metrics=['acc'])

    # make training data
    dataX_array = []
    dataY_array = []
    dataX_list = []
    dataY_list = []
    for k in training_filename:
        data = np.loadtxt('data/' + k, delimiter=",")
        for i in range(len(data) - 15):
            dataX_list.append(data[i:i + 16, 0:9])
            dataY_list.append(data[i + 15, 0:2])

    dataX_array = np.array(dataX_list)
    dataY_array = np.array(dataY_list)

    # make validation data
    val_dataX_array = []
    val_dataY_array = []
    val_dataX_list = []
    val_dataY_list = []
    for k in validating_filename:
        data = np.loadtxt('data/' + k, delimiter=",")
        for i in range(len(data) - 15):
            val_dataX_list.append(data[i:i + 16, 0:9])
            val_dataY_list.append(data[i + 15, 0:2])

    val_dataX_array = np.array(val_dataX_list)
    val_dataY_array = np.array(val_dataY_list)

    hist = []

    # training
    hist = model.fit(dataX_array, dataY_array, epochs=50, validation_data=(val_dataX_array, val_dataY_array), batch_size=16, verbose=2)

    #get date
    d = datetime.date.today()
    now = d.year + '_' + d.month + '_' + d.day

    # save history of model
    # save visualiztion
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.savefig('result/' + now + '_' + lossFun + '.png')

    # convert keras model to tflite
    # Save the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open("models/" + now + '_' + lossFun + ".tflite", "wb").write(tflite_model)

for lf in lossF:
    trainingmodel(lf)










