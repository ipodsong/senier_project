import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Using GPU
# tensorflow가 GPU를 사용하게 함
# 점진적으로 GPU의 메모리를 증가시키며 사용
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


# clear session
# tf.keras.backend.tensorflow_backend.K.clear_session()


# 테스트에 사용할 loss function 목록
lossF = ['mean_absolute_error',
         'mean_squared_error',
         'mean_squared_logarithmic_error']

# Training data filename
# Speed, Steering, 가속도계(x, y, z), 자이로스코프(pitch, roll, yaw) 값이 저장되어있는 csv 파일

filename = ['20_46_33.txt.csv',
            '21_01_03.txt.csv',
            '22_18_48.txt.csv',
            '22_49_29.txt.csv',
            '23_26_36.txt.csv',
            '23_45_20.txt.csv',
            '23_58_10.txt.csv']



def trainingmodel(lossFun):
    # make model
    #
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64), input_shape=(16, 8)))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(1, 64)))
    model.add(tf.keras.layers.LSTM(64, return_sequences=False))
    model.add(tf.keras.layers.Dense(2, activation='relu'))

    model.summary()

    # set model training process
    model.compile(loss=lossFun, optimizer='adam', metrics=['acc'])

    # make validation data
    val_dataX_list = []
    val_dataY_list = []
    val_data = np.loadtxt("00_21_56.txt.csv", delimiter=",")

    for i in range(len(val_data) - 15):
        val_dataX_list.append(val_data[i:i + 16, 0:9])
        val_dataY_list.append(val_data[i + 15, 0:2])

    val_dataX_array = np.array(val_dataX_list)
    val_dataY_array = np.array(val_dataY_list)
    hist = []

    # make training data
    dataX_array = []
    dataY_array = []
    dataX_list = []
    dataY_list = []
    for k in filename:
        data = np.loadtxt(k, delimiter=",")
        for i in range(len(data) - 15):
            dataX_list.append(data[i:i + 16, 0:9])
            dataY_list.append(data[i + 15, 0:2])

    dataX_array = np.array(dataX_list)
    dataY_array = np.array(dataY_list)

    print(dataX_array)

    # training
    '''
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, verbose=1)
    csv_logger = tf.keras.callbacks.CSVLogger(lossFun + '.csv', append=True, separator=',')
    hist = model.fit(dataX_array, dataY_array, epochs=1, validation_data=(val_dataX_array, val_dataY_array),
                     batch_size=16, verbose=2, callbacks=[early_stop, csv_logger])
                     '''
    hist = model.fit(dataX_array, dataY_array, epochs=50, validation_data=(val_dataX_array, val_dataY_array), batch_size=16, verbose=2)
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

    plt.savefig(lossFun + '.png')

    # save model
    # model.save(lossFun + ".h5")


    # convert keras model to tflite
    # Save the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    open(lossFun + ".tflite", "wb").write(tflite_model)



for lf in lossF:
    trainingmodel(lf)










