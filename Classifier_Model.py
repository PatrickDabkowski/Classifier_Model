import Dataset
import tensorflow
import tensorflow.keras as keras
import os
class Model:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def Bayes(self):
        from sklearn.naive_bayes import ComplementNB
        Bayesian = ComplementNB()
        Bayesian.fit(self.X_train, self.y_train)
        return Bayesian
    def CNN(self):
        crit = int(self.X_train.shape[0] * 0.1)
        self.X_train, self.X_valid = self.X_train[:-crit], self.X_train[-crit:]
        self.y_train, self.y_valid = self.y_train[:-crit], self.y_train[-crit:]
        early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        model = keras.models.Sequential([
            keras.layers.Input(shape=[28, 28, 1]),
            keras.layers.Dense(100, activation=tensorflow.keras.activations.relu),
            keras.layers.Dropout(0.1),
            keras.layers.Conv2D(10, 6, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.Conv2D(10, 5, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.Dropout(0.3),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='VALID'),
            keras.layers.Conv2D(20, 5, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.Conv2D(20, 5, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            keras.layers.Conv2D(40, 4, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.Conv2D(40, 4, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(80, 3, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.Conv2D(80, 3, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(160, 3, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.Conv2D(160, 2, activation=tensorflow.keras.activations.relu, padding='SAME'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            keras.layers.Dense(60, tensorflow.keras.activations.relu),
            keras.layers.Flatten(),
            keras.layers.Dense(30, tensorflow.keras.activations.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, tensorflow.keras.activations.softmax)
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        model.summary()

        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_valid, self.y_valid), epochs=20, callbacks=[early_stopping])
        model.save('Model.CNN')
        return model


if os.path.exists("Model.CNN"):
    model = keras.models.load_model('Model.CNN')
    print('Model Load')
else:
    X_train = Dataset.X_train
    y_train = Dataset.y_train

    class_model = Model(X_train, y_train)
    model = class_model.CNN()
    print('Model creat')
