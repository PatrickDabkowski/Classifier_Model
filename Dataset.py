from tensorflow.keras.datasets import mnist
import tensorflow as tf

def inv(img):
    inv = 255 - img
    return inv

def get_dataset():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print('X_train.shape:', X_train.shape)
    print('X_test.shape:', X_test.shape)
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = get_dataset()
