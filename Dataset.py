from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
