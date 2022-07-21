class Model():
    def __int__(self, X_train, y_train):
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
