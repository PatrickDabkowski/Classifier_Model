class Model():
    def Bayes(self, X_train, y_train):
        from sklearn.naive_bayes import ComplementNB
        Bayesian = ComplementNB()
        Bayesian.fit(X_train, y_train)
        return Bayesian