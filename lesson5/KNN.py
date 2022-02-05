import numpy as np

class KNearestNeighbors:
    def __init__(self, k):
        self.k = k

    # train
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def euclideanDistance(self, a, b):
        dis = np.sqrt(np.sum((a - b) ** 2))
        return dis

    def nearNeighbors(self, X_test):
        dists = []
        for x_train in self.X_train:
            dist = self.euclideanDistance(x_train, X_test)
            dists.append(dist)

        index_sorted = np.argsort(dists)
        gender_sorted = self.Y_train[index_sorted]
        return gender_sorted[0:self.k]

    def evaluate(self, X_test, Y_test):
        KNN_predicts = []
        for x in X_test:
            KNN_predicts.append(self.predict(x))
        correct = 0
        for i in range(len(X_test)):
            if KNN_predicts[i] == Y_test[i]:
                correct += 1
        return correct/len(Y_test)

    # test
    def predict(self, X_test):
        neighbors = self.nearNeighbors(X_test)
        Y_test = np.argmax(np.bincount(neighbors))
        return Y_test
        