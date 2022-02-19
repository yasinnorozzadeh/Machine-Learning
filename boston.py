import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data["MEDV"] = boston.target
data = data.loc[data["MEDV"] < 100]

X = data[['ZN',"AGE"]].values
Y = data["MEDV"].values
Y = Y.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

class Perceptron:
    def __init__(self, epochs=4, learning_rate=0.0001):
        # Hyper Parameters
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.W = np.random.rand(1,2)
    
    def fit(self, X, Y):
        x_x = np.arange(X[:, 0].min(),X[:, 0].max())
        y_y = np.arange(X[:, 1].min(),X[:, 1].max())
        x_x, y_y = np.meshgrid(x_x, y_y)
        fig = plt.figure(figsize=(12, 6))
        Errors = []
        
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                x = X[i].reshape(-1, 1)
                y_pred = np.matmul(self.W, x) 
                e = Y[i] - y_pred
                
                # Update Weights
                x = x.reshape(1, 2)
                self.W += self.learning_rate * e * x

                Z = x_x * self.W[0, 0]  + y_y * self.W[0, 1]
                fig.clear()
                ax1 = fig.add_subplot(1,2,1, projection="3d")
                
                ax1.scatter(X[:, 0], X[:, 1], Y, c='cyan')
                ax1.plot_wireframe(x_x, y_y, Z, color='red', alpha=0.5)
                ax1.plot_surface(x_x, y_y, Z, alpha=0.5)
                ax1.set_xlabel("ZN")
                ax1.set_ylabel("AGE")
                ax1.set_zlabel("MEDV")

                W = self.W.reshape(2, 1)
                Y_pred = np.matmul(X, W)
                Error = np.mean(np.abs(Y - Y_pred))
                Errors.append(Error)
                    
                ax2 = fig.add_subplot(1,2,2)
                ax2.clear()
                ax2.plot(Errors,linestyle = '--',marker = '*',c='orange')
                
                
                plt.pause(0.01)
            plt.show()
        
    def predict(self, X_test):
        W = self.W.T
        Y_pred = np.matmul(X_test, W)
        return Y_pred
    
    def evaluate(self, X_test, Y_true):
        Y_pred = self.predict(X_test)
        MAE = np.mean(np.abs(Y_true - Y_pred))
        return MAE
                
perceptron = Perceptron(epochs=3, learning_rate=0.0001)
perceptron.fit(X_train, Y_train)
print("predict:\n", perceptron.predict(X_test))
print("evaluate:", perceptron.evaluate(X_test, Y_test))