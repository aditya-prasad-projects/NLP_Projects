from sklearn.neural_network import MLPRegressor
import sklearn
from scipy.stats import pearsonr
import sys

class Classifiers():

    def __init__(self, training_data, testing_data, classifier):
        self.training_data = training_data
        self.testing_data = testing_data
        self.classifier = classifier

    def build_model(self):
        for i in range(len(self.classifier)):
            if(self.classifier[i] == "mlp"):
                training_error, testing_error, testing_pearson, training_pearson= self._MLP_Classifier()
                print("Training mlp = ", training_error)
                print("Testing mlp = ", testing_error)
                print("testing pearson = ", testing_pearson)
                print("training_pearson = ", training_pearson)
        return  training_error, testing_error, testing_pearson, training_pearson


    def _MLP_Classifier(self):
        mlp = MLPRegressor(hidden_layer_sizes=(200, 10), random_state=42, activation='relu', max_iter=2000,
                            learning_rate_init=0.001, learning_rate='constant')
        X_train = self.training_data[:,:-1]
        Y_train = self.training_data[:,-1]
        mlp.fit(X_train, Y_train)
        y_pred_train = mlp.predict(self.training_data[:,:-1])
        y_pred_test = mlp.predict(self.testing_data[:, :-1])
        testing_error = sklearn.metrics.mean_squared_error(self.testing_data[:,-1], y_pred_test)
        training_error = sklearn.metrics.mean_squared_error(self.training_data[:,-1], y_pred_train)
        testing_pearson = pearsonr(self.testing_data[:,-1], y_pred_test)
        training_pearson = pearsonr(self.training_data[:,-1], y_pred_train)
        return training_error, testing_error, testing_pearson, training_pearson







