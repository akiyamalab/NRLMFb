import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from functions import *
from time import time

class Optimizer(object):

    def __init__(self, scoring="auc"):
        assert scoring in ["auc", "aupr"]
        self.scoring = scoring

    def evaluate(self, model, dataset):
        intMat, drugMat, targetMat, cv_data = dataset
        aupr_vec, auc_vec = list(), list()
        for seed in cv_data.keys():
            for W, test_data, test_label in cv_data[seed]:
                model.fix_model(W, intMat, drugMat, targetMat, seed)
                aupr_val, auc_val = model.evaluation(test_data, test_label)
                aupr_vec.append(aupr_val)
                auc_vec.append(auc_val)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        score = auc_avg if self.scoring == "auc" else aupr_avg
        return score


class GPMI(Optimizer):
    
    def __init__(self, delta=1e-100, max_iter=10000, n_init=1, scoring="auc",
                 logger=None):
        super(GPMI, self).__init__(scoring=scoring)
        self.logger = logger
        self.delta = float(delta)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.alpha = np.log(2/delta)

    def optimize(self, model_class, params_grid, x_grid, dataset, seed=None):
        X, y = self.initialize(model_class, dataset, params_grid, x_grid, seed)
        gamma = 0
        best_param = None
        max_iter = len(x_grid) if len(x_grid) < self.max_iter else self.max_iter
        for i in range(max_iter):
            start = time() 
            gp = GaussianProcessRegressor()
            gp.fit(X, y)
            mean, sig = gp.predict(x_grid, return_std=True)
            phi = np.sqrt(self.alpha) * (np.sqrt(sig**2+gamma)-np.sqrt(gamma))
            i_next = np.argmax(mean + phi)
            params_next = params_grid[i_next]
            x_next = x_grid[i_next]
            gamma = gamma + sig[i_next]**2
            model = model_class(**params_next)
            y_next = self.evaluate(model, dataset)
            end = time()
            self.logger.info("i=%d, time=%.3f[sec]" %(i, end-start))
            if np.array_equal(x_next, X[-1]): best_param = params_next; break
            X = np.concatenate((X, [x_next]), axis=0)
            y = np.concatenate((y, [y_next]), axis=0)
        return best_param

    def initialize(self, model_class, dataset, params_grid, x_grid, seed):
        np.random.seed(seed=seed)
        i_init = np.random.permutation(range(len(x_grid)))[0:self.n_init]
        X = np.array([x_grid[i] for i in i_init])
        y = np.array(list())
        for i in i_init:
            model = model_class(**params_grid[i])
            score = self.evaluate(model, dataset)
            y = np.concatenate((y, [score]), axis=0)
        return X, y


class GridSearch(Optimizer):
    
    def __init__(self, scoring="auc"):
        super(GridSearch, self).__init__(scoring=scoring)

    def optimize(self, model_class, params_grid, x_grid, dataset):
        best_score = 0
        best_param = params_grid[0]
        for param in params_grid:
            model = model_class(**param)
            score = self.evaluate(model, dataset)
            if score > best_score:
                best_score = score
                best_param = param
        return best_param


