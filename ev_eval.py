import numpy as np
from nrlmf import NRLMF
from nrlmfb import NRLMFb
from optimization import GPMI, GridSearch
from functions import *
from time import time


def nrlmf_ev_eval(method, ev_data, intMat, drugMat, targetMat, logger,
                  gpmi=None, scoring="auc", params=None):

    if gpmi is not None: optimizer = GPMI(**gpmi, scoring=scoring, logger=logger)
    else: optimizer = GridSearch(scoring=scoring)
    
    params_grid, x_grid = list(), list()
    for param in params:
        if param['lambda_d'] != param['lambda_t']: continue
        params_grid.append({'cfix':param['c'],'K1':param['K1'],'K2':param['K2'],
                            'num_factors':param['r'],'lambda_d':param['lambda_d'],
                            'lambda_t':param['lambda_t'],'alpha':param['alpha'],
                            'beta':param['beta'],'theta':param['theta'],
                            'max_iter':param['max_iter']})
        x_grid.append([param['c'],param['K1'],param['K2'],param['r'],param['lambda_d'],
                       param['lambda_t'],param['alpha'],param['beta'],param['theta'],
                       param['max_iter']])

    logger.info("External validation of NRLMF:")
    start = time()
    count = 1
    aupr_vec, auc_vec = list(), list()
    for seed in sorted(ev_data.keys()):
        for W, test_data, test_label, cv_data in ev_data[seed]:
            total = len(ev_data[seed]) * len(ev_data.keys())
            logger.info('')
            logger.info('Evaluate(%d/%d) seed=%d ...' %(count, total, seed))
            s = time()
            dataset = (intMat, drugMat, targetMat, cv_data)
            best_param = optimizer.optimize(NRLMF, params_grid, x_grid,
                                            dataset, seed=seed)
            model = NRLMF(**best_param)
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr_vec.append(aupr_val)
            auc_vec.append(auc_val)
            e = time()
            logger.info("auc:%.6f, aupr:%.6f, time:%.6f" %(auc_val, aupr_val, e-s))
            count += 1
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    
    end = time()
    cmd = "Result:\n"
    cmd += "auc:%.6f, aupr:%.6f, auc_conf:%.6f, aupr_conf:%.6f, time:%.6f\n"\
        %(auc_avg, aupr_avg, auc_conf, aupr_conf, end-start)
    logger.info('')
    logger.info(cmd)    


def nrlmfb_ev_eval(method, ev_data, intMat, drugMat, targetMat, logger,
                  scoring="auc", gpmi=None, params=None):
    
    if gpmi is not None: optimizer = GPMI(**gpmi, scoring=scoring, logger=logger)
    else: optimizer = GridSearch(scoring=scoring)
    
    params_grid, x_grid = list(), list()
    for param in params:
        if param['lambda_d'] != param['lambda_t']: continue
        params_grid.append({'cfix':param['c'],'K1':param['K1'],'K2':param['K2'],
                            'num_factors':param['r'],'lambda_d':param['lambda_d'],
                            'lambda_t':param['lambda_t'],'alpha':param['alpha'],
                            'beta':param['beta'],'theta':param['theta'],
                            'max_iter':param['max_iter'],'eta1':param['eta1'],
                            'eta2':param['eta2']})
        x_grid.append([param['c'],param['K1'],param['K2'],param['r'],param['lambda_d'],
                       param['lambda_t'],param['alpha'],param['beta'],param['theta'],
                       param['max_iter'],param['eta1'],param['eta2']])

    logger.info("External validation of NRLMFb:")
    start = time()
    count = 1
    aupr_vec, auc_vec = list(), list()
    for seed in sorted(ev_data.keys()):
        for W, test_data, test_label, cv_data in ev_data[seed]:
            total = len(ev_data[seed]) * len(ev_data.keys())
            logger.info('')
            logger.info('Evaluate(%d/%d) seed=%d ...' %(count, total, seed))
            s = time()
            dataset = (intMat, drugMat, targetMat, cv_data)
            best_param = optimizer.optimize(NRLMFb, params_grid, x_grid,
                                            dataset, seed=seed)
            model = NRLMFb(**best_param)
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr_vec.append(aupr_val)
            auc_vec.append(auc_val)
            e = time()
            logger.info("auc:%.6f, aupr:%.6f, time:%.6f" %(auc_val, aupr_val, e-s))
            count += 1
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    
    end = time()
    cmd = "Result:\n"
    cmd += "auc:%.6f, aupr:%.6f, auc_conf:%.6f, aupr_conf:%.6f, time:%.6f\n"\
        %(auc_avg, aupr_avg, auc_conf, aupr_conf, end-start)
    logger.info('')
    logger.info(cmd)    



