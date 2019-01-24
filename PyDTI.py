#!/home/5/15D38037/.pyenv/versions/anaconda3-2.4.0/bin/python3.5
#
# Tomohiro Ban edited this script at January 9, 2018.
#
#==============================================================================

import os
import sys
import logging
import time
import getopt
import cv_eval
import ev_eval
from functions import *
from nrlmf import NRLMF
from nrlmfb import NRLMFb
from netlaprls import NetLapRLS
from blm import BLMNII
from wnngip import WNNGIP
#from kbmf import KBMF
from cmf import CMF
from new_pairs import novel_prediction_analysis


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "m:d:f:c:e:s:o:n:p:g:q:r:l:w", ["method=","dataset=","data-dir=","cvs=","external=","specify-arg=","method-opt=","predict-num=","scoring=","gpmi=","params=","output-dir=","log=","workdir="])
    except getopt.GetoptError:
        sys.exit()

#    data_dir = os.path.join(os.path.pardir, 'data')
#    output_dir = os.path.join(os.path.pardir, 'output')
    method = "nrlmf"
    dataset = "nr"
    data_dir = '.'
    output_dir = '.'
    cvs, sp_arg, model_settings, predict_num = 1, 1, [], 0
    external = 0
    scoring='auc'
    gpmi = None
    params = None
    workdir = "./"
    logfile = 'job.log'
    
    seeds = [7771, 8367, 22, 1812, 4659]
    # seeds = np.random.choice(10000, 5, replace=False)
    for opt, arg in opts:
        if opt == "--method":
            method = arg
        if opt == "--dataset":
            dataset = arg
        if opt == "--data-dir":
            data_dir = arg
        if opt == "--output-dir":
            output_dir = arg
        if opt == "--cvs":
            cvs = int(arg)
        if opt == "--external":
            external = int(arg)
        if opt == "--specify-arg":
            sp_arg = int(arg)
        if opt == "--method-opt":
            model_settings = [s.split('=') for s in str(arg).split()]
        if opt == "--predict-num":
            predict_num = int(arg)
        if opt == "--scoring":
            scoring=str(arg)
        if opt == "--gpmi":
            gpmi = dict()
            for s in str(arg).split():
                key, val = s.split('=')
                gpmi[key] = float(val)
        if opt == "--params":
            params = read_params(str(arg))
        if opt == "--log":
            logfile = str(arg)
        if opt == "--workdir":
            workdir = str(arg)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # set logger
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    filename = logfile
    fh = logging.FileHandler(workdir+"/"+filename)
    fh.name = filename
    logger.addHandler(fh)

    # default parameters for each methods
    if method == 'nrlmf':
        args = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, 'lambda_t': 0.125, 'alpha': 0.25, 'beta': 0.125, 'theta': 0.5, 'max_iter': 100}
    if method == 'nrlmfb':
        args = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, 'lambda_t': 0.125, 'alpha': 0.25, 'beta': 0.125, 'theta': 0.5, 'max_iter': 100}
    if method == 'netlaprls':
        args = {'gamma_d': 10, 'gamma_t': 10, 'beta_d': 1e-5, 'beta_t': 1e-5}
    if method == 'blmnii':
        args = {'alpha': 0.7, 'gamma': 1.0, 'sigma': 1.0, 'avg': False}
    if method == 'wnngip':
        args = {'T': 0.8, 'sigma': 1.0, 'alpha': 0.8}
    if method == 'kbmf':
        args = {'R': 50}
    if method == 'cmf':
        args = {'K': 50, 'lambda_l': 0.5, 'lambda_d': 0.125, 'lambda_t': 0.125, 'max_iter': 30}

    for key, val in model_settings:
        args[key] = float(val)

    intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'dataset'))
    drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'dataset'))

    if predict_num == 0:
        if cvs == 1:  # CV setting CVS1
            X, D, T, cv = intMat, drugMat, targetMat, 1
        if cvs == 2:  # CV setting CVS2
            X, D, T, cv = intMat, drugMat, targetMat, 0
        if cvs == 3:  # CV setting CVS3
            X, D, T, cv = intMat.T, targetMat, drugMat, 0
        cv_data = cross_validation(X, seeds, cv)
        ev_data = external_validation(X, seeds, cv)

    if sp_arg == 0 and predict_num == 0 and external == 0:
        if method == 'nrlmf':
            cv_eval.nrlmf_cv_eval(method,dataset,cv_data,X,D,T,cvs,args,logger,scoring=scoring,gpmi=gpmi,params=params)
        if method == 'nrlmfb':
            cv_eval.nrlmfb_cv_eval(method,dataset,cv_data,X,D,T,cvs,args,logger,scoring=scoring,gpmi=gpmi,params=params)
        if method == 'netlaprls':
            cv_eval.netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'blmnii':
            cv_eval.blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'wnngip':
            cv_eval.wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, args, logger)
        if method == 'kbmf':
            cv_eval.kbmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'cmf':
            cv_eval.cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)

    if sp_arg == 0 and predict_num == 0 and external == 1:
        if method == 'nrlmf':
            ev_eval.nrlmf_ev_eval(method,ev_data,X,D,T,logger,scoring=scoring,gpmi=gpmi,params=params)
        if method == 'nrlmfb':
            ev_eval.nrlmfb_ev_eval(method,ev_data,X,D,T,logger,scoring=scoring,gpmi=gpmi,params=params)

    if sp_arg == 1 or predict_num > 0:
        if method == 'nrlmf':
            model = NRLMF(cfix=args['c'], K1=args['K1'], K2=args['K2'], num_factors=args['r'], lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], alpha=args['alpha'], beta=args['beta'], theta=args['theta'], max_iter=args['max_iter'])
        if method == 'nrlmfb':
            model = NRLMFb(cfix=args['c'], K1=args['K1'], K2=args['K2'], num_factors=args['r'], lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], alpha=args['alpha'], beta=args['beta'], theta=args['theta'], max_iter=args['max_iter'], eta1=args['eta1'], eta2=args['eta2'])
        if method == 'netlaprls':
            model = NetLapRLS(gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], beta_d=args['beta_t'], beta_t=args['beta_t'])
        if method == 'blmnii':
            model = BLMNII(alpha=args['alpha'], gamma=args['gamma'], sigma=args['sigma'], avg=args['avg'])
        if method == 'wnngip':
            model = WNNGIP(T=args['T'], sigma=args['sigma'], alpha=args['alpha'])
        if method == 'kbmf':
            model = KBMF(num_factors=args['R'])
        if method == 'cmf':
            model = CMF(K=args['K'], lambda_l=args['lambda_l'], lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], max_iter=args['max_iter'])
        cmd = str(model)
        if predict_num == 0:

            tic = time.time()
            print("Dataset:"+dataset+" CVS:"+str(cvs)+"\n"+cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print("auc:%.6f, aupr:%.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.time()-tic))
#            write_metric_vector_to_file(auc_vec, os.path.join(output_dir, method+"_auc_cvs"+str(cvs)+"_"+dataset+".txt"))
#            write_metric_vector_to_file(aupr_vec, os.path.join(output_dir, method+"_aupr_cvs"+str(cvs)+"_"+dataset+".txt"))
            logger.info(cmd+', '+"auc:%.6f, aupr:%.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.time()-tic))

        elif predict_num > 0:
            print("Dataset:"+dataset+"\n"+cmd)
            seed = 7771 if method == 'cmf' else 22
            model.fix_model(intMat, intMat, drugMat, targetMat, seed)
            x, y = np.where(intMat == 0)
            scores = model.predict_scores(zip(x, y), 5)
            ii = np.argsort(scores)[::-1]
            predict_pairs = [(drug_names[x[i]], target_names[y[i]], scores[i]) for i in ii[:predict_num]]
            new_dti_file = os.path.join(output_dir, "_".join([method, dataset, "new_dti.txt"]))
            novel_prediction_analysis(predict_pairs, new_dti_file, os.path.join(data_dir, 'biodb'))

if __name__ == "__main__":
    main(sys.argv[1:])
