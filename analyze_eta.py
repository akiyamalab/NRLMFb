#!/usr/bin/env python3
#
# Tomohiro Ban developed this script at April 18, 2018.
#
#==============================================================================
import os
import argparse
from time import *
from logging import *
import numpy as np
import pandas as pd
from nrlmfb import NRLMFb
from functions import *

def execute(param,key,cvs,dataset,cv_data,X,D,T):

    filename = '%s_nrlmfb_cvs%s_%s.log'%(key,cvs,dataset)
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    logger.addHandler(FileHandler(filename))
    
    hparams = getHparams(key,param)
    
    if key == 'eta': labels = ['eta1','eta2','auc','aupr']
    else: labels = [key,'auc','aupr']
    table = pd.DataFrame(columns=labels)
    
    for params in hparams:
        start  = time()
        model = NRLMFb(**params)
        aupr_vec, auc_vec = train(model,cv_data,X,D,T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        auc, aupr = round(auc_avg,4), round(aupr_avg,4)
        if key == 'eta': result = pd.DataFrame([[params['eta1'],params['eta2'],auc,aupr]],columns=labels)
        else: result = pd.DataFrame([[params[key],auc,aupr]],columns=labels)
        table = pd.concat([table,result],axis=0)
        end = time()
        logger.info(pd.concat([result,pd.DataFrame([[round(end-start,0)]],columns=['time'])],axis=1))

    print(table)
    table.to_csv('%s_nrlmfb_cvs%s_%s.txt'%(key,cvs,dataset),sep='\t',index=0)

    return None


def getData(cvs,dataset,data_dir='.'):

    Y, Kd, Kt = load_data_from_file(dataset,os.path.join(data_dir,'dataset'))

    cv = 1 if cvs == 1 else 0

    if cvs == 1:  X, D, T, cv = Y, Kd, Kt, 1
    elif cvs == 2:  X, D, T, cv = Y, Kd, Kt, 0
    elif cvs == 3:  X, D, T, cv = Y.T, Kt, Kd, 0
    else: print("Error at getData()"); quit(1)

    seeds = [7771, 8367, 22, 1812, 4659]
    cv_data = cross_validation(X,seeds,cv)

    return cv_data, X, D, T


def getHparams(key,param):

    parameter = dict()
    range_a = [2**i for i in np.arange(10)]
    range_b = [2**i for i in np.arange(10)]

    if param is not None:
        for line in open(param,'r'):
            items = line.strip().split()
            for item in items:
                k, v = item.split('=')
                if k == 'c': parameter['cfix'] = float(v)
                if k == 'r': parameter['num_factors'] = float(v)
                else: parameter[k] = float(v)
                
    if key == "eta":
        list_a = [x for x in range_a for y in range_b]
        list_b = [y for x in range_a for y in range_b]

    if key == "eta1":
        list_a = [x for x in range_a]
        list_b = np.ones(len(list_a)) * 3

    elif key == "eta2":
        list_b = [x for x in range_b]
        list_a = np.ones(len(list_b)) * 7

    values = zip(list_a,list_b)
    hparams = list()

    # Hyperparameter settings
    for a, b in values:
        hparams.append({'cfix':parameter['cfix'],
                        'K1':parameter['K1'],
                        'K2':parameter['K2'],
                        'num_factors':parameter['num_factors'],
                        'lambda_d':parameter['lambda_d'],
                        'lambda_t':parameter['lambda_t'],
                        'alpha':parameter['alpha'],
                        'beta':parameter['beta'],
                        'eta1':a,
                        'eta2':b,
                        'theta':parameter['theta'],
                        'max_iter':100})
    
    return hparams


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-k',action="store",dest="key",
                        help="A parameter that you want to analyze: eta1, eta2, eta")
    parser.add_argument('-c',action="store",dest="cvs",
                        help="Cross Validation Sinario: 1, 2, 3")
    parser.add_argument('-d',action="store",dest="dataset",
                        help="Dataset: nr, gpcr, ic, e")
    parser.add_argument('--data_dir',action="store",dest="data_dir",
                        help="Directory: .")
    parser.add_argument('-p',action="store",dest="param",
                        help="Parameter file: param_nrlmf_cvs1_nr.txt")
    args = parser.parse_args()

    cv_data, X, D, T = getData(int(args.cvs),args.dataset,data_dir=args.data_dir)
    execute(args.param,args.key,args.cvs,args.dataset,cv_data,X,D,T)


