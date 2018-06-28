#!/usr/bin/env python3
#
# Tomohiro Ban developed this script at April 11, 2018.
#
#==============================================================================
import os
import argparse
import numpy as np
import pandas as pd
from nrlmf import NRLMF
from nrlmfb import NRLMFb
from functions import *

def execute(method,cvs,dataset,data_dir="."):

    params = getparams(method)
    if method == 'nrlmf': model = NRLMF(**params)
    if method == 'nrlmfb': model = NRLMFb(**params)
    
    if cvs == 1: result = leave_one_out1(model,dataset,data_dir=data_dir)
    if cvs == 2: result = leave_one_out2(model,dataset,data_dir=data_dir)
    if cvs == 3: result = leave_one_out3(model,dataset,data_dir=data_dir)

    result.to_csv('loo_%s_cvs%s_%s.txt'%(method,cvs,dataset),sep='\t',index=0)
    
    return None


def leave_one_out1(model,dataset,data_dir='.',seed=0):
    
    intMat, Kd, Kt = load_data_from_file(dataset,os.path.join(data_dir,'dataset'))
    drugs, targets = get_drugs_targets_names(dataset,os.path.join(data_dir,'dataset'))
    Y = pd.DataFrame(intMat,index=drugs,columns=targets).astype(int)
    
    nd = len(Y.index)
    nt = len(Y.columns)
    
    pairs = [[d,t] for d in range(nd) for t in range(nt)]
    actives = Y.get_values().ravel()
    W = np.ones(intMat.shape)
    model.fix_model(W,intMat,Kd,Kt,seed)
    scores = model.predict_scores(pairs,1).reshape((nd,nt))
    scmat = pd.DataFrame(scores,index=drugs,columns=targets)

    labels = ['drug','target','active','score','gamma']
    table = pd.DataFrame(columns=labels)

    num = 0
    for d in Y.index:
        print(d,np.where(Y.index==d)[0][0],'/',nd)
        for t in Y.columns:
            if Y.ix[d,t] == 1:
                W_ = pd.DataFrame(np.ones(intMat.shape),index=drugs,columns=targets)
                W_.ix[d,t] = 0
                W = W_.get_values()
                model.fix_model(W,intMat,Kd,Kt,seed)
                d_ = np.where(Y.index==d)[0][0]
                t_ = np.where(Y.columns==t)[0][0]
                a = int(1)
                s = round(model.predict_scores([[d_,t_]],1)[0],4)
                g = int(gamma(W_*Y,d,t))
            else:
                a = int(0)
                s = round(scmat.ix[d,t],4)
                g = int(gamma(Y,d,t))
            result = pd.DataFrame([[d,t,a,s,g]],index=[num],columns=labels)
            table = pd.concat([table,result],axis=0)
            num += 1

    return table


def leave_one_out2(method,dataset,data_dir=".",seed=0):

    return None


def leave_one_out3(method,dataset,data_dir=".",seed=0):

    return None


def getparams(method):

    params = dict()

    if method == 'nrlmfb':
        params['cfix'] = 5
        params['K1'] = 5
        params['K2'] = 5
        params['num_factors'] = 100
        params['lambda_d'] = 1.0
        params['lambda_t'] = 1.0
        params['alpha'] = 1.0
        params['beta'] = 0.5
        params['eta1'] = 7
        params['eta2'] = 3
        params['theta'] = 0.125
        params['max_iter'] = 100

    if method == 'nrlmf':
        params['cfix'] = 5
        params['K1'] = 5
        params['K2'] = 5
        params['num_factors'] = 100
        params['lambda_d'] = 1.0
        params['lambda_t'] = 1.0
        params['alpha'] = 1.0
        params['beta'] = 0.5
        params['theta'] = 0.125
        params['max_iter'] = 100        

    return params


def gamma(Y,d,t):

    Yd = np.sum(Y,axis=1)
    Yt = np.sum(Y,axis=0)
    gamma = Yd[d]+Yt[t]-2 if Y.ix[d,t]==1 else Yd[d]+Yt[t]

    return gamma


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',action='store',dest='method',help='Method: nrlmf, nrlmfb')
    parser.add_argument('-c',action='store',dest='cvs',help='Cross Validation Sinario: 1, 2, 3')
    parser.add_argument('-d',action='store',dest='dataset',help='Dataset: nr, gpcr, ic, e')
    parser.add_argument('--data_dir',action='store',dest='data_dir',help='.')
    arg = parser.parse_args()
    
    execute(arg.method,int(arg.cvs),arg.dataset,data_dir=arg.data_dir)

