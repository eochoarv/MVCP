#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:32:10 2024

@author: eochoa
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils.fixes import parse_version, sp_version

# This is line is to avoid incompatibility if older SciPy version.
# You should use `solver="highs"` with recent version of SciPy.
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

from sklearn.linear_model import QuantileRegressor

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.interpolate import interp1d
from collections import defaultdict

def process_data(rebecca_data_ir, rebecca_data_raman, IR_spectra, Raman_spectra):
    
    x_train_ir = rebecca_data_ir.iloc[:,:-2].to_numpy()
    x_train_ir = (x_train_ir - x_train_ir.mean(1)[:, None]) / x_train_ir.std(1)[:, None]
    
    y_train_ir = rebecca_data_ir.loc[:,'polymer']

    x_train_raman = rebecca_data_raman.iloc[:,:-2].to_numpy()
    x_train_raman = (x_train_raman - x_train_raman.mean(1)[:, None]) / x_train_raman.std(1)[:, None]
    
    y_train_raman = rebecca_data_raman.loc[:,'polymer']

    x_ir = IR_spectra.iloc[:,:-2].to_numpy()
    x_ir = (x_ir - x_ir.mean(1)[:, None]) / x_ir.std(1)[:, None]    

    x_raman = Raman_spectra.iloc[:,:-2].to_numpy()
    x_raman = (x_raman - x_raman.mean(1)[:, None]) / x_raman.std(1)[:, None]
    y_test = IR_spectra.loc[:,'polymer']
    
    #idx_test = np.sum(np.isnan(x_raman), axis=1) == 0
    #ids_test = IR_spectra.loc[idx_test,'id']
    
    #x_ir = x_ir[idx_test,:]
    #x_raman = x_raman[idx_test,:]
    
    y_raman = y_test#[idx_test]
    y_ir = y_test#[idx_test]
    
    
    poly_list = list(np.unique(np.concatenate([y_train_ir, y_train_raman, y_ir]))) 
    label_table = pd.DataFrame({'label':np.arange(len(poly_list)), 'polymer':poly_list})
    
    label_train_Raman = np.array(pd.DataFrame(y_train_raman).merge(label_table, left_on="polymer", right_on="polymer")['label'])
    label_train_IR = np.array(pd.DataFrame(y_train_ir).merge(label_table, left_on="polymer", right_on="polymer")['label'])
    
    label_ir = np.array(pd.DataFrame(y_ir).merge(label_table, how='left', left_on="polymer", right_on="polymer")['label'])
    
    label_raman = np.array(pd.DataFrame(y_raman).merge(label_table, how='left', left_on="polymer", right_on="polymer")['label'])
    
    return x_train_ir, x_train_raman, label_train_IR, label_train_Raman, x_ir, x_raman, label_ir, label_raman, label_table

# Pearson correlation coefficient 
def smx_pcc(X_train, X_test, labels_train, n_labels=None):
    
    n_ref = X_train.shape[0]
    n_test = X_test.shape[0]
    
    df_R = np.corrcoef(X_train,  X_test)[:n_ref, n_ref:]
    
    df_R = pd.DataFrame(df_R)
    df_R['polymer_ref'] = labels_train
    
    if n_labels is not None:
        labales_not_in = list(set(np.arange(n_labels)).difference(set(labels_train)))
        for l in labales_not_in:
            df_aux = pd.DataFrame(np.zeros((1, n_test)))
            df_aux['polymer_ref'] = l
            df_R = pd.concat([df_R, df_aux], axis=0).reset_index(drop=True)

    smx = (df_R.melt(id_vars=['polymer_ref']).
             groupby(['variable','polymer_ref']).
             max().
             reset_index().
             pivot(index='variable', columns='polymer_ref', values='value').
             to_numpy())
    
    return smx

# nearest neighbor
def smx_nn(X_train, X_test, labels_train, tau=10, n_labels=None):
    
    
    n_test = X_test.shape[0]
    D = pairwise_distances(X_train, X_test)
    df_D = pd.DataFrame(D)
    df_D['polymer_ref'] = labels_train
    
    if n_labels is not None:
        labales_not_in = list(set(np.arange(n_labels)).difference(set(labels_train)))
        for l in labales_not_in:
            df_aux = pd.DataFrame(10e16*np.ones((1, n_test)))
            df_aux['polymer_ref'] = l
            df_D = pd.concat([df_D, df_aux], axis=0).reset_index(drop=True)
            
    smx = (df_D.melt(id_vars=['polymer_ref']).
                 groupby(['variable','polymer_ref']).
                 min().
                 reset_index().
                 pivot(index='variable', columns='polymer_ref', values='value').
                 to_numpy())
    smx_1 = (smx / smx.min(1)[:,None])
    smx_2 = (smx / np.sort(smx, 1)[:,1][:,None])
    smx_1[smx_1 == 1] = smx_2[smx_1 == 1]
    smx = np.exp(-smx_1/tau)

    return smx

def smx_nn_all(X_train, X_test, labels_train, tau=10, n_labels=None):
    
    n_test = X_test.shape[0]
    D = pairwise_distances(X_train, X_test)
    df_D = pd.DataFrame(D)
    df_D['polymer_ref'] = labels_train
    
    df_D_min = df_D.groupby('polymer_ref').min()

    if n_labels is not None:
        labales_not_in = list(set(np.arange(n_labels)).difference(set(labels_train)))
        for l in labales_not_in:
            df_aux = pd.DataFrame(10e16*np.ones((1, n_test)))
            df_aux['polymer_ref'] = l
            df_D = pd.concat([df_D, df_aux], axis=0).reset_index(drop=True)
            
    D_list = []
    for i in range(D.shape[1]):
        D_list.append(D[:,i]/np.array([np.min(df_D_min.drop(j, axis=0).iloc[:,i]) for j in labels_train]))
    
    smx = np.exp(-np.array(D_list).T/tau)

    return smx

def project_u_mv(s, u):
    s_u = np.transpose(np.array(s),(1, 2, 0)) @ u
    return s_u, None

def gen_u(K, n_th=400, seed=0):
    np.random.seed(seed=seed)
    U_list = []
    for i in range(1, K):
        U = np.abs(np.random.multivariate_normal(np.zeros(i+1), np.eye(i+1), size=n_th))
        U = np.array([u/np.linalg.norm(u) for u in U])
        U_list.append(U)
    return U_list


# function to split data into calibration and teste sets

def split_strat(labels, cal_prop=0.5, seed=0):
    np.random.seed(seed=seed)
    labels_list = np.unique(labels)
    idx_strat = [np.where(labels == l)[0] for l in labels_list]
    ids_ref = np.concatenate([np.random.choice(x, size=int(x.shape[0]*cal_prop), replace=False) for x in idx_strat])
    idx = np.array([i in ids_ref for i in range(labels.shape[0])])
    
    return idx

def split_cal_test(smx, labels_test, cal_prop=0.5, seed=0):
    np.random.seed(seed=seed)
    labels_list = np.unique(labels_test)
    idx_strat = [np.where(labels_test == l)[0] for l in labels_list]
    ids_ref = np.concatenate([np.random.choice(x, size=int(x.shape[0]*cal_prop), replace=False) for x in idx_strat])
    idx = np.array([i in ids_ref for i in range(smx.shape[0])])
    cal_smx, val_smx = smx[idx,:], smx[~idx,:]
    cal_labels, val_labels = labels_test[idx], labels_test[~idx]
    
    return cal_smx, val_smx, cal_labels, val_labels, idx

def mvcp_split(score_list, labels, quant_prop, seed):

    K = len(score_list)
    quant_smx_list = []
    cal_smx_list = []
    for i in range(K):

        (quant_smx, 
         cal_smx, 
         quant_labels, 
         cal_labels, 
         idx_quant) = split_cal_test(score_list[i], 
                                     labels, 
                                     cal_prop=quant_prop, 
                                     seed=seed)
        
        quant_smx_list.append(quant_smx)
        cal_smx_list.append(cal_smx)

    n = cal_labels.shape[0]
    
    return (quant_smx_list, quant_labels, cal_smx_list, cal_labels)

def conform_prediction(cal_smx, val_smx, cal_labels, n, alpha, val_labels=None, cond=0):
    
    # 1: get conformal scores. n = calib_Y.shape[0]
    
    cal_scores = cal_smx[np.arange(n),cal_labels]
    # 2: get adjusted quantile
    #q_level = np.ceil((n+1)*(1-alpha))/n
    q_level = np.clip(np.ceil((n+1)*(1-alpha))/n, 0, 1)
    qhat = np.quantile(cal_scores, q_level, method='higher')
    # 3: form prediction sets
    prediction_sets = qhat >= val_smx 
    
    if val_labels is None: 
        return prediction_sets, qhat
    # Calculate empirical coverage
    empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    return (prediction_sets, empirical_coverage), qhat


def calibrate_beta_mv(alpha, 
                   score_list, 
                   cal_labels, 
                   n_th=400,
                   U=None,
                   max_itr=10):
    beta = alpha
    diff = 3
    change = .1
    M = len(score_list)
    n = cal_labels.shape[0]
    i = 0
    if U is None:
        U = np.abs(np.random.multivariate_normal(np.zeros(M), np.eye(M), size=n_th))
    else:
        n_th=U.shape[0]
        
    while abs(diff)>1 and i<max_itr:
        beta_prev = beta
        cov_list = []
        q_list = []
        for th in range(n_th):
            u = U[th,:] #/ np.linalg.norm(U[th,:])
            cal_s_u, _ = project_u_mv(score_list, u)
            cal_scores = cal_s_u[np.arange(n),cal_labels]
            q_level = np.clip(np.ceil((n+1)*(1-beta))/n, 0, 1)
            #q_level = min(q_level, 1)
            qhat = np.quantile(cal_scores, q_level, method='higher')
            cov_list.append(cal_scores <= qhat)
            q_list.append(qhat)
            
        diff = np.ceil((n+1)*(1-alpha)) - np.prod(cov_list,0).sum()
        if np.prod(cov_list,0).sum() < np.ceil((n+1)*(1-alpha)):
            beta = beta_prev - change / 2
        else:
            beta = beta_prev + change / 2
        change = abs(beta - beta_prev)
        i = i + 1
    return beta_prev, np.array(q_list), U

def mvcp_2(quant_scores_list, 
           quant_labels, 
           cal_scores_list, 
           cal_labels, 
           val_scores_list, 
           alpha, 
           val_labels=None, 
          n_th=400,
          U=None):
    
    beta, q_array, U = calibrate_beta_mv(alpha, quant_scores_list, quant_labels, n_th, U)
    n_th = U.shape[0]
    prediction_sets_list = []
    n = cal_labels.shape[0]
    t = np.max([project_u_mv(cal_scores_list, U[th,:])[0][np.arange(n),cal_labels]/q for th, q in enumerate(q_array)], 0)
    q_level = np.clip(np.ceil((n+1)*(1-alpha))/n, 0, 1)
    t_quant = np.quantile(t, q_level, method='higher')
    
    for j, th in enumerate(np.linspace(0,np.pi/2,n_th)):

        val_s_u, val_s_u_orth = project_u_mv(val_scores_list, U[j,:])
        
        prediction_sets = val_s_u <= q_array[j] * t_quant
        
        prediction_sets_list.append(prediction_sets)
        
    prediction_sets_ = np.prod(prediction_sets_list,0) == 1
    U_quant_env = np.array([U[j] * q_array[j] * t_quant for j in range(n_th)])
    
    if val_labels is None: 
        return prediction_sets_, U_quant_env
    
    empirical_coverage = prediction_sets_[np.arange(prediction_sets_.shape[0]),val_labels].mean()
    
    return prediction_sets_, empirical_coverage, U_quant_env


def envelope(quant_smx_list, 
            quant_labels, 
            cal_smx_list,
            cal_labels,
            alpha, 
            label_table, 
            plot=True):
    
    K = len(quant_smx_list)
    U_list = gen_u(K, n_th=400, seed=0)
    
    beta, q_array, U = calibrate_beta_mv(alpha, quant_smx_list, quant_labels, U=U_list[0])

    n = cal_smx_list[0].shape[0]
    t = np.max([project_u_mv(cal_smx_list, U[th,:])[0][np.arange(n),cal_labels]/q for th, q in enumerate(q_array)], 0)
    q_level = np.clip(np.ceil((n+1)*(1-alpha))/n, 0, 1)
    t_quant = np.quantile(t, q_level, method='higher')
    cal_smx_arr = np.array([cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels],
                            cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels]])
    

    idx_quant_reg = np.argsort(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels]/t)#[::-1]
    quant_reg_x = t_quant*np.sort(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels]/t)#[::-1]
    quant_reg_y = t_quant*(cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels]/t)[idx_quant_reg]
    
    env_res = [quant_reg_x, quant_reg_y]
    
    if plot:
        
        _ = plt.scatter(1-quant_smx_list[0][np.arange(quant_labels.shape[0]),quant_labels],
                    1-quant_smx_list[1][np.arange(quant_labels.shape[0]),quant_labels], 
                    color='gold',
                    alpha=.6,
                    label='calibration set (envelope shape)')
        
        _ = plt.scatter(1-cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels],
                    1-cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels], 
                    color='blue',
                    alpha=.6,
                    label='calibration set (envelope size)')
        
        _ = plt.plot(1-env_res[0],
                 1-env_res[1], 
                 label=str(int(100*(1-alpha)))+' envelope',
                 c='orange',
                 linewidth=3.0
             )    
        
        _ = plt.xlabel("IR HQI")
        _ = plt.ylabel("Raman HQI")
        _ = plt.legend()
        _ = plt.show()

    return env_res

