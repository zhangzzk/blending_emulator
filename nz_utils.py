import numpy as np
import pandas as pd

import xgboost as xgb

import os
from joblib import Parallel, delayed

import utils, data_utils

# from scipy.stats import poisson
# from scipy.stats import bootstrap


# maximum separation for two emulators
R_MAX = 7/3600
R_MAX_CLA = 3/3600

# from where to use the detection probability, below which p=0
DETECTION_THRESH = 0.

CLASSIFICATION_NAMES_SEC = ['Re_input_s', 'r_input_s', 'sersic_n_input_s', 'axis_ratio_input_s', 'distance']


def response_across_z(z_bins, target_values, features, weights):

    R = np.zeros((len(z_bins)-1,len(z_bins)-1))
    R_err = np.zeros((len(z_bins)-1,len(z_bins)-1))
    for i in range(len(z_bins)-1):
        for j in range(len(z_bins)-1):
            # print(i,j)
            bin_idx = np.where((features['redshift_input_p']>z_bins[i])&(features['redshift_input_p']<z_bins[i+1])\
                                &(features['redshift_input_s']>z_bins[j])&(features['redshift_input_s']<z_bins[j+1]))[0]
            R_bin = target_values[bin_idx]
            w_bin = weights[bin_idx]

            a = features[['input_index','case']].iloc[bin_idx].values
            a = a[:,0] + a[:,1]*1j
            _, idx, inv = np.unique(a, return_index=True, return_inverse=True)

            # print(idx.shape)
            R_sum = np.bincount(inv, weights=R_bin) # sum over neighbours for each primary galaxy
            R[i,j], R_err[i,j] = data_utils.std_of_the_weighted_mean(R_sum, w_bin[idx], axis=0)
            
    return R, R_err


def make_reg_features(icat1, icat2, r_min=0, r_max=99, k=30):
    
    # neighbor finder    
    pos = ['RA','DEC']
    pri_idx, sec_idx, separation = utils.kdt_neighbor_finder(icat1[pos].to_numpy(), 
                                                            icat2[pos].to_numpy(), 
                                                            r_min=r_min, r_max=r_max, k=k)
    
    # concatenate secondary and primary catalog for regression predicter
    sec_cat = icat2.iloc[sec_idx].reset_index(drop=True)
    sec_cat = sec_cat.rename(columns={f:f+'_input_s' for f in icat2.columns})
    
    pri_cat = icat1.iloc[pri_idx].reset_index(drop=True)
    pri_cat = pri_cat.rename(columns={f:f+'_input_p' for f in icat1.columns})

    response_features = pd.concat((pri_cat,sec_cat),axis=1)
    response_features['distance'] = separation
    
    return response_features
    

def icat2reg(icat_i_pri,icat_i_sec,model,conditions):

    # match neighbors, make pair catalog and select sources
    reg_features_i = make_reg_features(icat_i_pri, icat_i_sec, r_max=R_MAX, k=30)
    reg_features_i = data_utils.source_select_reg(reg_features_i)

    reg_features_i['distance'] *= 3600

    # rescale everything
    reg_features_i = data_utils.rescale(reg_features_i,zero_mag=conditions['zero_point'],
                                        pixel_rms=conditions['pixel_rms'],pixel_size=conditions['pixel_size'],
                                        psf_fwhm=conditions['psf_fwhm'],moffat_beta=conditions['moffat_beta'])

    # emulator responses
    # response_pred = data_utils.xgb_pred(model,reg_features_i)

    # reverse standardization
    # reg_features_i['response'] = data_utils.reverse_standardize(response_pred, y_mean, y_std)
    return reg_features_i

def icat2cla(icat_i_pri,icat_i_sec,model,conditions,cla_k=2):

    # make a copy for classification
    reg_features_i_cla = make_reg_features(icat_i_pri, icat_i_sec, r_max=999, k=cla_k) # r_max=99 makes sure there is at least one neighbour; this is important
    idx_cla = np.where(reg_features_i_cla['distance']>R_MAX_CLA)[0]
    reg_features_i_cla.loc[idx_cla,CLASSIFICATION_NAMES_SEC] = np.nan
    reg_features_i_cla['neighbored'] = np.full(reg_features_i_cla.shape[0],True)
    reg_features_i_cla.loc[idx_cla,'neighbored'] = False

    reg_features_i_cla['distance'] *= 3600

    # rescale everything
    reg_features_i_cla = data_utils.rescale(reg_features_i_cla,zero_mag=conditions['zero_point'],
                                        pixel_rms=conditions['pixel_rms'],pixel_size=conditions['pixel_size'],
                                        psf_fwhm=conditions['psf_fwhm'],moffat_beta=conditions['moffat_beta'])
    
    # emulator detections 
    detect_pred = data_utils.xgb_pred(model, reg_features_i_cla)
    reg_features_i_cla['detection'] = detect_pred
    return reg_features_i_cla


def bootstrap_sum_diff(d1,d2,n_resamples=10):
    # print(len(d1),len(d2))
    if (len(d1)!=0)&(len(d2)!=0):
        def sum_diff(d1, d2):
                return np.sum(d1) - np.sum(d2)
            
        # Use bootstrap (BCa interval by default)
        res = bootstrap((d1, d2), statistic=sum_diff, confidence_level=0.95, 
                    n_resamples=n_resamples, method='basic', vectorized=False, paired=False)
        stat = sum_diff(d1,d2)
        
    elif (len(d2)==0):
        def sum_diff(d1):
                return np.sum(d1)
            
        # Use bootstrap (BCa interval by default)
        res = bootstrap((d1,), statistic=sum_diff, confidence_level=0.95, 
                    n_resamples=n_resamples, method='basic', vectorized=False, paired=False)
        stat = sum_diff(d1)
    else:
        raise Exception('Something wrong with the bins/slices.')

    bootstrap_ci = res.confidence_interval
    return np.array([stat, bootstrap_ci.low, bootstrap_ci.high])


def first_term(x,y,dz,nz):
    idx = np.where((x['redshift_input_s']>dz[0])&(x['redshift_input_s']<dz[1]))[0]

    detection_weights = x['detection'].array
    detection_weights[detection_weights<DETECTION_THRESH] = 0. # to avoid far outliers of R response]

    boot_values = y[idx]*detection_weights[idx] 
    return boot_values /(dz[1]-dz[0])

def second_term(x,y,dz,nz):
    idx = np.where((x['redshift_input_p']>dz[0])&(x['redshift_input_p']<dz[1]))[0]

    detection_weights = x['detection'].array
    detection_weights[detection_weights<DETECTION_THRESH] = 0. # to avoid far outliers of R response]

    boot_values = y[idx]*detection_weights[idx] 
    return boot_values /(dz[1]-dz[0])

def n_correction(x,y,dz,nz,norm):
    f = lambda idx: (np.sum(first_term(x,y,dz[idx],nz)) - np.sum(second_term(x,y,dz[idx],nz)))/norm
    delta = Parallel(n_jobs=-1, backend='threading')(delayed(f)(i) for i in range(len(dz)))
    return np.array(delta)

def term1_normed(x,y,dz,nz,norm):
    f = lambda idx: np.sum(first_term(x,y,dz[idx],nz)/norm)
    term1 = Parallel(n_jobs=-1, backend='threading')(delayed(f)(i) for i in range(len(dz)))
    return term1

def term2_normed(x,y,dz,nz,norm):
    f = lambda idx: np.sum(second_term(x,y,dz[idx],nz)/norm)
    term2 = Parallel(n_jobs=-1, backend='threading')(delayed(f)(i) for i in range(len(dz)))
    return term2