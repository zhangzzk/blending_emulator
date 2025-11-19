import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from itertools import combinations
import xgboost as xgb
import data_utils


def get_bins(x, y, bins=11):
    
    upper, lower = np.max(x), np.min(x)
    bins = np.linspace(upper,lower,bins)
    
    digitized = np.digitize(x,bins)
    bin_means = np.array([std_of_mean(y[digitized == i]) for i in range(1, len(bins))])
    # bin_means2 = np.array([std_of_mean(y2[digitized == i]) for i in range(1, len(bins))])
    
    return bins, bin_means

def std_of_mean(data,axis=0):
    
    mean = np.nanmean(data,axis=axis)
    std_mean = np.nanstd(data,axis=axis)/np.sqrt(np.count_nonzero(~np.isnan(data),axis=axis))
    return mean,std_mean


# def kdt_neighbor_finder(icat1, icat2,
#                         features=['RA','DEC'],
#                         r_min=0,r_max=10,k=30):
    
#     sec_pos = icat2[features].to_numpy()
#     pri_pos = icat1[features].to_numpy()

#     kdt_in = KDTree(sec_pos)
#     dst,ind = kdt_in.query(pri_pos,k=k,distance_upper_bound=r_max,workers=-1)

#     found_idx = np.where((ind.reshape(-1) != sec_pos.shape[0])&\
#                          (dst.reshape(-1) > r_min)\
#                          )[0]
    
#     idx1 = np.array([np.arange(0,icat1.shape[0]),]*k).T.reshape(-1)[found_idx]
#     idx2 = ind.reshape(-1)[found_idx]
    
#     return idx1, idx2, dst.reshape(-1)[found_idx]

def kdt_neighbor_finder(pos1, pos2,
                        r_min=0,r_max=10,k=30):

    kdt_in = KDTree(pos2)
    dst,ind = kdt_in.query(pos1,k=k,distance_upper_bound=r_max,workers=-1)

    found_idx = np.where((ind.reshape(-1) != pos2.shape[0])&\
                        (dst.reshape(-1) > r_min)\
                        )[0]
    
    idx1 = np.array([np.arange(0,pos1.shape[0]),]*k).T.reshape(-1)[found_idx]
    idx2 = ind.reshape(-1)[found_idx]
    
    return idx1, idx2, dst.reshape(-1)[found_idx]

def remove_detection_w_bright_neighbour(x,y,flux,ratio_max=10,r_min=0,r_max=5/3600):

    pos = np.vstack((x,y)).T
    idx1,idx2,_ = kdt_neighbor_finder(pos,pos,r_min=r_min,r_max=r_max,k=30)
    flux_ratio = flux[idx2]/flux[idx1]

    reject_idx = idx1[np.where(flux_ratio>ratio_max)[0]]
    return np.unique(reject_idx)

def boundary_selection(cat,keys,boundary):
    
    assert len(keys)==boundary.shape[0]
    # assert len(keys)==7
    idx = np.where(
                (cat[keys[0]].values>boundary[0,0])&(cat[keys[0]].values<boundary[0,1])\
                &(cat[keys[1]].values>boundary[1,0])&(cat[keys[1]].values<boundary[1,1])\
                &(cat[keys[2]].values>boundary[2,0])&(cat[keys[2]].values<boundary[2,1])\
                &(cat[keys[3]].values>boundary[3,0])&(cat[keys[3]].values<boundary[3,1])\
                # &(cat[keys[4]].values>boundary[4,0])&(cat[keys[4]].values<boundary[4,1])\
                # &(cat[keys[5]].values>boundary[5,0])&(cat[keys[5]].values<boundary[5,1])\
                &(cat[keys[6]].values>boundary[6,0])&(cat[keys[6]].values<boundary[6,1])\
                )[0]
    
    return idx

def remove_detection_w_bright_neighbour_prob(pos,flux,detection_prob,ratio_max=10,r_min=0,r_max=5/3600):

    kdt_in = KDTree(pos)
    dst,ind = kdt_in.query(pos,k=30,distance_upper_bound=r_max,workers=-1)
    found_idx = np.where((ind != pos.shape[0])&\
                        (dst > r_min)\
                        )
    
    flux_neighbor = np.full(ind.shape,0.)
    flux_neighbor[found_idx] = flux[ind[found_idx]]
    ratio = np.divide(flux_neighbor,flux.reshape(-1,1))
    found_idx = np.where(ratio>ratio_max)

    prob = np.full(ind.shape,0.)
    prob[found_idx] = detection_prob[ind[found_idx]]
    return detection_prob * np.prod(1-prob,axis=1)

def cla_predict(x,bst,boundaries,**obs_cons):

    # x_ = x
    x_ = data_utils.rescale(x.copy(), 
                            pixel_rms=obs_cons['pixel_rms'],pixel_size=obs_cons['pixel_size'],
                            psf_fwhm=obs_cons['psf_fwhm'],moffat_beta=obs_cons['moffat_beta'])
    
    idx = boundary_selection(x_,bst.feature_names,boundaries)
    DM_ = xgb.DMatrix(x_[bst.feature_names].iloc[idx])
    
    y_ = bst.predict(DM_,iteration_range=[0,bst.best_iteration])
    # print(idx.shape, x_.shape)
    return y_,x_,idx


def contour_plot():
    
    combs = combinations(regression_names,2)
    plt.figure(figsize=(10,10))

    for i,comb in enumerate(combs):

        hist1, xedges1, yedges1 = np.histogram2d(reg_features_detected[comb[0]].iloc[idx],
                                                reg_features_detected[comb[1]].iloc[idx],bins=10);
        hist2, xedges2, yedges2 = np.histogram2d(x_test[comb[0]].iloc[idx],
                                                x_test[comb[1]].iloc[idx],bins=10);

        x1,y1 = np.meshgrid(xedges1[1:], yedges1[1:])
        x2,y2 = np.meshgrid(xedges2[1:], yedges2[1:])

        plt.subplot(10,10,i+1)

        plt.contour(x1.T,y1.T,hist1,cmap='Reds')
        plt.contour(x2.T,y2.T,hist2,cmap='Blues',linestyles='dashed')

        # if k in [f'sfr_{i}' for i in range(6)]:
            # plt.xscale('log')

        plt.xlabel(comb[0])
        plt.ylabel(comb[1])

    plt.tight_layout()
    
    
def hist1d_compare(names,data1,data2,figsize=(10,5),subplots=(2,4),bins=50):

    plt.figure(figsize=figsize)

    for i,k in enumerate(names):

        plt.subplot(subplots[0],subplots[1],i+1)
        plt.hist(data1[k],density=True,histtype='step',bins=bins)
        plt.hist(data2[k],density=True,histtype='step',bins=bins)

        plt.xlabel(k)

    plt.tight_layout()
    