import pandas as pd
import numpy as np
import xgboost as xgb

def source_select_reg(dataset,cuts=[[18,26],[18,26],[0.1,1.5],[0.1,1.5],[0,10]]):
    '''
    Perform source selection before going into training.
    '''
    idx_sel = np.where((\
                        (dataset['r_input_s']>cuts[0][0])&(dataset['r_input_s']<cuts[0][1])\
                        &(dataset['r_input_p']>cuts[1][0])&(dataset['r_input_p']<cuts[1][1])\
                        &(dataset['Re_input_s']>cuts[2][0])&(dataset['Re_input_s']<cuts[2][1])\
                        &(dataset['Re_input_p']>cuts[3][0])&(dataset['Re_input_p']<cuts[3][1])\
                        &(dataset['distance']>cuts[4][0])&(dataset['distance']<cuts[4][1])\
    ))[0] 
    dataset = dataset.iloc[idx_sel].reset_index()
    return dataset

def source_select_cla(dataset,cuts=[[18,26],[18,26],[0.1,1.5],[0.1,1.5],[0,10]]):
    '''
    Perform source selection before going into training.
    '''
    idx_sel = np.where(
                        (dataset['r_input_p']>cuts[1][0])&(dataset['r_input_p']<cuts[1][1])\
                        &(dataset['Re_input_p']>cuts[3][0])&(dataset['Re_input_p']<cuts[3][1])\
                        &((dataset['distance']>cuts[4][0])&(dataset['distance']<cuts[4][1])|(~dataset['neighbored']))
                        )[0]
    dataset = dataset.iloc[idx_sel].reset_index()
    return dataset

def mag2flux(mag,zero_mag):
    return 10**(-0.4*(mag-zero_mag))

def flux2mag(flux,zero_mag):
    return -2.5*np.log10(flux)+zero_mag
    
def rescale(dataset,
            pixel_rms=6,pixel_size=0.2,
            zero_mag=30, 
            psf_fwhm=0.6,moffat_beta=2.4):

    psf_size = moffat_fwhm2Re(psf_fwhm,moffat_beta)
    aperture_rms = pixel_rms*(psf_size/pixel_size)**2*np.pi

    # if 'neighbored' not in dataset.columns:
    post_Re_p = _convolved_size(dataset['Re_input_p'].array, psf_size)
    post_Re_s = _convolved_size(dataset['Re_input_s'].array, psf_size)
    
    # unit transformation
    dataset['distance_scaled']= dataset['distance'] / post_Re_p
    # dataset['distance_scaled'] = dataset['distance'] / (post_Re_s+post_Re_p)
    
    dataset['Re_input_p_scaled'] = dataset['Re_input_p'] / post_Re_p
    dataset['Re_input_s_scaled'] = dataset['Re_input_s'] / post_Re_s

    flux_input_p = mag2flux(dataset['r_input_p'],zero_mag)
    flux_input_s = mag2flux(dataset['r_input_s'],zero_mag)
    dataset['flux_ratio'] = np.log10(flux_input_p/flux_input_s)

    dataset['r_input_p_scaled'] = flux2mag(flux_input_p/aperture_rms,zero_mag)
    dataset['r_input_s_scaled'] = flux2mag(flux_input_s/aperture_rms,zero_mag)

    return dataset


def rescale_v2(dataset):
    """
    Rescale quantities using per-row conditions.
    Assumes the following columns exist in `dataset`:
    ['pixel_rms', 'pixel_size', 'zero_point', 'psf_fwhm', 'moffat_beta']
    """
    
    # Compute psf_size for each row
    dataset["psf_size"] = moffat_fwhm2Re(dataset["psf_fwhm_input_p"], dataset["moffat_beta_input_p"])
    
    # Aperture rms (vectorized)
    dataset["aperture_rms"] = (
        dataset["pixel_rms_input_p"] * (dataset["psf_size"] / dataset["pixel_size_input_p"])**2 * np.pi
    )
    
    # Convolved sizes
    dataset["post_Re_p"] = _convolved_size(dataset["Re_input_p"], dataset["psf_size"])
    dataset["post_Re_s"] = _convolved_size(dataset["Re_input_s"], dataset["psf_size"])
    
    # Unit transformation
    dataset["distance_scaled"] = dataset["distance"] / dataset["post_Re_p"]
    # Or alternative:
    # dataset["distance_scaled"] = dataset["distance"] / (dataset["post_Re_p"] + dataset["post_Re_s"])
    
    dataset["Re_input_p_scaled"] = dataset["Re_input_p"] / dataset["post_Re_p"]
    dataset["Re_input_s_scaled"] = dataset["Re_input_s"] / dataset["post_Re_s"]
    
    # Fluxes
    flux_input_p = mag2flux(dataset["r_input_p"], dataset["zero_point_input_p"])
    flux_input_s = mag2flux(dataset["r_input_s"], dataset["zero_point_input_p"])
    dataset["flux_ratio"] = np.log10(flux_input_p / flux_input_s)
    
    # Scaled magnitudes
    dataset["r_input_p_scaled"] = flux2mag(flux_input_p / dataset["aperture_rms"], dataset["zero_point_input_p"])
    dataset["r_input_s_scaled"] = flux2mag(flux_input_s / dataset["aperture_rms"], dataset["zero_point_input_p"])
    
    return dataset

def moffat_fwhm2Re(fwhm,beta):
    factor = np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))/2
    return fwhm*factor

def moffat_Re2fwhm(re,beta):
    factor = np.sqrt((2**(1/beta)-1)/(2**(1/(beta-1))-1))*2
    return re*factor


def _convolved_size(r,psf_size):
    return np.sqrt(np.power(r,2)+np.power(psf_size,2))  


def _deconvolved_size(r,psf_size):
    return np.sqrt(np.power(r,2)-np.power(psf_size,2))
    

def xgb_pred(model,cat):

    features = cat[model.feature_names]
    DM = xgb.DMatrix(features)
    pred = model.predict(DM,iteration_range=[0,model.best_iteration])
    return pred


def double_pred(double_model, cat, standard):

    [[y_mean1,y_std1],[y_mean2,y_std2]] = standard
    regression_names = double_model[0].feature_names
    
    idx1 = np.where(cat['distance']<3)[0]
    DM1 = xgb.DMatrix(cat[regression_names].iloc[idx1])
    pred1 = double_model[0].predict(DM1)
    pred1 = data_utils.reverse_standardize(pred1,y_mean1,y_std1)

    idx2 = np.where(cat['distance']>3)[0]
    DM2 = xgb.DMatrix(cat[regression_names].iloc[idx2])
    pred2 = double_model[1].predict(DM2)
    pred2 = data_utils.reverse_standardize(pred2,y_mean2,y_std2)

    pred = np.zeros(cat.shape[0])
    # print(idx1.shape,pred1.shape)
    pred[idx1] = pred1
    pred[idx2] = pred2
    return pred

def load_double_model(path,model1,model2):
     
    bst1 = xgb.Booster({'device':'cuda'})
    bst1.load_model(os.path.join(path,model1))
    [y_mean1, y_std1] = np.load(os.path.join(path,'train_standardization_1.npy'))
    
    bst2 = xgb.Booster({'device':'cuda'})
    bst2.load_model(os.path.join(path,model2))
    [y_mean2, y_std2] = np.load(os.path.join(path,'train_standardization_2.npy'))

    boundary1 = np.load(os.path.join(path,'train_boundary_1.npy'))
    boundary2 = np.load(os.path.join(path,'train_boundary_2.npy'))

    return [bst1,bst2], [[y_mean1,y_std1],[y_mean2,y_std2]], [boundary1,boundary2]


def boundary_selection(cat,keys,boundary):
    
    assert len(keys)==boundary.shape[0]
    if len(keys)==7:
        idx = np.where(
                     (cat[keys[0]].values>boundary[0,0])&(cat[keys[0]].values<boundary[0,1])\
                    &((cat[keys[1]].values>boundary[1,0])&(cat[keys[1]].values<boundary[1,1])|np.isnan(cat[keys[1]].values))\
                    &(cat[keys[2]].values>boundary[2,0])&(cat[keys[2]].values<boundary[2,1])\
                    &((cat[keys[3]].values>boundary[3,0])&(cat[keys[3]].values<boundary[3,1])|np.isnan(cat[keys[3]].values))\
                    &(cat[keys[4]].values>boundary[4,0])&(cat[keys[4]].values<boundary[4,1])\
                    &((cat[keys[5]].values>boundary[5,0])&(cat[keys[5]].values<boundary[5,1])|np.isnan(cat[keys[3]].values))\
                    &((cat[keys[6]].values>boundary[6,0])&(cat[keys[6]].values<boundary[6,1])|np.isnan(cat[keys[6]].values))\
                       )[0]
    if len(keys)==3:
        idx = np.where(
                     (cat[keys[0]].values>boundary[0,0])&(cat[keys[0]].values<boundary[0,1])\
                    &((cat[keys[1]].values>boundary[1,0])&(cat[keys[1]].values<boundary[1,1])|np.isnan(cat[keys[1]].values))\
                    &(cat[keys[2]].values>boundary[2,0])&(cat[keys[2]].values<boundary[2,1])\
                    # &((cat[keys[3]].values>boundary[3,0])&(cat[keys[3]].values<boundary[3,1])|np.isnan(cat[keys[3]].values))\
                    # &(cat[keys[4]].values>boundary[4,0])&(cat[keys[4]].values<boundary[4,1])\
                    # &(cat[keys[5]].values>boundary[5,0])&(cat[keys[5]].values<boundary[5,1])\
                    # &((cat[keys[6]].values>boundary[6,0])&(cat[keys[6]].values<boundary[6,1])|np.isnan(cat[keys[6]].values))\
                       )[0]
    
    return idx


def load_data(path, target_name, 
                  normalized=False, rescaled=False,
                  pixel_rms=6, zero_mag=30, psf_size=0.6, shear=0.1,
              select_cuts=[[18,26],[18,26],[0.1,1.5],[0.1,1.5],[0,10]]):
    '''
    Load data and preprocess.
    '''
    
    dataset = pd.read_feather(path)
    if shear:
        dataset = source_select_reg(dataset,cuts=select_cuts)
    else:
        dataset = source_select_cla(dataset,cuts=select_cuts)
    
    if rescaled:
        dataset = rescale(dataset)

    x = dataset.drop(target_name,axis=1)
    y = dataset[target_name]
    
    if shear:
        y = y/shear
        
    if normalized:
        y_mean = np.mean(y); y_std = np.std(y,ddof=1)
        # y = (y - y_mean)/y_std # standardize; usually not necessary in tree-based models
        y = standardize(y,y_mean,y_std) # standardize; usually not necessary in tree-based models
        print(f'Labels standardized with: y = (y - {y_mean})/{y_std} .')
    
    return x, y

def standardize(y,y_mean,y_std):
    return (y-y_mean)/y_std
    
def reverse_standardize(y,y_mean,y_std):
    return y*y_std+y_mean



def get_bins(x, y, bins=11):
    
    upper, lower = np.max(x), np.min(x)
    bins = np.linspace(upper,lower,bins)
    
    digitized = np.digitize(x,bins)
    bin_means = np.array([std_of_mean(y[digitized == i]) for i in range(1, len(bins))])
    
    return (bins[1:]+bins[:-1])/2, bin_means

def get_bins_sum(x, y, bins=11):
    '''
    sum in each bin divided by the bin width.
    '''
    
    upper, lower = np.max(x), np.min(x)
    bins = np.linspace(lower,upper,bins)
    
    digitized = np.digitize(x,bins)
    bin_sums = np.array([np.sum(y[digitized == i]) for i in range(1, len(bins))]) # !be careful with (bins[i]-bins[i-1])
    bin_sums_err = np.array([np.std(y[digitized == i],ddof=1)*np.sqrt(y[digitized == i].shape[0]) for i in range(1, len(bins))])
    
    return (bins[1:]+bins[:-1])/2, bin_sums, bin_sums_err




def polyfit(x,y):
    
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    
    # Extract the best-fit values and standard deviations
    k, b = coeffs
    std_k = np.sqrt(cov[0, 0])
    std_b = np.sqrt(cov[1, 1])
    return [k,std_k,b,std_b]
    


def std_of_mean(data,axis=0):
    '''
    non-nan.
    '''
    
    mean = np.nanmean(data,axis=axis)
    std_mean = np.nanstd(data,axis=axis,ddof=1)/np.sqrt(np.count_nonzero(~np.isnan(data),axis=axis))
    return mean,std_mean

def std_of_the_weighted_mean(values, weights, axis=0):
    """
    Return the weighted mean and the unbiased standard deviation of the weighted mean.
    """
    average = np.average(values, weights=weights, axis=axis)
    sum_w = np.sum(weights, axis=axis)
    sum_w2 = np.sum(weights**2, axis=axis)
    sample_variance = np.sum(weights*(values-average)**2, axis=axis) /(sum_w - sum_w2/sum_w)
    variance = sample_variance * np.sum(weights**2,axis=axis) / np.sum(weights,axis=axis)**2
    return (average, np.sqrt(variance))