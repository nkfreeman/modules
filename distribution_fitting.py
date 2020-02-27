# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 06:23:56 2018

@author: Nick Freeman, PhD (nkfreeman@cba.ua.edu)
"""

def info():
    print('''
    This submodule is build on the Scipy stack and includes various functions for:
    1) fitting parameteric and non-parametric distributions to data, and 
    2) randomly generating numbers from these distributions.
    ''')

    
def generate_multimodal_data(centers = 3, mean_LB = 500, mean_UB = 1000,
                             max_cv = 0.05, seed = 0, size = 100, 
                             include_plot = False):
    '''
    This function generates random multimodal data using gaussian mixtures.
    
    Arguments:
    ----------
    data: the number of (normal) distributions to use when creating
    the gaussian mixture
    
    mean_LB: the lower bound for the mean value to use for each
    distribution used
    
    mean_UB: the upper bound for the mean value to use for each
    distribution used
    
    max_cv: the maximum coefficient of variation for each distribution
    
    seed: the seed value for Numpy's random number generator
    
    size: the number of data points to generate for each distribution,
    i.e., the size of the generated sample will be size*centers
    
    include_plot: True or False to denote whether or not a histogram is
    plotted to show the sample 
       
    Returns:
    --------
    a Numpy array that contains the generated sample
    
    Dependencies:
    ------------
    This function depends on Numpy and Matplotlib (if include_plot = True)     
        
    '''
    import numpy as np

    np.random.seed(seed)

    means = []
    st_devs = []
    for i in range(centers):
        means.append(np.random.rand()*(mean_UB-mean_LB) + mean_LB)
        cv = np.random.rand()*max_cv
        st_devs.append(means[i]*cv)
    data = np.random.normal(loc = means[0], scale = st_devs[0], size = size)
    for i in range(1, centers):
        data = np.append(data, np.random.normal(loc = means[i], scale = st_devs[i], size = size))
    np.random.shuffle(data)

    if include_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize = (12, 6))

        ax.hist(data,  
                bins = 50, 
                edgecolor = 'k')
        ax.set_xlabel('Value', fontsize = 15)
        ax.set_ylabel('Frequency', fontsize = 15)
        ax.set_title('Histogram for Sample', fontsize = 15)

        plt.show()
    
    return data

def remove_outliers(data, include_plot = False, remove_mild_outliers = False):
    '''
    This function removes outliers from a dataset that is structured
    as a 1-dimensional array or list. The function always removes extreme
    outliers, i.e., data values x that satisfy 'x < q25 - 3*IQR' or 
    'x > q75 + 3*IQR', where q25 denotes the 25th percentile value, q75 
    denotes the 75th percentile value, and IQR denotes the inter-quartile
    range (q75 - q25). In addition, setting the optional remove_mild_outliers
    argument to True allows users to also remove mild outliers, i.e., data 
    values x that satisfy 'x < q25 - 1.5*IQR' or 'x > q75 + 1.5*IQR'.
    
    Arguments:
    data: a 1-dimensional list or Numpy array that includes the data
    
    include_plot: True or False to denote whether or not a histogram is
    plotted to show the data before and after outlier removal
    
    remove_mild_outliers: True or False to denote whether or not mild 
    outliers should be removed.
    
    Returns:
    data: a modifed copy of the data with outliers removed. If the data
    was provided in a list, a list is returned. Otherwise, a Numpy array
    is returned.
    
    Examples:
    
    # data provided as a list
    >>> data = [1,1,1,1,2,2,1,1,1,1,5,10,10]
    
    >>> remove_outliers(data)
    [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 5]
    
    # data provided as Numpy array
    >>> data = np.array([1,1,1,1,2,2,1,1,1,1,5,10,10])
    
    >>> remove_outliers(data, include_plot = False, remove_mild_outliers = True)
    array([1, 1, 1, 1, 2, 2, 1, 1, 1, 1])
     
        
    '''
    import numpy as np
    import matplotlib.pyplot as plt
      
    if include_plot:
        plt.hist(data, 
                 density = True, 
                 bins = 'auto', 
                 label = 'Original',  
                 color = 'grey', 
                 edgecolor = 'k')
        
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    lower_inner_fence = q25 - 1.5*iqr
    upper_inner_fence = q75 + 1.5*iqr
    lower_outer_fence = q25 - 3.0*iqr
    upper_outer_fence = q75 + 3.0*iqr
    data = data[np.logical_not((data > upper_outer_fence) | (data < lower_outer_fence))] 
    if remove_mild_outliers:
        data = data[np.logical_not((data > upper_inner_fence) | (data < lower_inner_fence))]     
            
    if include_plot:
        plt.hist(data, 
                 density = True, 
                 bins = 'auto', 
                 label = 'Outliers Removed',
                 alpha = 0.8, 
                 color = 'green', 
                 edgecolor = 'k')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
        
    return data




def fit_distributions(data, fit_all = False):
    '''
    This function fits all of the parametric distributions available
    in the scipy.stats module to a provided dataset, computes the 
    sum-of-squares error (SSE) for each distribution, and returns a
    dictionary that specifes the SSE, distribution parameters, and
    a frozen distribution generator for each distribution. The distribution 
    generator may be used with '.rvs()' method availble in scipy.stats to 
    generate a random sample.
    
    Arguments:
    data: a 1-dimensional list or Numpy array that includes the data
    
    fit_all: True or False to specify whether of not the function will
    attempt to fit all available distributions. If False, only a subset of
    common distributions are fit.
    
    Returns:
    data: a dictionary that specifes the SSE, distribution parameters, and
    a generator object for each distribution. The keys of the dictionary are
    the index values for each distribution when sorted by SSE. Thus, the 
    distribution associated with key 0 is the best fitting distribution.
         
        
    '''
    import warnings
    import numpy as np
    import scipy.stats as st
        
    # The following lines convert the data to a histogram and
    # compute the midpoints of the bins
    y, x = np.histogram(data, bins='auto', density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    
    # Initialize empty list for storing fit information
    dist_list = []
    dist_gen_list = []
    sse_list = []
    args_list = []

    # Distributions to check
    all_distributions = []
    if fit_all:   
        for this in dir(st):
            if ("fit" in eval("dir(st." + this +")")) and ("rvs" in eval("dir(st." + this +")")):
                all_distributions.append(this)
    else:
        for this in ['beta', 'chi2', 'erlang', 'expon', 'gamma', 'logistic',
                     'lognorm', 'norm', 'triang', 'truncnorm', 'uniform']:
            if ("fit" in eval("dir(st." + this +")")) and ("rvs" in eval("dir(st." + this +")")):
                all_distributions.append(this)

    # Estimate distribution parameters from data
    for dist_string in all_distributions:
        distribution = eval("st." + dist_string)
        
        sse = None
        args = None
        dist = None
        
        # Try to fit the distribution
        try:
                      
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                args = distribution.fit(data)
                dist = distribution.freeze(*args)

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, *args)
                sse = np.sum(np.power(y - pdf, 2.0))
        
        except:
            pass
        
        else:
            # Update lists
            dist_list.append(dist_string)
            sse_list.append(sse)
            args_list.append(args)
            dist_gen_list.append(dist)
            
    
    # Use the lists to construct a dictionary object        
    fit_comparison = zip(sse_list, dist_list, args_list, dist_gen_list)
    fit_comparison = sorted(fit_comparison)
    sse_list, dist_list, args_list, dist_gen_list = map(list, zip(*fit_comparison))
    
    
    fit_comparison = {index: {'Name': name, 
                              'Generator': gen, 
                              'SSE': sse, 
                              'Args': args} \
                      for index, (name, gen, sse, args) in enumerate(zip(dist_list, 
                                                                         dist_gen_list, 
                                                                         sse_list, 
                                                                         args_list))}

    return fit_comparison



def find_min_max_diff(data):
    '''
    This function finds the minimum and maximum difference between any
    two consecutive values in a data set
    
    Arguments
    data:
        a 1-dimensional list or Numpy array that includes the data
        
    Returns
    min_val, max_val: 
        a tuple specifiying the minimum and maximum difference between any
        two consecutive values in a data set
    
    '''
    
    import numpy as np
    
    data = np.array(data)
    
    data_copy = data.copy()
    data_copy = np.unique(data_copy)
    data_copy.sort()
    min_val = np.min(np.roll(data_copy, -1)[:-1] - data_copy[:-1])
    max_val =  np.max(np.roll(data_copy, -1)[:-1] - data_copy[:-1])
    
    return min_val, max_val 



def fit_best_kde(data, steps = 25, rtol = 0.1, cv = 3, fit_sample_size = None):
    '''
    This function determines a best fitting kernel density estimate
    using scikit-learn's sklearn.neighbors.KernelDensity method along 
    scikit-learn's sklearn.model_selection.GridSearchCV method. In 
    particular, the GridSearchCV method is used to try all possible
    kernel types with 100 evenly spaced bandwidths between the minimum
    and maximum differences between values in the provided data.
    
    Arguments:
    data: a 1-dimensional list or Numpy array that includes the data
    
    rtol: the relative tolerance passed to sklearn.neighbors.KernelDensity 
    method. Higher values offer faster computational times at the cost of
    accuracy.
    
    cv: the number of cross-validation splits the sklearn.model_selection.GridSearchCV 
    method uses to identify the best kde.
    
    fit_sample_size: a value that, if specified, denotes that a random sample
    of size sample_size should be used to fit the kernel density estimate. This
    functionality is added to reduce the high computational times that may
    occur when the provided data is large.
    
    Returns:
    data: a dictionary specifes the best bandwidth and kernel.         
        
        
    '''
    import sklearn.neighbors as skneighbor
    from sklearn.model_selection import GridSearchCV
    import warnings
    import numpy as np
    
    data = np.array(data)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        if fit_sample_size is not None:
            data = np.random.choice(data.ravel(), size = fit_sample_size, replace = False)

        min_val, max_val = find_min_max_diff(data)

        params = {'bandwidth': np.linspace(min_val, max_val, steps),
                 'kernel': skneighbor.kde.VALID_KERNELS}
        grid = GridSearchCV(skneighbor.KernelDensity(rtol = rtol), params, cv = cv)
        grid.fit(data.reshape(-1, 1))

        return grid.best_params_
    
    
    
def pdf_from_kde(data, min_val = 0, max_val = None, bandwidth = 1.0, kernel = 'gaussian'):
    '''
    This function generates a probability density function (PDF) that is 
    based on a kernel density estimate that is fit using scikit-learn's
    sklearn.neighbors.KernelDensity method. Specifically, it returns two
    objects, pdfx and pdfy, that contain the support and probability values
    that define the PDF, respectively. 
    
    Arguments:
    data: a 1-dimensional list or Numpy array that includes the data
    
    min_val: the minimum value to include in the PDF support (default
    is min_value - 0.10*[range between max_val and min_val values])
    
    max_val: the maximum value to include in the PDF support (default
    is max_value + 0.10*[range between max_val and min_val values])
    
    bandwidth: the bandwidth for the kernel density estimate.
    
    cv: the kernel type, which is passed directly to scikit-learn's
    sklearn.neighbors.KernelDensity method
    
    Returns:
    data: a dictionary with two keys, x and y. The values are NumPy arrays for the 
    support (x) and probability values (y) that define the PDF.
        
        
    '''

    import sklearn.neighbors as skneighbor
    import numpy as np
    
    data = np.array(data)
        
    if min_val is None:
        min_val = data.min() - 0.10*(data.max() - data.min())
        
    if max_val is None:
        max_val = data.max() + 0.10*(data.max() - data.min())

    pdfx = np.linspace(min_val, max_val, 1000)
    pdfy = np.exp(skneighbor.KernelDensity(bandwidth = bandwidth, 
                                           kernel= kernel,
                                           rtol = 0.1).fit(data.reshape(-1,1)).score_samples(pdfx.reshape(-1,1)))
    pdfy = pdfy/pdfy.sum()
    return {'x': pdfx, 'y': pdfy}


def trunc_kde_sample(pdfx, pdfy, low = None, high = None, sample_size = 100, seed = 0):
    '''
    This function requires two array-like data objects, pdfx and pdfy,
    that specify the support and probabilities for a probability density
    function (PDF) defined by a kernel density estimate (see the 
    pdf_from_kde function). These data objects are used to generate a sample
    from the defined PDF that falls between optional lower and upper bounds.
    
    Arguments:
    pdfx: a 1-dimensional list or Numpy array that specifies the PDF support
    
    pdfy: a 1-dimensional list or Numpy array that specifies the probability
    for each value in the PDF support
    
    low: the lower bound for values in the generated sample
    
    high: the upper bound for values in the generated sample
    
    sample_size: the size of the sample to be drawn (1-dimensional only)
    
    seed: seed for Numpy's random number generator
    
    Returns:
    gen_sample: the generated sample as a Numpy array.
        
        
    '''
    import numpy as np
    
    pdfx = np.array(pdfx)    
    pdfy = np.array(pdfy)
    
    if (low != None) and (high != None):
        mask = pdfx >= low
        pdfx = pdfx[mask]
        pdfy = pdfy[mask]
        
        mask = pdfx <= high
        pdfx = pdfx[mask]
        pdfy = pdfy[mask]
        
    elif (low != None):
        mask = pdfx >= low
        pdfx = pdfx[mask]
        pdfy = pdfy[mask]
        
    elif (high != None):
        mask = pdfx <= high
        pdfx = pdfx[mask]
        pdfy = pdfy[mask]
        
    pdfy = pdfy/pdfy.sum()
    cdf = pdfy.cumsum()
    cdf = cdf - cdf.min()
    
    np.random.seed(seed)
    random_nums = np.random.rand(sample_size)
    gen_sample = []
    for i in random_nums:
        gen_sample.append(pdfx[(cdf < i).argmin() - 1])
    
    return np.array(gen_sample)