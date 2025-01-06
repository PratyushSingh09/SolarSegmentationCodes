'''COMPLETELY UNEDITED'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sunkit_image.time_lag import time_lag,cross_correlation,get_lags,_get_bounds_indices,_dask_check

from astropy.io import fits
import astropy.units as u

from scipy.ndimage import shift
from scipy.signal import correlate,correlate2d

from numba import jit

from FITSImageFunctions import get_maps_for_pair_sunpy,get_maps_for_pair_fits,time_int_check
def ccr_slow(signal_a,signal_b,lags,method='direct',rev=False,ccrsize=None,st=None,sp=None):
        '''Modified from sunkit image cross correlation function'''
        delta_lags = np.diff(lags)
        if not u.allclose(delta_lags, delta_lags[0]):
            raise ValueError("Lags must be evenly sampled")
        n_time = (lags.shape[0] + 1) // 2
        if signal_a.shape != signal_b.shape:
            raise ValueError("Signals must have same shape.")
        if signal_a.shape[0] != n_time:
            raise ValueError("First dimension of signal must be equal in length to time array.")
        # Reverse the first timeseries #WHY?
        if rev:
            signal_a = signal_a[::-1]
        
        # Normalize by mean and standard deviation
        fill_value = signal_a.max()
        std_a = signal_a.std(axis=0)
        # Avoid dividing by zero by replacing with some non-zero dummy value. Note that
        # what this value is does not matter as it will be mulitplied by zero anyway
        # since std_dev == 0 any place that signal - signal_mean == 0. We use the max
        # of the signal as the fill_value in order to support Quantities.
        std_a = np.where(std_a == 0, fill_value, std_a)
        v_a = (signal_a - signal_a.mean(axis=0)[np.newaxis]) / std_a[np.newaxis]
        std_b = signal_b.std(axis=0)
        std_b = np.where(std_b == 0, fill_value, std_b)
        v_b = (signal_b - signal_b.mean(axis=0)[np.newaxis]) / std_b[np.newaxis]

        c=correlate([v_a[:,0,0]],[v_b[:,0,0]],method='direct')

        cc=np.zeros((c.shape[1],v_a.shape[1],v_a.shape[2]))
        for i in range(v_a.shape[1]):
            for j in range(v_a.shape[2]):
                print(i,j)
                if method=='direct':
                    c=correlate([v_a[:,i,j]],[v_b[:,i,j]],method='direct')
                elif method=='fft':
                    c=correlate([v_a[:,i,j]],[v_b[:,i,j]],method='fft')
                elif method=='fill':
                    c=correlate2d([v_a[:,i,j]],[v_b[:,i,j]],boundary='fill')
                elif method=='symm':
                    c=correlate2d([v_a[:,i,j]],[v_b[:,i,j]],boundary='symm')
                elif method=='c_corr_pro':
                    c=c_correlate_pro(v_a[:,i,j],v_b[:,i,j],np.array((lags/12).value,dtype=int),size=ccrsize,st=st,sp=sp)
                else:
                    c=correlate2d([v_a[:,i,j]],[v_b[:,i,j]],boundary='wrap')
                cc[:,i,j]=c.flatten()
        # cc= np.reshape(np.array(cc),(cc[0].shape[1],siga.shape[1],siga.shape[2]))
        # cc=np.array(cc,dtype=np.float32)
        # return cc / signal_a.shape[0]
        return cc 

def c_correlate_pro(s_1, s_2, lags,size='const',st=None,sp=None):
        """
        Numpy implementation of c_correlate.pro IDL routine
        """
        # ensure signals are of equal length
        assert s_1.shape == s_2.shape
        n_s = s_1.shape[0]
        # center both signals
        s_1_center = s_1 - s_1.mean()
        s_2_center = s_2 - s_2.mean()
        # allocate space for correlation
        correlation = np.zeros(lags.shape)
        # iterate over lags
        for i,l in enumerate(lags[st:sp]):
            if size=='const':
                if l >= 0:
                    # print(l,n_s)
                    tmp = s_1_center[:(n_s//2)] * s_2_center[l:(n_s//2+l)]
                else:
                    # print(l,n_s)
                    tmp = s_1_center[-l:(n_s//2-l)] * s_2_center[:(n_s//2)]
            else:
                if l >= 0:
                    tmp = s_1_center[:(n_s - l)] * s_2_center[l:]
                else:
                    tmp = s_1_center[-l:] * s_2_center[:(n_s + l)]
            correlation[i] = tmp.sum()
        # Divide by standard deviation of both
        correlation /= np.sqrt((s_1_center**2).sum() * (s_2_center**2).sum())
        
        return correlation
def timelag_slow(signal_a, signal_b, time, lag_bounds = None,method='direct', rev=False,ccrsize=None,**kwargs):
    r"""
    CHANGED TO COMPUTE CORRELATION WITHOUT FFT
    Compute the time lag that maximizes the cross-correlation
    between ``signal_a`` and ``signal_b``.

    For a pair of signals :math:`A,B`, e.g. time series from two EUV channels
    on AIA, the time lag is the lag which maximizes the cross-correlation,

    .. math::

        \tau_{AB} = \mathop{\mathrm{arg\,max}}_{\tau}\mathcal{C}_{AB},

    where :math:`\mathcal{C}_{AB}` is the cross-correlation as a function of
    lag (computed in :func:`cross_correlation`). Qualitatively, this can be
    thought of as how much `signal_a` needs to be shifted in time to best
    "match" `signal_b`. Note that the sign of :math:`\tau_{AB}`` is determined
    by the ordering of the two signals such that,

    .. math::

        \tau_{AB} = -\tau_{BA}.

    Parameters
    ----------
    signal_a : array-like
        The first dimension must be the same length as ``time``.
    signal_b : array-like
        Must have the same dimensions as ``signal_a``.
    time : `~astropy.units.Quantity`
        Time array corresponding to the intensity time series
        ``signal_a`` and ``signal_b``.
    lag_bounds : `~astropy.units.Quantity`, optional
        Minimum and maximum lag to consider when finding the time
        lag that maximizes the cross-correlation. This is useful
        for minimizing boundary effects.

    Other Parameters
    ----------------
    array_check_hook : function
        Function to apply to the resulting time lag result. This should take in the
        `lags` array and the indices that specify the location of the maximum of the
        cross-correlation and return an array that has used those indices to select
        the `lags` which maximize the cross-correlation. As an example, if `lags`
        and `indices` are both `~numpy.ndarray` objects, this would just return
        `lags[indices]`. It is probably only necessary to specify this if you
        are working with arrays that are something other than a `~numpy.ndarray`
        or `~dask.array.Array` object.

    Returns
    -------
    array-like
        Lag which maximizes the cross-correlation. The dimensions will be
        consistent with those of ``signal_a`` and ``signal_b``, i.e. if the
        input arrays are of dimension ``(K,M,N)``, the resulting array
        will have dimensions ``(M,N)``. Similarly, if the input signals
        are one-dimensional time series ``(K,)``, the result will have
        dimension ``(1,)``.

    References
    ----------
    * Viall, N.M. and Klimchuk, J.A.
      Evidence for Widespread Cooling in an Active Region Observed with the SDO Atmospheric Imaging Assembly
      ApJ, 753, 35, 2012
      (https://doi.org/10.1088/0004-637X/753/1/35)
    """
   
    

    array_check = kwargs.get("array_check_hook", _dask_check)
    lags = get_lags(time)
    # print(lags,'l')
    start, stop = _get_bounds_indices(lags, lag_bounds)
    # print(start,stop,'ss')
    # print(lags[start:stop])
    cc = ccr_slow(signal_a, signal_b, lags,method,rev=rev,ccrsize=ccrsize,st=start,sp=stop)
    if ccrsize=='const':
        i_max_cc = cc.argmax(axis=0)
    else:
        i_max_cc = cc[start:stop].argmax(axis=0)
    return array_check(lags[start:stop], i_max_cc),cc,lags
def vkcmap():
    '''Colourmap for time lag map similar to the one used by Viall and Klimchuk'''

    c_file = np.loadtxt(r"E:\AIA Data\Coronal Loop\Extra\IDLcolortable4.txt")
    c_file=c_file/255
    cmap = ListedColormap(c_file)

    return cmap

def save_time_lag_segmented(w1,w2,pix_dir,seg_dir,tlag_file,tlag_img_file):
    '''
    Create and save time lag maps between segmented images in two different wavelengths. The sign of time lags is defined by the order of the two filters such that time_lag(w1,w2)=-time_lag(w2,w1)


    w1: The first AIA filter
    w2: The second AIA filter
    pix_dir: The directory where the original AIA maps are stored. The length of this should be equal to seg_dir. This is used to check if the images are equally spaced in time and to obtain those timestamps. It is inefficient and can be implemented better
    seg_dir: The directory where the segmented AIA maps are stored. The length of this should be equal to pix_dir
    tlag_file: The save location to save time lag map as a FITS image
    tlag_img_file: The save location to save time lag map as a image (for eg as a png)


    '''

    Mseq335_p,Mseq211_p = get_maps_for_pair_sunpy(pix_dir,w1,w2)
    Mseq335,Mseq211 = get_maps_for_pair_fits(seg_dir,w1,w2)

    timelist = time_int_check(Mseq335_p,Mseq211_p)[::5]
    '''Extract numpy arrays from map sequences'''
    arr211 =Mseq211.as_array()
    arr335 =Mseq335.as_array()
    
    '''2D arrays for time lags obtained my this function. Lag bounds of 3600 seconds important for two hour time intervals to avoid edge effects'''
    tlag = time_lag(arr335.T,arr211.T,timelist*u.s,lag_bounds=(-3600*u.s,3600*u.s))

    hdu = fits.PrimaryHDU(tlag)
    hdu.writeto(tlag_file,overwrite=True)

    '''Colourmap for time lag map similar to the one used by Viall and Klimchuk'''
    c_file = np.loadtxt(r"E:\AIA Data\Coronal Loop\Extra\IDLcolortable4.txt")
    c_file=c_file/255
    cmap = ListedColormap(c_file)

    plt.figure(dpi=300)
    plt.imshow(np.array(hdu.data),cmap=cmap,origin='lower')
    plt.savefig(tlag_img_file)

def save_time_lag(w1,w2,pix_dir,tlag_file=None,tlag_img_file=None,save=True,tlagfunc='FFT',method='direct',rev=True,ccrsize=None):
    '''
    Create and save time lag maps between AIA images in two different wavelengths. The sign of time lags is defined by the order of the two filters such that time_lag(w1,w2)=-time_lag(w2,w1)


    w1: The first AIA filter

    w2: The second AIA filter

    pix_dir: The directory where the original AIA maps are stored.

    tlag_file: The save location to save time lag map as a FITS image

    tlag_img_file: The save location to save time lag map as a image (for eg as a png)

    save: If True, saves the file to a given location

    '''
    Mseq335_p,Mseq211_p = get_maps_for_pair_sunpy(pix_dir,w1,w2)

    timelist = time_int_check(Mseq335_p,Mseq211_p)
    '''Extract numpy arrays from map sequences'''
    arr211 =Mseq211_p.as_array()
    arr335 =Mseq335_p.as_array()
    
    '''2D arrays for time lags obtained my this function. Lag bounds of 3600 seconds important for two hour time intervals to avoid edge effects'''
    if tlagfunc=='FFT':
        tlag = time_lag(arr335.T,arr211.T,timelist*u.s,lag_bounds=(-3600*u.s,3600*u.s))
    elif tlagfunc=='slow':
        tlag,cc,lags = timelag_slow(arr335.T,arr211.T,timelist*u.s,lag_bounds=(-3600*u.s,3600*u.s),method=method,rev=rev,ccrsize=ccrsize)

    hdu = fits.PrimaryHDU(tlag)
    if save:
        hdu.writeto(tlag_file,overwrite=True)

    '''Colourmap for time lag map similar to the one used by Viall and Klimchuk'''
    c_file = np.loadtxt(r"E:\AIA Data\Coronal Loop\Extra\IDLcolortable4.txt")
    c_file=c_file/255
    cmap = ListedColormap(c_file)

    plt.figure(dpi=300)
    plt.imshow(np.array(hdu.data).T,cmap=cmap,origin='lower')
    plt.colorbar()
    if save:
        plt.savefig(tlag_img_file)
    else:
        plt.show()
        return tlag,cc,lags
    
def save_max_ccr(w1,w2,pix_dir,tlag_file=None,tlag_img_file=None,save=True,tlagfunc='FFT',ccrsize=None):
    '''
    Create and save max cross correlation maps between AIA images in two different wavelengths. The sign of time lags is defined by the order of the two filters such that time_lag(w1,w2)=-time_lag(w2,w1)


    w1: The first AIA filter
    w2: The second AIA filter
    pix_dir: The directory where the original AIA maps are stored.
    tlag_file: The save location to save max cross correlation map as a FITS image
    tlag_img_file: The save location to save max cross correlation map as a image (for eg as a png)
    save: If True, saves the file to a given location

    '''
    Mseq335_p,Mseq211_p = get_maps_for_pair_sunpy(pix_dir,w1,w2)

    timelist = time_int_check(Mseq335_p,Mseq211_p)
    '''Extract numpy arrays from map sequences'''
    arr211 =Mseq211_p.as_array()
    arr335 =Mseq335_p.as_array()
    
    '''2D arrays for time lags obtained my this function. Lag bounds of 3600 seconds important for two hour time intervals to avoid edge effects'''

    lags=get_lags(timelist*u.s)
    start, stop = _get_bounds_indices(lags,(-3600*u.s,3600*u.s))
    if tlagfunc=='FFT':
        ccr=cross_correlation(arr335.T,arr211.T,get_lags(timelist*u.s) ) 
    elif tlagfunc=='slow':
        ccr = ccr_slow(arr335.T,arr211.T,lags=lags,ccrsize=ccrsize,method='c_corr_pro',st=start,sp=stop)
    t3600=np.argwhere((np.abs(lags)==3600*u.s))
    if ccrsize=='const':
        ccrmap = np.max(ccr,axis=0)
    else:
        ccrmap = np.max(ccr[t3600[0][0]:t3600[1][0],:,:],axis=0)

    hdu = fits.PrimaryHDU(ccrmap)
    if save:
        hdu.writeto(tlag_file,overwrite=True)

    '''Colourmap for time lag map similar to the one used by Viall and Klimchuk'''
    c_file = np.loadtxt(r"E:\AIA Data\Coronal Loop\Extra\IDLcolortable4.txt")
    c_file=c_file/255
    cmap = ListedColormap(c_file)

    plt.figure(dpi=300)
    plt.imshow(np.array(hdu.data).T,cmap=cmap,origin='lower')
    plt.colorbar()
    if save:
        plt.savefig(tlag_img_file)
        return hdu,ccr,lags
    else:
        plt.show()
        return hdu,ccr,lags