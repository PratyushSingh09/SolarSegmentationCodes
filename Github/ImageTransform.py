import numpy as np

from scipy.ndimage import gaussian_filter
import sunpy
from sunpy.map import Map
from astropy.io import fits
from datetime import datetime

def EM_normalize_old(DEM,dlgT):
    '''I forgot what heppened here'''
    demXdlgT = DEM*dlgT
    # print(demXdlgT)
    # print(demXdlgT.sum(axis=0))
    DEM = np.array(DEM,dtype=np.float64)
    demXdlgT = np.array(demXdlgT,dtype=np.float64)
    # return DEM/(demXdlgT.sum(axis=0))
    rdem=[]
    for d in DEM:
        rdem.append(d/np.max(d))
    # return np.array(rdem,dtype=np.float64)
    return DEM  

def EM_normalize(DEM,dlgT):
    '''Normalise the pixel DEMs with the EM at those pixels
    
    DEM: the pixel wise DEMs in n temperature bins with shape (n,x,y)
    dlgT: The change in logT parameter with each bin
    '''
    demXdlgT = DEM*dlgT
    # print(demXdlgT)
    # print(demXdlgT.sum(axis=0))
    DEM = np.array(DEM,dtype=np.float64)
    demXdlgT = np.array(demXdlgT,dtype=np.float64)
    return DEM/(demXdlgT.sum(axis=0))

'''__________________UNEDITED FUNCTIONS_________________________'''

def transform(y,beta=0.5,gamma=0.5,sigma=2):
    """
    Transforms the data in AIA images according the following equation: (y+gamma)^beta. Originally  made by Nathan Stein, modified  by me (Pratyush Singh)

    Arguments
    ---------
    
    y: Image data as 3D array. The first two dimensions are the spatial dimensions, and the third dimension is filter/band/wavelength.  
    n_clusters: The number of clusters into which the image will be segmented.
    beta: Power parameter of dissimilarity measure.
    gamma: Pseudocount parameter of dissimilarity measure.
    sigma: Standard deviation of Gaussian filter to be applied before segmentation.

    Returns:Transformed features as used by the segmentation algorithm
    """
    npix = y.shape[0]*y.shape[1]
    yfil = np.zeros(y.shape) # filtered
    for ii in range(y.shape[2]):
        if sigma==0:
            yfil=y
        else:
            yfil[:,:,ii] = gaussian_filter(y[:,:,ii], sigma=sigma)

    yreshape = yfil.reshape((npix, yfil.shape[2]))
    yorig=y.reshape((npix, y.shape[2]))
    denom = ((yreshape+gamma)**(2.*beta)).sum(axis=1)
    denom.shape = (len(denom), 1)
    features = np.sqrt((yreshape+gamma)**(2.*beta) / denom)
    f=features.reshape(y.shape)
    return f

def transform_save(files,save_loc,beta=1,gamma=0,sigma=2,med_pixels=None,fits_index=1):
    """
    Transforms the data in AIA images at several epochs using the transform() function and saves the resulting images

    files: Pandas table of AIA images. Each row correspond to different epochs and each column correspond to one of 6 wavelengths. Can be created by the get_paths function in FileFunctions.py
    save_loc: Directory where the images created using this function are saved
    y: Image data as 3D array. The first two dimensions are the spatial dimensions, and the third dimension is filter/band/wavelength.  
    n_clusters: The number of clusters into which the image will be segmented.
    beta: Power parameter of dissimilarity measure.
    gamma: Pseudocount parameter of dissimilarity measure.
    sigma: Standard deviation of Gaussian filter to be applied before segmentation.
    med_pixels: 
    fits_index: Index of the hdu where the fits image data is stored. This is 1 for images obtained from JSOC. Images I created from other methods had this as 0
    Returns:Transformed features as used by the segmentation algorithm
    """
    for i, ro in files.iterrows():
        fits_arrays = [fits.open(fits_file) for fits_file in ro]
        image_arrays = [fits_file[fits_index].data for fits_file in fits_arrays]
        array_3d = np.dstack(image_arrays)

        tr=transform(array_3d,beta,gamma,sigma)
        if med_pixels:
            tr= tr*np.array(med_pixels)
        # return tr
        for i in range(len(ro)):
            smp= sunpy.map.Map(ro[i])

            tm=smp.fits_header['T_REC']
            wl= smp.fits_header['WAVELNTH']
            datetime_object = datetime.strptime(tm, '%Y-%m-%dT%H:%M:%S.%f')
            time = '{:02d}'.format(datetime_object.hour)+'{:02d}'.format(datetime_object.minute)+'{:02d}'.format(datetime_object.second)

            tr_map = sunpy.map.Map(tr[:,:,i],smp.fits_header)
            tr_map.save(save_loc+'/tr_'+time+'_'+str(wl)+'.fits',overwrite=True)