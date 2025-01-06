'''COMPLETELY UNEDITED'''


import numpy as np
from datetime import datetime


import sunpy
from sunpy.map import Map

from sunkit_image.enhance import mgn
from astropy.io import fits

from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN,SpectralClustering
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label


def cell_init_20(X,n_clusters,random_state):
    '''Function for initializing initial centroids for the KMeans algorithm. Passed as the 'init' argument to the KMeans function.
    This function finds the maximum intensity in each filter for evey window of size (20,20) from an image of size (400,400). This gives
    us 400 potential centroids. From this, <n_clusters> centroids are randomly chosen. 

    This function ended up not being any more useful than the k-means++ algorithm so was not used

    
    X: The data array. Must be of size (400,400)
    n_clusters: Number of segments to be used for the segmentation
    ramdom_state: Was required by the 'init' argument for the KMeans function, not sure how to use this exactly
    '''
    X_win=np.lib.stride_tricks.sliding_window_view(X.reshape((400,400,6)),(20,20),(0,1))[::5,::5]
    X_mean=X_win.max(axis=(3,4))
    # print(random_state,'?')
    # np.random.RandomState(random_state)
    ri = np.random.randint(0,X_mean.shape[0],size=n_clusters)
    rj = np.random.randint(0,X_mean.shape[1],size=n_clusters)

    # return X_mean[ri,rj,:]
    return X_mean
    
    

def thermal_segmentation(y, n_clusters=5, beta=0.5, gamma=0.5, sigma=5,algo='KM', mask=None, n_init=5,init='k-means++'):
    """
    Segment a multi-band solar image according to its thermal properties. Originally  made by Nathan Stein, modified  by me (Pratyush Singh)

    Arguments
    ---------
    
    y : 
        Image data as 3D array. The first two dimensions are the spatial
        dimensions, and the third dimension is filter/band/wavelength.
       
    n_clusters : 
        The number of clusters into which the image will be segmented.
       
    beta :
        Power parameter of dissimilarity measure.
       
    gamma : 
        Pseudocount parameter of dissimilarity measure.
    
    sigma : 
        Standard deviation of Gaussian filter to be applied before 
        segmentation.

    algo:
        The segmentation algorithm to use
        KM: KMeans algorithm (default)
        BKM: Bisecting KMeans
        mgnKM: KMeans with mgn normalization applied before instead of gaussian smoothing. I'm not sure why I did that
        origKM: KMeans segmentation with no smoothing
    mask : 
        Image mask to be applied before segmentation.
        
    n_init : 
        Number of time the k-means algorithm will be run with different
        centroid seeds.
    init:
        The centroid initialization function to use. Default is KMeans

    Returns: Segment alloted for each data point as a 2D Segmented Image, the km object if required, transformed features used by the segmentation algorithm
    """
    npix = y.shape[0]*y.shape[1]
    yfil = np.zeros(y.shape) # filtered
    for ii in range(y.shape[2]):  #-2 to avoid coordinates
        if mask is not None:
            if sigma==0:
                yfil[:,:,ii] = y[:,:,ii]    
            else:
                yfil[:,:,ii] = gaussian_filter(y[:,:,ii], sigma=sigma) * mask
        else:
            if sigma==0:
                yfil[:,:,ii] = y[:,:,ii]    
                # print(yfil[:,:,ii])
            else:
                # print('not here!!')
                yfil[:,:,ii] = gaussian_filter(y[:,:,ii], sigma=sigma)
    y_filter = yfil[:,:,0:6]
    # y_coord = y[:,:,6:]/np.max(y[:,:,6:])
    y_filter_reshape = y_filter.reshape((npix, y_filter.shape[2]))
    # y_coord_reshape = y_coord.reshape((npix, y_coord.shape[2]))
    # print(y_filter_reshape.shape)
    # print(y_)
    yorig=y_filter.reshape((npix, y_filter.shape[2]))
    denom = ((y_filter_reshape+gamma)**(2.*beta)).sum(axis=1)
    denom.shape = (len(denom), 1)
    features = np.sqrt((y_filter_reshape+gamma)**(2.*beta) / denom)
    # print(features)
    features=np.nan_to_num(features)
    # features=np.concatenate((features,y_coord_reshape),axis=1)
    assert (algo=='KM' or algo=='BKM' or algo=='mgnKM' or algo=='origKM')
    if algo=="KM":
        km = KMeans(n_clusters=n_clusters, n_init=n_init,init=init)    
        km.fit(features);

    # km = AffinityPropagation()    
    elif algo=='BKM':
        km = BisectingKMeans(n_clusters=n_clusters, n_init=n_init,init='k-means++')    
        km.fit(features);
    elif algo=='mgnKM':
        km = KMeans(n_clusters=n_clusters, n_init=n_init)    
        y_mgn = [mgn(yo,k=5) for yo in np.array(y,dtype=np.float32)]
        y_mgn_f=np.array(y_mgn).reshape((npix, y.shape[2]))
        print('mgn')
        km.fit(y_mgn_f);
    elif algo=='origKM':
        km = KMeans(n_clusters=n_clusters, n_init=n_init)    
        km.fit(yorig)
    # km.fit(features);
    # kmcentre = km.cluster_centers_
    # print("test")
    # print(kmcentre)

    den=yorig.sum(axis=1)
    den.shape=(len(den),1)

    # yorig = np.concatenate((yorig/den,y_coord_reshape),axis=1)

    return km.labels_.reshape((y.shape[0], y.shape[1])), km,features
    

def km_func(cal_list=None,n_clusters=32,beta=2,gamma=0.5,sigma=5,algo='KM',mask=None,hdu_index=1,init='k-means++',n_init=5,crop=None):
    '''Wrapper function to obtain data from 6 different AIA filters from their respective fits image paths and pass that to thermal_segmentation function
    
     n_clusters: The number of clusters into which the image will be segmented.
       
    beta: Power parameter of dissimilarity measure.
       
    gamma: Pseudocount parameter of dissimilarity measure.
    
    sigma: Standard deviation of Gaussian filter to be applied before segmentation.

    algo: The segmentation algorithm to use
        KM: KMeans algorithm (default)
        BKM: Bisecting KMeans
        mgnKM: KMeans with mgn normalization applied before instead of gaussian smoothing. I'm not sure why I did that
        origKM: KMeans segmentation with no smoothing
    mask: Image mask to be applied before segmentation.

    hdu_index: Index of the hdu where the fits image data is stored. This is 1 for images obtained from JSOC. Images I created from other methods had this as 0
        
    cal_list: List of 6 image paths used for segmentation. 

    n_init: Number of time the k-means algorithm will be run with different centroid seeds.
    init: The centroid initialization function to use. Default is KMeans

    Returns: Segment alloted for each data point as a 2D Segmented Image, the km object if required, transformed features used by the segmentation algorithm

    crop: Crop the given set of images. Is a tuple of (bottom left coordinate, top right coordinate)
    '''
    stack_list = [i for i in cal_list]
        # sun_compr = .resample([1024,1024]*u.pixel)
        
    # array size check
    # print([fits.open(fits_file)[0].shape for fits_file in stack_list])

    fits_arrays = [sunpy.map.Map(fits_file) for fits_file in stack_list]
    if crop is not None:
        fits_arrays = [fits_file.submap(crop[0], top_right=crop[1]) for fits_file in fits_arrays]
    image_arrays = [fits_file.data for fits_file in fits_arrays]
    array_3d = np.dstack(image_arrays)
    # return array_3d
    # imgy = np.tile(np.arange(0,array_3d.shape[0]),array_3d.shape[1]).reshape((array_3d.shape[1],array_3d.shape[0])).T
    # imgx = np.tile(np.arange(0,array_3d.shape[1]),array_3d.shape[0]).reshape((array_3d.shape[0],array_3d.shape[1]))


    # image_arrays.append(imgx)
    # image_arrays.append(imgy)
    # array_3d = np.dstack(image_arrays)

    # # call Nathan's code
    # n_init = 5
    result,km,tr_features = thermal_segmentation(array_3d, n_clusters=n_clusters, beta=beta, gamma=gamma, sigma=sigma,algo=algo, mask=mask, n_init=n_init,init=init)
    # hdu = fits.PrimaryHDU(result)
    # hdu.writeto('E:/AIA Data/Coronal Loop/Calibrated/Cropped/Images/N'+str(n_clusters)+'b'+str(beta)+'g'+str(gamma)+'s'+str(sigma)+'.fits',overwrite=True)
    return result,km, tr_features

def split_cluster(img,Ncl,eps=0.4,min_samples=25):
    '''Attempts to split a segment with multiple disjoint regions into multiple subsegments using DBSCAN. Not very useful as it is dependent on parameters eps and min_samples. Use split_cluster_l instead
    
    
    img: Segmented image to split
    Ncl: No of clusters
    eps: the eps parameter for DBSCAN
    min_samples: the min_samples for DBSCAN
    
    returns: Attempt at an image containing the split segments
    '''
    split_img = np.empty(img.shape)
    Gbl_Clr_no=0

    for i in range(Ncl):
        m=np.ma.array(img,mask=img!=i)
        m=m+1
        nz=np.nonzero(m)
        nz = np.array(nz)        
        X = StandardScaler().fit_transform(nz.T)#This did something I guess?
        
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        for j in range(n_clusters_):
            inds=nz.T[db.labels_==j]
            # print('j',j)
            inds_flat=inds[:,0]*img.shape[1]+inds[:,1]
            np.put(split_img,inds_flat,Gbl_Clr_no)
            Gbl_Clr_no+=1
            
    return split_img

def split_cluster_l(img,Ncl):
    '''Splits a segment with multiple disjoint regions into multiple subsegments using the label function from the numpy library
    
    
    img: Segmented image to split
    Ncl: No of clusters
    
    returns: An image containing the split segments
    '''
    img=img+1
    split_img = np.empty(img.shape)
    Gbl_Clr_no= np.zeros(img.shape,dtype = img.dtype)

    for i in range(0,Ncl+1):
        limg,lnum = label(img==i,structure=np.ones((3,3)))
        limg[limg>0]=limg[limg>0]+ Gbl_Clr_no[limg>0]

        split_img+=limg
        Gbl_Clr_no+=lnum

    return split_img

def create_segments_in_dir(img_paths,n_cluster=128,beta=0.5,gamma=0.5,sigma=2,crop = None,mask=None,dir=None,suff='seg'):
    '''From a table of AIA images at various epochs, create and save thermal segments for each epoch.
    
    img_paths: Pandas table of AIA images. Each row correspond to different epochs and each column correspond to one of 6 wavelengths. Can be created by the get_paths function in FileFunctions.py
    n_clusters: The number of clusters into which the image will be segmented.
    beta: Power parameter of dissimilarity measure.
    gamma: Pseudocount parameter of dissimilarity measure.
    sigma: Standard deviation of Gaussian filter to be applied before segmentation.
    mask: Image mask to be applied before segmentation.
    crop: Crop the given set of images. Is a tuple of (bottom left coordinate, top right coordinate)
    dir: Directory where the segmented images will be saved
    suff: To add a suffix to file name
    '''
    imgs_for_seg_list=[]
    for i in range(len(img_paths)):
        imgs_for_seg_list.append(img_paths.iloc[i].values.flatten().tolist())   

    imgs_for_seg_list=np.array(imgs_for_seg_list)

    files=[]

    for img_list in imgs_for_seg_list:
        res,_,_,fits_array=km_func(img_list,n_clusters=n_cluster,beta=beta,gamma=gamma,sigma=sigma,crop=crop,mask=mask)
        hdu = fits.PrimaryHDU(res)
        tm=fits_array[0].fits_header['T_REC']
        datetime_object = datetime.strptime(tm, '%Y-%m-%dT%H:%M:%S.%f')
        time = '{:02d}'.format(datetime_object.hour)+'{:02d}'.format(datetime_object.minute)+'{:02d}'.format(datetime_object.second)
        clstr = '_n'+str(n_cluster)+'b'+str(beta)+'g'+str(gamma)+'s'+str(sigma)
        hdu.writeto(dir+suff+time+clstr+'.fits',overwrite=True)
        files.append(dir+suff+time+clstr+'.fits')

    return files
def split_files(files,NCl,hdu_index,directory,flin1=-33,flin2=-5):
    '''For a given list of segmented images, splits the segments in each image and saves it
    files: List of paths of segmented images
    NCl: Number of segments in segmented images
    hdu_index: Index of the hdu where the fits image data is stored. This is 1 for images obtained from JSOC. Images I created from other methods had this as 0
    directory: Directory where the split segmented images will be saved
    flin1: I have generated new file names for split images from file paths of existing segmented images. flin1 is the index where the segmented image name starts 
    flin2: I have generated new file names for split images from file paths of existing segmented images. flin2 is the index where the segmented image name ends
    '''
    for file in files:
        ft= fits.open(file)
        res = split_cluster_l(ft[hdu_index].data,NCl)
        hdu = fits.PrimaryHDU(res)
        hdu.writeto(directory+file[flin1:flin2]+'spl.fits',overwrite=True)