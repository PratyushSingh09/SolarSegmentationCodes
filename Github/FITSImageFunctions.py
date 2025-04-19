
import os
import sunpy
from sunpy.map import Map
from astropy.io import fits
from aiapy.calibrate import normalize_exposure
from astropy.table import Table

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def pandas_to_Table(df_arr,tablepaths,overwrite=False):
    '''Save several dataframes as astropy tables. Very simple function I know
        df_arr: List of dataframes to be saved
        tablepaths: List of paths where fits tables will be saved to
        overwrite: Overwrite existing saved fits tables'''
    for (df,tablepath) in zip(df_arr,tablepaths):
        
        t = Table.from_pandas(df)
        t.write(tablepath,format='fits',overwrite = overwrite)


'''COMPLETELY UNEDITED'''
from ImageTransform import transform

def get_maps_for_pair_sunpy(rootdir:str,f1,f2,suf='.image.fits'):
    '''For all files within the root directory, return sunpy map sequences. Use this for images created using sunpy/ssw(I assume)/obtained from JSOC
    This is mainly used for obtaining map sequences for two filters I later obtained the time lag for
    
    rootdir: The root directory
    f1: 1st filter wavelength
    f2: 2nd filter wavelength
    suf: common suffix all the images end with, for images obtained from JSOC it is '.image.fits'
       '''
    file_list = os.listdir(rootdir)
    img_list1 = [file for file in file_list if file.endswith(str(f1)+suf)]
    img_path_list1 = [rootdir+"\\"+file for file in img_list1]
    img_path_list1.sort()
    # print('File list:'+str(f1),file_list)

    img_list2 = [file for file in file_list if file.endswith(str(f2)+suf)]
    # img_path_list = ["D:\AIA Data\Coronal Loop\\94A\\"+file for file in img_list]
    img_path_list2 = [rootdir+"\\"+file for file in img_list2]
    img_path_list2.sort()

    map1=[]
    map2=[]
    # map335=[]
    # map211=[]
    for file in img_path_list1:
        map1.append(sunpy.map.Map(file))
    for file in img_path_list2:
        map2.append(sunpy.map.Map(file))

    Mseq1 = sunpy.map.Map(map1,sequence=True)
    Mseq2 = sunpy.map.Map(map2,sequence=True)
    
    return Mseq1,Mseq2

def get_maps_for_pair_fits(rootdir:str,f1,f2,suf='.fits'):
    '''For all files within the root directory, return sunpy map sequences. Use this for images stored only as fits files. This function converts them to sunpy maps to create one sunpy sequence. 
    This is mainly used for obtaining map sequences from segmented images for two filters I later obtained the time lag for
    rootdir: The root directory
    f1: 1st filter wavelength
    f2: 2nd filter wavelength
    suf: common suffix all the images end with, for images obtained from JSOC it is '.image.fits'
    
       '''
    file_list = os.listdir(rootdir)
    img_list1 = [file for file in file_list if file.endswith('M'+str(f1)+suf)]
    img_path_list1 = [rootdir+"\\"+file for file in img_list1]
    img_path_list1.sort()

    # print('File list:'+str(f1),file_list)

    img_list2 = [file for file in file_list if file.endswith('M'+str(f2)+suf)]
    # img_path_list = ["D:\AIA Data\Coronal Loop\\94A\\"+file for file in img_list]
    img_path_list2 = [rootdir+"\\"+file for file in img_list2]
    img_path_list2.sort()

    map1=[]
    map2=[]
    # map335=[]
    # map211=[]
    for file in img_path_list1:
        data = fits.open(file)[0].data
        header = fits.open(file)[0].header
        header['cunit1'] = 'arcsec' 
        header['cunit2'] = 'arcsec' 
        map1_1 = sunpy.map.Map(data, header) 
        map1.append(map1_1)
    for file in img_path_list2:
        data = fits.open(file)[0].data
        header = fits.open(file)[0].header
        header['cunit1'] = 'arcsec' 
        header['cunit2'] = 'arcsec' 
        map1_1 = sunpy.map.Map(data, header) 
        map2.append(map1_1)

    Mseq1 = sunpy.map.Map(map1,sequence=True)
    Mseq2 = sunpy.map.Map(map2,sequence=True)
    
    return Mseq1,Mseq2

def time_int_check(Mseq1,Mseq2,verbose=False):
    '''Make sure there are no maps missing in either mapsequences and maps at the same index are not at seperate timestamps
    Mseq1: First Map sequence to be used
    Mseq2: Second Map sequence to be used
    verbose: Show more details about length of timestamps or any shift in timestamp at the same index

    Returns timestamps to be used by the time lag function'''
    timelist=[]
    for mp in Mseq1:
        datetime_str = mp.fits_header['T_REC']

        datetime_object = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%f')

        timelist.append(datetime_object.timestamp())

    timelist2=[]
    for mp in Mseq2:
        datetime_str = mp.fits_header['T_REC']

        datetime_object = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%f')

        timelist2.append(datetime_object.timestamp())
    # timelist
    if verbose:
        print("Lengths of timelists 1 and 2 resp: ",len(timelist),len(timelist2))

    #CHANGE THE INDEXES
    diff=np.array(timelist)-np.array(timelist2)
    # print("Time Difference(Should be 0): ",diff)
    if verbose:
        print("Just to confirm sum of differences is 0: ",diff.sum())
    assert len(timelist)==len(timelist2)
    assert diff.sum()==0
    return timelist

def difference_image(img_path1,img_path2,norm=False,save_loc=False):
    '''Find the difference image between two AIA images. This function finds the difference between the intensities in two images at each pixel.

    img_path1: Path to the first image
    img_path2: Path to the second image
    norm: Normalize both images by their respective maximum intensity. Do this if comparing images in different wavelengths
    save_loc: The location where the file will be saved. If False, no file is saved
    '''
    img1 = sunpy.map.Map(img_path1)
    img2 = sunpy.map.Map(img_path2)

    max131=np.unravel_index(img1.data.argmax(),img1.data.shape)
    max211=np.unravel_index(img2.data.argmax(),img2.data.shape)

    if norm:
        imgnorm1=img1.data/img1.data[max131]
        imgnorm2=img2.data/img2.data[max211]
        
        imgdiff=imgnorm1-imgnorm2
    else:
        imgdiff=img1-img2
    a=sunpy.map.Map(imgdiff,img1.fits_header)
    a.peek()
    
    if save_loc:
        a.save(save_loc)
    else:
        print('WARNING: file not saved')

def sun_view(location):
    '''Shows the AIA image and a violinplot of all the pixel intensities in the image
    location: Location of the AIA map'''
    mp = map.Map(location)
    mp.peek()
    mp_data = mp.data.flatten()
    
    plt.violinplot(mp_data)
    plt.show()

def sun_view_transformed(location,n_clusters=32,beta=1,gamma=0.3,sigma=2):
    '''Shows the AIA image and a violinplot of all the transformed pixel intensities in the image
    location: Location of the AIA map
    n_clusters: The number of clusters into which the image will be segmented.
    beta: Power parameter of dissimilarity measure.
    gamma: Pseudocount parameter of dissimilarity measure.
    sigma: Standard deviation of Gaussian filter to be applied before segmentation.
    '''
    mp = map.Map(location)

    features= transform(mp.data,n_clusters,beta,gamma,sigma)
    # # print(np.shape(transformed[:,5]))
    plt.imshow(features[:,5].reshape((mp.data.shape[0],mp.data.shape[1])))
    plt.colorbar()
    plt.show()

    plt.violinplot(features[:,5])
    plt.show()

def aia_violinplot(features):
    '''Violinplot of intensities in 6 AIA wavelengths plotted together. It is recommended to do this for transformed and normalized intensities only
    
    features: Intensities in 6 AIA wavelengths
    '''
    ft_db=pd.DataFrame(features)
    plt.violinplot(ft_db)

def aia_corrmap(features):
    '''Correlation map of intensities in 6 AIA wavelengths plotted together. It is recommended to do this for transformed and normalized intensities only
    
    features: Intensities in 6 AIA wavelengths
    '''
    ft_db=pd.DataFrame(features)
    dataplot = sns.heatmap(ft_db.corr(), cmap="YlGnBu", annot=True)

def normalize_all(img_list,save_dir=None):
    '''
    For each image in img_list, normalize exposure for each of them and save the images created. JSOC processing does not do this, so I do it here. 
    It is important to do this prior to obtaining DEMs using Cheung et al's sparse inversion code

    img_list: List of paths of all images to be normalized by exposure time.
    '''
    stack_list = [i for i in img_list]
    fits_arrays = [sunpy.map.Map(fits_file) for fits_file in stack_list]
    fits_arrays = [normalize_exposure(fits_file) for fits_file in fits_arrays]
    i=0
    for img in fits_arrays:
        img.save(filepath= stack_list[i][:-5]+'expn.fits',overwrite=True)
        i+=1
            
    return fits_arrays

def get_header(dir,wavelength,maxfiles=None,suf='.image.fits'):
    '''Get headers for all files (with a given suffix) in the given directory
    dir: The root directory
    wavelength: Choose AIA wavelength to obtain headers for (expects images named as {wavelength}.{suf})
    maxfiles: If set, obtains headers for only first {maxfiles} images
    suf: common suffix all the images end with, for images obtained from JSOC it is '.image.fits'
    '''
    file_list = os.listdir(dir)
    img_list1 = [file for file in file_list if file.endswith(str(wavelength)+suf)]
    files = [dir+"\\"+file for file in img_list1]
    files.sort()
    maparr=[]
    i=0
    # print(files)

    for img in files:
        i+=1
        maparr.append(sunpy.map.Map(img))
        if maxfiles is not None:
            if  maxfiles == i:
                break

    return maparr
