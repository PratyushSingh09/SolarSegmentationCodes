from scipy.ndimage import prewitt
import numpy as np
from ImageTransform import EM_normalize
import mode
import hipd_interval

import sunpy
from astropy.io import fits
from aiapy.calibrate import normalize_exposure

from datetime import datetime

import seaborn as sns   
import matplotlib.pyplot as plt

def get_boundaries_for_cluster(data,lab):
    '''For a given segment value, define the boundaries of that segment
        data: the segmented image
        lab: segment value 

        Returns boundary coordinates
    '''
    f=prewitt(np.ma.array(data,mask=data!=lab).mask)
    return np.argwhere(f>0)    


def all_clusters():
    '''This is pointless'''
    neighbours=set()

def get_neighbour_clusters(data,pt_arr):
    '''For a given segment boundary, return the neighbouring segments
    data: the segmented image
    pt_arr: coordinates of the segment boundary, obtained using get_boundaries_for_cluster

    Returns a set of neighbouring segments
    '''
    ne_x = [0,0,1,-1,1,1,-1,-1]
    ne_y = [1,-1,0,0,1,-1,1,-1]
    neighbours=set()
    for pt in pt_arr:
        nx=ne_x+pt[0]
        ny=ne_y+pt[1]

        for i in range(len(nx)):
            if(nx[i]>=0 and ny[i]>=0 and nx[i]<data.shape[0] and ny[i]<data.shape[1]):
                neighbours.add(data[int(nx[i])][int(ny[i])])
                # print(nx[i],ny[i])

    return neighbours
    
def clus_stats(im_data,k_data,log_scale=False):
    '''Evaluates statistics for all segments using the DEMs obtained from sparse DEM inversion. Outdated function using mean and std, not used'''
    a_list=[]

    a_mean_list=[]
    a_std_list=[]
    a_err_list=[]
    a_len_list=[]

    i=1    #Only used for bookeeping
    for c in range(int(np.max(k_data)+1)):
        print(i)
        i=i+1
        # if pair is None:
        #     print('Pair is none')
        # p_iter = iter(pair)
        # print(pair)
        # if(len(pair)==1):
            # continue
        # else:
        #     a = next(p_iter)
        #     b = next(p_iter)
        a=c
        a_mask = []
        aml=[]
        asl=[]
        ael=[]
        alen=[]
        for dem in im_data:
            am = dem[k_data==a]
            a_mask.append(am)
            alen.append(len(am))
            # a_mask = np.mean(a_mask,axis=1)
            if log_scale:
                a_mean = np.nanmean(np.log(am,where=am>0))
                a_std = np.nanstd(np.log(am,where=am>0))
            else:    
                a_mean = np.nanmean(am)
                a_std = np.nanstd(am)
            a_err = a_std/np.sqrt(len(am))
            aml.append(a_mean)
            asl.append(a_std)
            ael.append(a_err)

        a_mean_list.append(aml)
        a_std_list.append(asl)
        a_err_list.append(ael)
        
        a_list.append(a_mask)
        a_len_list.append(alen)
        # b_mean = np.nanmean(b_mask)
        # b_std = np.nanstd(b_mask)

        # snr = abs(a_mean-b_mean)/np.sqrt(a_std**2+b_std**2)
        # return a_mask, b_mask
        # snr = chisquare(np.median(a_mask),np.median(b_mask))
        # snr_list.append(snr)

    return a_list,a_mean_list,a_std_list,a_err_list,a_len_list

def clus_stats_hpd(im_data,k_data,log_scale=False,hdip=None,em_norm=0,aia_data=None,aia_tr=None,dlgT=0.1):
    '''Evaluates statistics for all segments using the DEMs obtained from sparse DEM inversion. This function uses mode and hpdi
    
    im_data: Pixel wise DEMs of a active region in some n temperature bins with shape (n, x,y). Can also use sunpy maps directly. Must be the same size as k_data, aia_data and aia_tr
    k_data: The segmented image obtained from KMeans or other segmenation algorithms. Must be the same size as im_data, aia_data and aia_tr
    neighouring_cells_only: Choose to use all segments or only neighbouring segments to obtain pairs of statistics
    log_scale: Use the log scale on the DEM maps
    hdip: hdi probability
    em_norm: EM normalization method. 0 means none, 1 means log of normalized data and 2 means normalizing log data. 2 doesn't work well
    aia_data: AIA maps in 6(or any other number) filters. Must be the same size as im_data, k_data and aia_tr
    aia_tr: Transformed AIA maps in 6(or any other number) filters. Must be the same size as im_data, aia_data and k_tr

    RETURNS:
    All lists with length N
    a_list: all the points within each segment in each DEM bin
    a_mean_list: List of all mode of all points within the segment in each temp bin
    a_std_list: List of all hpdi values
    a_err_err: List of all stderr values. Not used
    a_coord_list: List of all coordinates of all the points within each segment 
    aia_dat_list: List of all aia intensities of all the points within each segment
    aia_av_list: List of average of transformed aia intensities of all the points within each segment
    a_len_list: List of number of all pixels in each segment
    '''
    a_list=[]

    a_mean_list=[]
    a_std_list=[]
    a_err_list=[]
    a_len_list=[]
    aia_dat_list=[]
    aia_av_list=[]
    a_coord_list=[]
    i=1    #Only used for bookeeping
    for c in range(int(np.max(k_data)+1)):
        a=c
        ac = np.argwhere(k_data==a)
        if aia_data is not None:
            aiad = aia_data[ac[:,0],ac[:,1],:]
        if aia_tr is not None:
            aiatr = aia_tr[ac[:,0],ac[:,1],:]
            aiaav = np.nanmean(aiatr,axis=0)
        print(i)
        i=i+1
        # if pair is None:
        #     print('Pair is none')
        # p_iter = iter(pair)
        # print(pair)
        # if(len(pair)==1):
            # continue
        # else:
        #     a = next(p_iter)
        #     b = next(p_iter)
        a_mask = []
        # a_coord=[]
        aml=[]
        asl=[]
        ael=[]
        alen=[]
        for dem in im_data:
            am = dem[k_data==a]
            # a_mask = np.mean(a_mask,axis=1)
            if log_scale:
                if em_norm==2:#DO NOT USE
                    amlg = EM_normalize(np.log10(np.array(am),dtype=np.float64,casting='unsafe'),dlgT)
                    a_mask.append(amlg)
                    #fix
                    a_mean = mode.modalpoint(amlg)
                    # mhdi = hdi(amlg,skipna=True,multimodal=False,hdi_prob=hdip)
                    mhdi = hipd_interval.hipd_interval(amlg,clev=0.68,fsample=True)#fsample = true is important
                    # print(mhdi)
                    # a_std = np.sum(mhdi[:,1]-mhdi[:,0])    
                    a_std = mhdi[1]-mhdi[0]
                else:
                    a_mask.append(np.log10(am))
                    #fix
                    a_mean = mode.modalpoint(np.log10(am))
                    # mhdi = hdi(np.log10(am),skipna=True,multimodal=False,hdi_prob=hdip)
                    mhdi = hipd_interval.hipd_interval(np.log10(am),clev=0.68,fsample=True)#fsample = true is important

                    # print(mhdi)
                    # a_std = mhdi[-1,1]-mhdi[0,0]
                    a_std = mhdi[1]-mhdi[0]

            else:    
                a_mask.append(am)
                a_mean = mode.modalpoint(am)
                # mhdi = hdi(am,skipna=True,multimodal=False,hdi_prob=hdip)
                mhdi = hipd_interval.hipd_interval(np.log10(am),clev=0.68,fsample=True)#fsample = true is important

                # a_std = mhdi[-1,1]-mhdi[0,0]
                a_std = mhdi[1]-mhdi[0]


            
            # a_coord.append(ac)
            alen.append(len(am))
            a_err = a_std/np.sqrt(len(am))
            aml.append(a_mean)
            asl.append(a_std)
            ael.append(a_err)

        a_mean_list.append(aml)
        a_std_list.append(asl)
        a_err_list.append(ael)
        
        a_list.append(np.array(a_mask))
        a_len_list.append(alen)

        a_coord_list.append(ac)
        
        if aia_data is not None:
            aia_dat_list.append(aiad)
        if aia_tr is not None:
            aia_av_list.append(aiaav)
    return a_list,a_mean_list,a_std_list,a_err_list,a_coord_list,aia_dat_list,a_len_list,aia_av_list

'''__________________UNEDITED FUNCTIONS_________________________'''

def segment_to_mean(aia_image_data:np.ndarray,km_image_data:np.ndarray,use_sunpy=False,location=None,maparr=None,j=None,norm=None):
    '''Replace segment label in each segment with its representative mean value
    
    aia_image_data: AIA Image at a particular wavelength
    km_image_data: Segmented image 
    location: Save location for this new image
    maparr: Array of all AIA image maps (I think, check). Used for fits headers although probably not required and could be implemented better
    j: Index of relevant wavelength from the array: [94,131,171,193,211,335]
    norm: If set: normalizes image with exposure time

    Returns: The image created by the function
    '''
    #np empty corrupts array data
    tarr=np.zeros(km_image_data.shape,dtype=np.float64)
    mean_img=np.array(aia_image_data,dtype=np.float64)
    for i in range(int(max(km_image_data.flatten()))+1):
        img_mask = mean_img[km_image_data==i]
        mean = np.nanmean(img_mask)
        # mean = modalpoint(img_mask)
        stddev = np.std(img_mask)
        # mean_img2 = np.ma.array(mean_img,mask=km_image_data==i,dtype=np.float64)
        # mean_img = mean_img2.filled(fill_value=np.float64(mean))
        # tarr = [mean if c else yv for c, yv in zip(km_image_data.flatten()==i,tarr)]
        # tarr=np.where(km_image_data==i,np.random.normal(mean,stddev,(400,400)),tarr)
        # tarr=np.where(km_image_data==i,mean,tarr)
        tarr[km_image_data==i]=mean

    if use_sunpy:
        x=sunpy.map.ap((tarr),maparr[j].fits_header)
        print(x.data.max())
        if location is not None:
            x.save(location,overwrite=True)
        if norm:
            print('Normalizing')
            x=normalize_exposure(x)
        return x.data        
    else:
        x = fits.PrimaryHDU(tarr)
        if location is not None:
            x.writeto(location,overwrite=True)
        if norm:
            print('Normalize exposure of DEM? Not sure what you\'re trying to do')
            return None
        return x.data
        

def segment_files_to_mean(imgs,segfiles,wavelength,maparr,crop = None,norm = False):
    '''Iterates through a list of AIA and segmented images at same epochs and for each set of images, replaces segment label in each segment with its representative mean value

    imgs: Pandas table of AIA images. Each row correspond to different epochs and each column correspond to one of 6 wavelengths. Can be created by the get_paths function in FileFunctions.py
    segfiles: List of segmented image paths to be used
    wavelength: AIA Wavelength for which the images must be created
    maparr: Array of all AIA image maps (I think, check). Used for fits headers although probably not required and could be implemented better
    crop: Crop the given set of images. Is a tuple of (bottom left coordinate, top right coordinate)
    norm: If set: normalizes image with exposure time
      
    '''
    files=segfiles
    if wavelength==94:
        wavelength_index=0
    elif wavelength==131:
        wavelength_index=1
    elif wavelength==171:
        wavelength_index=2
    elif wavelength==193:
        wavelength_index=3
    elif wavelength==211:
        wavelength_index=4
    elif wavelength==335:
        wavelength_index=5

    mean_arr=[]
    mean_img_arr=[]
    mskarr = sunpy.map.Map(imgs.to_numpy()[:,wavelength_index].tolist()[::5],sequence=True)
    files.sort()
    for i in range(len(files)):
        print(i)
        # mask_image=fits.open(imgs.iloc[i].values.flatten().tolist()[wavelength_index])
        mask_image = mskarr[i]
        if crop is not None:
            mask_image=mask_image.submap(crop[0], top_right=crop[1])
        # mask_image_data = mask_image.data

        seg_image = fits.open(files[i])
        seg_image_data = seg_image[0].data

        mean_img=segment_to_mean(mskarr[i].data,seg_image[0].data,files[i][0:-5]+'M'+str(wavelength)+'.fits',maparr,i,norm=norm)
        mean_img_arr.append(mean_img)
        mean_arr.append(np.max(mean_img))
    print(np.max(np.array(mean_arr)))
    return mean_img_arr
        # save_image(mean_img,files[i][0:-5]+'M211.fits')
def avg_segment_file_to_mean(imgs,avg_file,wavelength,maparr,crop = None):
    '''NOT USED   Iterates through a list of AIA images and for each image, replaces segment label in each segment with its representative mean value. It uses a single segmentation (like mean segmentation) for AIA images at each epoch

    imgs: Pandas table of AIA images. Each row correspond to different epochs and each column correspond to one of 6 wavelengths. Can be created by the get_paths function in FileFunctions.py
    avg_file: Segmented image paths to be used
    wavelength: AIA Wavelength for which the images must be created
    maparr: Array of all AIA image maps (I think, check). Used for fits headers although probably not required and could be implemented better
    crop: Crop the given set of images. Is a tuple of (bottom left coordinate, top right coordinate)
    norm: If set: normalizes image with exposure time
      
    '''

    if wavelength==94:
        wavelength_index=0
    elif wavelength==131:
        wavelength_index=1
    elif wavelength==171:
        wavelength_index=2
    elif wavelength==193:
        wavelength_index=3
    elif wavelength==211:
        wavelength_index=4
    elif wavelength==335:
        wavelength_index=5

    mean_arr=[]
    mean_img_arr=[]
    seg_image = fits.open(avg_file)
    seg_image_data = seg_image[0].data

    for i in range(len(imgs)):
        print(i)
        # mask_image=fits.open(imgs.iloc[i].values.flatten().tolist()[wavelength_index])
        mask_image = sunpy.map.Map(imgs.iloc[i].values.flatten().tolist()[wavelength_index])
        if crop is not None:
            mask_image=mask_image.submap(crop[0], top_right=crop[1])
        # mask_image_data = mask_image.data


        tm=mask_image.fits_header['T_REC']

        datetime_object = datetime.strptime(tm, '%Y-%m-%dT%H:%M:%S.%f')
        # time = str(datetime_object.hour) +str (datetime_object.minute)+str(datetime_object.second)
        time = '{:02d}'.format(datetime_object.hour)+'{:02d}'.format(datetime_object.minute)+'{:02d}'.format(datetime_object.second)


        mean_img=segment_to_mean(mask_image.data,seg_image_data,avg_file[0:-5]+time+'M'+str(wavelength)+'.fits',maparr,i)
        mean_img_arr.append(mean_img)
        mean_arr.append(np.max(mean_img))
    print(np.max(np.array(mean_arr)))
    return mean_img_arr


def segment_to_std(aia_image_data:np.ndarray,km_image_data:np.ndarray,location):
    '''Obtains and saves standard deviation value of all pixels in each segment in the given segmented image
    
    aia_image_data: AIA Image at a particular wavelength
    km_image_data: Segmented image 
    location: Save location for this new image

    Returns: The image created by the function
    '''
    #np empty corrupts array data
    tarr=np.zeros(km_image_data.shape)
    mean_img=np.array(aia_image_data,dtype=np.float64)
    # plt.imshow(mean_img)
    # plt.show()
    sucessful = False

    std_arr=[]
    # while not sucessful:
    for i in range(int(max(km_image_data.flatten()))+1):
        # success=False
        # while not success:
        img_mask = aia_image_data[km_image_data==i]
        stdval = np.nanstd(img_mask)
        std_arr.append(stdval)

    np.savetxt(location,np.array(std_arr))
    return std_arr
def std_segment_file_to_mean(imgs,avg_file,wavelength_index,wavelength,crop = None):
    '''NOT USED PERHAPS MAKE A REGULAR VERION OF THIS FUNCTION Iterates through a list of AIA images and for each image, obtains standard deviation value of all pixels in each segment in the given segmented image. It uses a single segmentation (like mean segmentation) for AIA images at each epoch

    imgs: Pandas table of AIA images. Each row correspond to different epochs and each column correspond to one of 6 wavelengths. Can be created by the get_paths function in FileFunctions.py
    avg_file: Segmented image paths to be used
    wavelength: AIA Wavelength for which the images must be created
    crop: Crop the given set of images. Is a tuple of (bottom left coordinate, top right coordinate)
      
    '''
    mean_arr=[]
    mean_img_arr=[]

    for i in range(len(imgs)):
        print(i)
        # mask_image=fits.open(imgs.iloc[i].values.flatten().tolist()[wavelength_index])
        mask_image = sunpy.map.Map(imgs.iloc[i].values.flatten().tolist()[wavelength_index])
        if crop is not None:
            mask_image=mask_image.submap(crop[0], top_right=crop[1])
        # mask_image_data = mask_image.data

        seg_image = fits.open(avg_file)
        seg_image_data = seg_image[0].data

        tm=mask_image.fits_header['T_REC']

        datetime_object = datetime.strptime(tm, '%Y-%m-%dT%H:%M:%S.%f')
        # time = str(datetime_object.hour) +str (datetime_object.minute)+str(datetime_object.second)
        time = '{:02d}'.format(datetime_object.hour)+'{:02d}'.format(datetime_object.minute)+'{:02d}'.format(datetime_object.second)


        mean_img=segment_to_std(mask_image.data,seg_image_data,avg_file[0:-5]+time+'STD'+str(wavelength)+'.txt')
        mean_img_arr.append(mean_img)
        mean_arr.append(np.max(mean_img))
    print(np.max(np.array(mean_arr)))
    return mean_img_arr
def save_array_as_fits(image,location):
    '''Save a 2D numpy array as a fits image
    
    image: 2D array to be saved
    location: save location
    '''
    hdu = fits.PrimaryHDU(image)
    hdu.writeto(location,overwrite=True)
        # save_image(mean_img,files[i][0:-5]+'M211.fits')

def DEM_Seg_chisq(clus_eval_out=None,am=None,bm=None,ash=None,bsh=None):
    if clus_eval_out:
        assert (am==None and bm==None and ash==None and bsh==None), 'Supplied more arguments than required'
        am=np.array(clus_eval_out[2])
        bm=np.array(clus_eval_out[3])
        ash=np.array(clus_eval_out[4])/2
        bsh=np.array(clus_eval_out[5])/2
    elif (am and bm and ash and bsh):
        assert clus_eval_out==None, 'Supplied more arguments than required'
    else:
        assert 0,'Insufficient Arguments'
    chipairs=np.nansum(((am-bm)**2/(ash**2+bsh**2)),axis=1,where=np.isinf((am-bm)**2/(ash**2+bsh**2))==False)
    return chipairs

def plot_chisq(chipairs):
    plt.plot((chipairs))
    plt.xlabel('Index')
    plt.ylabel('Chi Square Differences')
    plt.axhline((21*1.3), color='black')#sqrt(2)*20
    plt.show()

def plot_chisq_distr(chipairs,title='Distribution for Chi Square Differences',csq_list_list=None):
    d=0
    def forward(x):
        return x**(1/2)


    def inverse(x):
        return x**2
    if csq_list_list is not None:
        for c in csq_list_list[:-1]:
            print(d)
            d+=1
            ax=sns.kdeplot(c)
        ax=sns.kdeplot(csq_list_list[-1])

    ax=sns.kdeplot(chipairs)
    ax.set_yscale('function', functions=(forward, inverse))
    plt.xlabel('Chi Square Differences')
    # plt.legend(['Actual DEMs','Simulated DEMs'],reverse=True)
    plt.title(title)
    # plt.legend(['Simulated','Actual '])