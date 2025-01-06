'''COMPLETELY UNEDITED'''

import matplotlib.pyplot as plt
import numpy as np
from ImageTransform import EM_normalize

dlgT=0.1
lgT = np.linspace(5.7,5.7+0.1*20,21,retstep=True)
fltr=['94','131','171','193','211','335']

def set_lgT(valarr):
    '''Set the value of log T for all relevant calculations in this file

    valarr: Output of np.linspace function. For eg, starting from logT=5.7 to logT=7.7 with 20 bins (dlogT=0.1): lgT = np.linspace(5.7,5.7+0.1*20,21,retstep=True)  '''
    global lgT
    lgT=valarr
def set_dlgT(val):
    '''Set the value of d(log T) for all relevant calculations in this file

    val: Value of d log T to be used'''

    global dlgT
    dlgT=val


def plotDEM(Dpix, savedir=None,save=False,cmap='Oranges'):
    '''Plots DEM map in each temp bin

    Dpix: the DEMs in several different temperature bins. Obtained from the sparse DEM sswidl files

    savedir: Save directory for all the images

    save: True if saving the files, False otherwise

    cmap: Cmap to be used for the DEM maps'''
    for pic, lgt in zip(Dpix['SAVEGEN0'],lgT[0]):
        plt.imshow((pic),cmap=cmap,origin='lower')
        plt.title('DEM at logT= '+str(np.round(lgt,1)))
        plt.colorbar()
        if save:
            plt.savefig(savedir+'\DEM'+str(np.round(lgt,1))+'.png')
        plt.show()

def getEM(Dpix):
    '''Get the emission measure from the given DEMs in all the provided temperature ranges

    Dpix: the DEMs in several different temperature bins. Obtained from the sparse DEM sswidl files
    '''
    global dlgT
    if (type(Dpix)==np.ndarray):
        demXdlgT = Dpix*dlgT
    else:
        demXdlgT = Dpix['SAVEGEN0']*dlgT
    demXdlgT = np.array(demXdlgT,dtype=np.float64)
    return (demXdlgT.sum(axis=0))

def plotEM(Dpix, log=False,savefilename=None,save=False,cmap='Oranges'):
    '''Plot emission measure obtained from the given DEMs.

    Dpix: the DEMs in several different temperature bins. Obtained from the sparse DEM sswidl files
    
    log: Plot in log scale

    savefilename: Path where the image will be saved
    
    save: True if saving the files, False otherwise

    cmap: Cmap to be used for the DEM maps
    '''
    global lgT
    EM=getEM(Dpix)
    if log:
        plt.imshow(np.log10(EM),cmap=cmap,origin='lower')
    else:
        plt.imshow(EM,cmap=cmap,origin='lower')
    plt.title('Emission Measure from lgT = '+str(np.round(lgT[0][0],1))+' to '+str(np.round(lgT[0][-1],1)))
    plt.colorbar()
    if save:
        plt.savefig(savefilename)
    plt.show()

def get_Average_Temperature(Dpix):
    '''Get the average temperature at each pixel from the given DEMs. 
    
    Dpix: the DEMs in several different temperature bins. Obtained from the sparse DEM sswidl files
'''
    global dlgT
    demXlgTXdlgT = EM_normalize(Dpix['SAVEGEN0'],0.1).reshape((Dpix['SAVEGEN0'].shape[0],Dpix['SAVEGEN0'].shape[1]*Dpix['SAVEGEN0'].shape[2])) * dlgT*np.reshape(lgT[0],(lgT[0].shape[0],1))
    demXlgTXdlgT = np.array(demXlgTXdlgT.reshape(Dpix['SAVEGEN0'].shape),dtype=np.float64)
    EM = getEM(Dpix)
    return (demXlgTXdlgT.sum(axis=0))

def get_Average_Temperature_seg(Dpix):
    '''Get the average temperature at each segment from a map of average DEMs in each segment. 
    
    Dpix: the average DEMs in each segment in several different temperature bins. Obtained from SegmentStatistics.segment_to_mean function
    '''
    global dlgT
    demXlgTXdlgT = Dpix['SAVEGEN0'].reshape((Dpix['SAVEGEN0'].shape[0],Dpix['SAVEGEN0'].shape[1]*Dpix['SAVEGEN0'].shape[2])) * dlgT*np.reshape(lgT[0],(lgT[0].shape[0],1))
    demXlgTXdlgT = np.array(demXlgTXdlgT.reshape(Dpix['SAVEGEN0'].shape),dtype=np.float64)
    EM = getEM(Dpix)
    return (demXlgTXdlgT.sum(axis=0))

def plotAverageTemperature(Dpix, log=False,savefilename=None,save=False,cmap='Oranges'):
    '''Plot average Temperature obtained from the given DEMs.

    Dpix: the DEMs in several different temperature bins. Obtained from the sparse DEM sswidl files
    
    log: Plot in log scale

    savefilename: Path where the image will be saved
    
    save: True if saving the files, False otherwise

    cmap: Cmap to be used for the DEM maps
    '''
    global lgT
    EM=get_Average_Temperature(Dpix)
    if log:
        print('No')
        exit(1)
    else:
        plt.imshow(EM,cmap=cmap,origin='lower',vmin=5.7,vmax=7.7)
    plt.title('Average Temperature at each pixel')
    plt.colorbar()
    if save:
        plt.savefig(savefilename)
    plt.show()