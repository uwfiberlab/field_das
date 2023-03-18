import numpy as np
from tqdm import tqdm
from time import perf_counter
import datetime
import h5py
import glob
from scipy.signal import detrend
from numpy.fft import fftshift, fft2, fftfreq
from datetime import datetime as DT
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def dt_to_utc_format(t):
    from obspy import UTCDateTime
    return UTCDateTime(t.strftime('%Y-%m-%dT%H:%M:%S'))

def utc_to_dt_format(t):
    dt_str = t.strftime('%Y/%m/%d %H:%M:%S')
    format1  = "%Y/%m/%d %H:%M:%S"
    dt_utc = DT.strptime(dt_str, format1)
    return dt_utc
    
def sintela_to_datetime(sintela_times):
    '''
    returns an array of datetime.datetime 
    ''' 
    days1970 = datetime.date(1970, 1, 1).toordinal()

    # Vectorize everything
    converttime = np.vectorize(datetime.datetime.fromordinal)
    addday_lambda = lambda x : datetime.timedelta(days=x)
    adddays = np.vectorize(addday_lambda )
    
    day = days1970 + sintela_times/1e6/60/60/24
    thisDateTime = converttime(np.floor(day).astype(int))
    dayFraction = day-np.floor(day)
    thisDateTime = thisDateTime + adddays(dayFraction)

    return thisDateTime

def open_sintela_file(file_base_name,t0,pth,
                      chan_min=0,
                      chan_max=-1,
                      number_of_files=1,
                      verbose=False,
                      pad=False):

    data = np.array([])
    time = np.array([])

    
    dt = datetime.timedelta(minutes=1) # Assume one minute file duration
    this_files_date = t0
    
    for i in range(number_of_files):
        
        # Construct the "date string" part of the filename
        date_str = this_files_date.strftime("%Y-%m-%d_%H-%M")
    
        # Construct the PARTIAL file name (path and name, but no second or filenumber):
#         this_file = f'{pth}{file_base_name}_{date_str}_UTC_{file_number:06}.h5'
        partial_file_name = f'{pth}{file_base_name}_{date_str}'
        file_search = glob.glob(f'{partial_file_name}*h5')
        if verbose:
            print(f'Searching for files matching: {partial_file_name}*h5')
        if len(file_search) > 1:
            raise ValueError("Why are there more than one files? That shouldn't be possible!")
        elif len(file_search) == 0:
            raise ValueError("Why are there ZERO files? That shouldn't be possible!")
        else:
            this_file = file_search[0]
        
        try:
            f = h5py.File(this_file,'r')
            this_data = np.array(
                f['Acquisition/Raw[0]/RawData'][:,chan_min:chan_max])
            this_time = np.array(
                f['Acquisition/Raw[0]/RawDataTime'])
            
            if i == 0:
                time = sintela_to_datetime(this_time)
                data = this_data
                attrs=dict(f['Acquisition'].attrs)
            else:
                data = np.concatenate((data, this_data ))
                time = np.concatenate((time, this_time ))
                
        except Exception as e: 
            print('File problem with: %s'%this_file)
            print(e)
            
            # There's probably a better way to handle this...
            #             return [-1], [-1], [-1]


        this_files_date = this_files_date + dt
    
    #if pad==True:
        # Add columns of zeros to give data matrix the correct dimensions
        
    return data, time, attrs

def local_earthquake_quicklook(dates,datafilt,st,st2,
                        x_max,stitle,filename=None,
                        skip_seismograms=False,
                        das_vmax=0.1,
                        network_name=''):
    '''
    Make a nice plot of the DAS data and some local seismic stations
    '''
    dx = x_max / datafilt.shape[1]
    fig,ax=plt.subplots(figsize=(8,12))
    date_format = mdates.DateFormatter('%H:%M:%S')
    
    # Subplot: DAS Data
    ax=plt.subplot(4,1,1)
    ax.set_title(f'{network_name}')
    # plt.imshow(datafilt.T,vmin=-0.1,vmax=0.1,cmap='seismic',aspect='auto')
    x_lims = mdates.date2num(dates)
    plt.imshow(datafilt.T,vmin=-das_vmax,vmax=das_vmax,
               cmap='seismic',aspect='auto', 
               extent=[x_lims[0],x_lims[-1],x_max,0])
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis_date()
    plt.grid()
    
    # Subplot: Single DAS Channel
    ax = plt.subplot(4,1,2)
    fig.patch.set_facecolor('w')
#     graph_spacing = -400
    graph_spacing = -20
    for jj in (41,400,800,1400):
        plt.plot(dates,datafilt[:,jj]-jj/graph_spacing,label=f'OD = {int(jj*dx)} m')
    plt.legend(loc='upper right')
    ax.set_title(f'{network_name} Individual Channels')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis_date()
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.grid()


    if skip_seismograms==False:
        
        # Subplot:  station 1
        ax = plt.subplot(4,1,3)
        for tr in st:
            times_from_das = np.linspace(x_lims[0],x_lims[-1],len(tr.data))
            plt.plot(times_from_das,tr.data)
        fig.patch.set_facecolor('w')
        ax.set_title('UW NOWS HNN')
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis_date()
        ax.set_xlim((min(times_from_das),max(times_from_das)))
        plt.grid()
    

        # Subplot:  station 2
        ax = plt.subplot(4,1,4)
        for tr in st2:
            times_from_das = np.linspace(x_lims[0],x_lims[-1],len(tr.data))
            plt.plot(times_from_das,tr.data)
        fig.patch.set_facecolor('w')
        ax.set_title('IU COR BH1')
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis_date()
        ax.set_xlim((min(times_from_das),max(times_from_das)))
        plt.grid()
    
    

    fig.suptitle(stitle,fontsize=20)
    plt.tight_layout()
    
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
    
    
def data_quicklook(     dates,datafilt,
                        x_max,stitle,filename=None,
                        das_vmax=0.1,
                        network_name='',
                        ylim=None):
    '''
    Make a nice plot of DAS data 
    '''
    dx = x_max / datafilt.shape[1]
    fig,ax=plt.subplots(figsize=(10,10))
    date_format = mdates.DateFormatter('%H:%M:%S')
    
    # Subplot: DAS Data

    ax.set_title(f'{network_name}')
    # plt.imshow(datafilt.T,vmin=-0.1,vmax=0.1,cmap='seismic',aspect='auto')
    x_lims = mdates.date2num(dates)
    plt.imshow(datafilt.T,vmin=-das_vmax,vmax=das_vmax,
               cmap='seismic',aspect='auto', 
               extent=[x_lims[0],x_lims[-1],x_max,0])
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis_date()
    plt.grid()
    if ylim is not None:
        ax.set_ylim(ylim)
    

    fig.suptitle(stitle,fontsize=20)
    plt.tight_layout()
    
    if filename==None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
        
def fk_analysis(t0, draw_figure = True,downsamplefactor=5,cable = 'whidbey', record_length = 1,
               channel_range=[1225,1600]):
    '''
    This function takes inputs that describe a subset of a DAS deployment and returns FK data.

    The default channel range represents the subsea part of the whidbey cable

    TODO the definition of f assumes d=0.01, i.e., 100 Hz data.  The sampling rate should be read from
    attrs instead.
    '''
    
    prefix, network_name, datastore = data_wrangler(cable,record_length,t0)
    try:
        data,dates,attrs = open_sintela_file(prefix,
                                         t0,
                                         datastore,
                                         number_of_files=record_length,
                                         verbose=True)
    except:
        print("error'ed out")
        return [np.nan], [np.nan], [np.nan]
    
    x1 = channel_range[0]
    x2 = channel_range[1]

    subsea_data = detrend(data[:,x1:x2])
    downsampled_subsea_data = subsea_data[::downsamplefactor,:]

    ft = fftshift(fft2(downsampled_subsea_data))
    f = fftshift(fftfreq(downsampled_subsea_data.shape[0], d=1/(2*attrs['MaximumFrequency']) * downsamplefactor))
    k = fftshift(fftfreq(downsampled_subsea_data.shape[1], d=attrs['SpatialSamplingInterval']))
    
    return ft,f,k

