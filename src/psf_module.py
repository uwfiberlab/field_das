import numpy as np

# from tqdm import tqdm
# from time import perf_counter
import os as os
import datetime
import h5py
import glob
from scipy.signal import butter, filtfilt, detrend
# from numpy.fft import fftshift, fft2, fftfreq
from datetime import datetime as DT
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime
from obspy.core.event import Catalog
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees, degrees2kilometers


def get_tstamp(fname):
    datestr = fname.split('_')[1].split('-')
    y = int(datestr[0])
    m = int(datestr[1])
    d = int(datestr[2])
    timestr = fname.split('_')[2].split('.')
    H = int(timestr[0])
    M = int(timestr[1])
    S = int(timestr[2])
    return UTCDateTime('%04d-%02d-%02dT%02d:%02d:%02d' % (y,m,d,H,M,S))


def ak_catalog(t1, t2, lat0=59.441, lon0=-152.028, a=-1, b=0.65):
    '''
    In:  t1, t2: start and ending timestamps
         lat0,lon0: Reference point of DAS network
         a, b: simple GMM parameters
    Out: cat : USGS AK catalog meeting GMM threshold
         ptimes : absolute P arrival times
    '''
    events = []; ptimes = []
    # Get local catalog
    catalog = Client('IRIS').get_events(
                starttime=t1,
                endtime=t2,
                includeallorigins=True,
                includeallmagnitudes=True)
    catalog.write("example.xml", format="QUAKEML") 
    # Loop through events 
    for event in catalog:
        lon = event.origins[0]['longitude']
        lat = event.origins[0]['latitude']
        dep = event.origins[0]['depth'] * 1e-3
        mag = event.magnitudes[0]['mag']
        distdeg = locations2degrees(lat0, lon0, lat, lon)
        distkm = degrees2kilometers(distdeg)
        rad = np.sqrt(distkm**2 + dep**2)
        
        if (mag - 10**(a + b*np.log10(rad)) >= 0):
            model = TauPyModel(model='iasp91')
            arr = model.get_travel_times(
                source_depth_in_km=dep,
                distance_in_degree=distdeg)
            
            t0 = event.origins[0]['time']
            ptimes.append(t0 + arr[0].time)
            events.append(event)

    return Catalog(events=events), np.array(ptimes)


class data_visualizer:
    def __init__(self, fpath, fac):
        with h5py.File(fpath, 'r') as fp:
            self.fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']//fac
            self.nx = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci']
            self.ns = len(fp['Acquisition']['Raw[0]']['RawDataTime'][:])//fac
            self.data = fp['Acquisition']['Raw[0]']['RawData'][::fac, :]
            self.dx = fp['Acquisition'].attrs['SpatialSamplingInterval']
        self.data -= np.tile(np.mean(self.data, axis=0),(self.ns, 1))
        self.filt = self.data.copy()
        return

    def filter_data(self, fmin, fmax):
        b, a = butter(4, (fmin, fmax), fs=self.fs, btype='bandpass')
        self.filt = filtfilt(b, a, self.data, axis=0)
        return

    def plot_data(self,xlims=[0,312],ylims=[60,0],clims=[-1,1],srcx=0):
        plt.figure(figsize=(10,6)); v = 0.8
        plt.imshow(self.filt,aspect='auto',cmap='seismic',\
                   extent=[0,self.nx*self.dx/1E3,self.ns/self.fs,0],vmin=clims[0],vmax=clims[1])
        plt.xlabel('offset (km)')
        plt.ylabel('Time (s)')
        plt.ylim(ylims)
        plt.xlim(xlims)
        plt.axvline(srcx,c='y',linewidth=3)
        return


def noise_PSD(fdir, fname, nx, ns, clims=[-1, 1]):
    # Load data and calculate spectrum for each file
    data = np.zeros((ns,nx))
    with h5py.File(os.path.join(fdir,fname),'r') as fp:
        data[:,:] = detrend(fp['Acquisition']['Raw[0]']['RawData'][:,:])
        fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']
        dx = fp['Acquisition'].attrs['SpatialSamplingInterval']
    crap=np.tile(np.hamming(ns),(nx,1))
    data *= crap.T
    freq = np.fft.rfftfreq(ns,d=1./fs)
    spec =np.log10(np.abs(np.fft.rfft(data,axis=0)))
    # Generate noise PDF
    ybins = np.logspace(np.log10(1/60),np.log10(fs/2),100)
    xbins = np.linspace(0,nx*dx/1E3,nx)
    plt.figure(figsize=(10,6)); v = 0.8
    plt.imshow(spec,aspect='auto',cmap='seismic',\
                extent=[0,nx*dx/1E3,ybins[-1],ybins[0]],vmin=clims[0],vmax=clims[1])
    plt.xlabel('offset (km)')
    plt.ylabel('frequency (Hz)')
    plt.colorbar(label='dB')
    plt.yscale('log')
    plt.show()
    return spec,freq,xbins,ybins


def noise_PDF(fdir, flist_, ns, chid, fmin, fmax):
    # Load data and calculate spectrum for each file
    data = np.zeros((len(flist_),ns))
    for ii, fname in enumerate(flist_):
        with h5py.File(os.path.join(fdir,fname),'r') as fp:
            data[ii,:] = detrend(fp['Acquisition']['Raw[0]']['RawData'][:,chid])
            fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']
    freq = np.tile(np.fft.rfftfreq(ns,d=1./fs),(len(flist_),1)).flatten()
    spec = (20 * np.log10(abs(np.fft.rfft(data,axis=1)))).flatten()
    # Generate noise PDF
    print(spec.shape)
    xbins = np.logspace(np.log10(fmin),np.log10(fmax),100)
    ybins = np.linspace(-20,120,140)
    H,xe,ye = np.histogram2d(freq,spec,bins=(xbins,ybins))
    for ix in range(len(xbins)-1):
        H[ix,:] /= sum(H[ix,:]) 
    xm = (xe[1:] + xe[:-1])/2
    ym = (ye[1:] + ye[:-1])/2
    return H,xm,ym


def noise_stats(H, xm, ym):
    nx = len(xm)
    mn = np.zeros(nx)
    vr = mn.copy()
    for ix in range(nx):
        mn[ix] = np.average(ym,weights=H[ix,:])
        vr[ix] = np.average((ym-mn[ix])**2,weights=H[ix,:])
    return xm,mn,vr


class simple_xc:

    def __init__(self, fdir, flist):
        self.fdir = fdir
        self.flist = [os.path.join(self.fdir, fname) for fname in flist]
        self.nf = len(self.flist)
        return

    def set_parameters(self, pdict):
        self.srcx = pdict['srcx']
        self.recmin = pdict['recmin']
        self.recmax = pdict['recmax']
        self.nns = pdict['nns']
        self.fmin = pdict['fmin']
        self.fmax = pdict['fmax']
        self.whiten = pdict['whiten']
        self.onebit = pdict['onebit']
        print('file at work', self.fdir, self.flist[0])

        with h5py.File(os.path.join(self.fdir, self.flist[0]), 'r') as fp:
            self.dx = fp['Acquisition'].attrs['SpatialSamplingInterval']
            self.fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']
            self.nx = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci']
            self.ns = len(fp['Acquisition']['Raw[0]']['RawDataTime'][:])
        x = np.arange(self.nx)*self.dx
        r1 = int(np.argmin(abs(x-self.recmin*self.dx)))
        r2 = int(np.argmin(abs(x-self.recmax*self.dx)))
        self.recid = np.arange(r1, r2+1)
        self.srcid = int(np.argmin(abs(x-self.srcx*self.dx))) - r1
        self.nc = len(self.recid)
        self.nw = self.nns//2 + 1
        self.nwin = int(self.ns//self.nns)
        self.spxc = np.zeros((self.nc, self.nw), dtype=np.complex_)
        self.lags = np.arange(-self.nns//2, self.nns//2)/self.fs
        self.offset = (self.recid - min(self.recid) - self.srcid) * self.dx
        return


    def preprocess_tr(self):
        '''
        this function will 1) detrend the data , 2) taper,
        3) bandpass filter, 4) fft
        '''
        self.tr = detrend(self.tr, axis=1)
        self.tr *= np.tile(np.hamming(self.nns), (self.nc, 1))
        b, a = butter(8, (self.fmin/(self.fs/2), self.fmax/(self.fs/2)),'bandpass')
        self.tr = filtfilt(b, a, self.tr, axis=1)
        self.sp = np.fft.rfft(self.tr, axis=1)
        return


    def whiten_tr(self):
        '''
        this function will whiten the data by wiping out the amplitude spectrum
        '''
        i1 = int(np.ceil(self.fmin/(self.fs/self.nns)))
        i2 = int(np.ceil(self.fmax/(self.fs/self.nns)))
        self.sp[:, i1:i2] = np.exp(1j*np.angle(self.sp[:,i1:i2]))
        self.sp[:, :i1] = np.cos(np.linspace(np.pi/2, np.pi, i1))**2 * \
                                 np.exp(1j*np.angle(self.sp[:,:i1]))
        self.sp[:, i2:] = np.cos(np.linspace(np.pi,np.pi/2,self.nw-i2))**2 *\
                                 np.exp(1j*np.angle(self.sp[:,i2:]))
        return


    def onebit_tr(self):
        '''
        the function takes the sighn of the time series after it has been FFTed.
        '''
        self.tr = np.fft.irfft(self.sp, axis=1)
        self.tr = np.sign(self.tr)
        self.sp = np.fft.rfft(self.tr, axis=1)
        return


    def process_file(self, fname):
        '''
        This function readss the data from a selected number of channels (recid)
        It then preprocess , whiten and onebit the data.
        it cross correlate the entire array with the source channel (srcid)
        '''
        with h5py.File(fname, 'r') as fp:
            self.data = fp['Acquisition']['Raw[0]']['RawData'][:,self.recid].T
        for iwin in range(self.nwin):
            self.tr = self.data[:, iwin*self.nns:(iwin+1)*self.nns]
            self.preprocess_tr()
            if self.whiten:
                self.whiten_tr()
            if self.onebit:
                self.onebit_tr()
            self.spxc += self.sp * np.tile(np.conj(self.sp[self.srcid,:]),(self.nc,1))
        return


    def compute_xc(self,pdict):
        self.set_parameters(pdict)
        for ii, fname in enumerate(self.flist):
            print('File %d/%d' % (ii+1,self.nf))
            self.process_file(fname)
        print(self.nf , self.nwin)
        self.spxc /= self.nf * self.nwin
        self.spxc -= np.tile(np.mean(self.spxc, axis=0), (self.nc, 1))
        self.trxc = np.fft.irfft(self.spxc, axis=1)
        self.trxc = np.concatenate((self.trxc[:,self.nns//2:],self.trxc[:,:self.nns//2]),axis=1)
        b, a = butter(8, (self.fmin/(self.fs/2),self.fmax/(self.fs/2)),'bandpass')
        self.trxc = filtfilt(b, a, self.trxc, axis=1)
        return


def get_tstamp(fname):
    datestr = fname.split('_')[1].split('-')
    y = int(datestr[0])
    m = int(datestr[1])
    d = int(datestr[2])
    timestr = fname.split('_')[2].split('.')
    H = int(timestr[0])
    M = int(timestr[1])
    S = int(timestr[2])
    return UTCDateTime('%04d-%02d-%02dT%02d:%02d:%02d' % (y,m,d,H,M,S))


def dt_to_utc_format(t):
    from obspy import UTCDateTime

    return UTCDateTime(t.strftime("%Y-%m-%dT%H:%M:%S"))


def utc_to_dt_format(t):
    dt_str = t.strftime("%Y/%m/%d %H:%M:%S")
    format1 = "%Y/%m/%d %H:%M:%S"
    dt_utc = DT.strptime(dt_str, format1)
    return dt_utc


def addday_lambda(x):
    return datetime.timedelta(days=x)


def sintela_to_datetime(sintela_times):
    """
    returns an array of datetime.datetime
    """
    # Convert sintela time to datetime
    days1970 = datetime.date(1970, 1, 1).toordinal()

    # Vectorize everything
    converttime = np.vectorize(datetime.datetime.fromordinal)
    adddays = np.vectorize(addday_lambda)

    day = days1970 + sintela_times / 1e6 / 60 / 60 / 24
    thisDateTime = converttime(np.floor(day).astype(int))
    dayFraction = day - np.floor(day)
    thisDateTime = thisDateTime + adddays(dayFraction)

    return thisDateTime


def open_sintela_file(
    file_base_name,
    t0,
    pth,
    chan_min=0,
    chan_max=-1,
    number_of_files=1,
    verbose=False,
    pad=False,
):
    data = np.array([])
    time = np.array([])

    dt = datetime.timedelta(minutes=1)  # Assume one minute file duration
    this_files_date = t0

    for i in range(number_of_files):
        # Construct the "date string" part of the filename
        date_str = this_files_date.strftime("%Y-%m-%d_%H-%M")

        # Construct the PARTIAL file name (path and name, but no second or filenumber):
        #         this_file = f'{pth}{file_base_name}_{date_str}_UTC_{file_number:06}.h5'
        partial_file_name = f"{pth}{file_base_name}_{date_str}"
        file_search = glob.glob(f"{partial_file_name}*h5")
        if verbose:
            print(f"Searching for files matching: {partial_file_name}*h5")
        if len(file_search) > 1:
            raise ValueError(
                "Why are there more than one files? That shouldn't be possible!"
            )
        elif len(file_search) == 0:
            raise ValueError("Why are there ZERO files? That shouldn't be possible!")
        else:
            this_file = file_search[0]

        try:
            f = h5py.File(this_file, "r")
            this_data = np.array(f["Acquisition/Raw[0]/RawData"][:, chan_min:chan_max])
            this_time = np.array(f["Acquisition/Raw[0]/RawDataTime"])

            if i == 0:
                time = sintela_to_datetime(this_time)
                data = this_data
                attrs = dict(f["Acquisition"].attrs)
            else:
                data = np.concatenate((data, this_data))
                time = np.concatenate((time, this_time))

        except Exception as e:
            print("File problem with: %s" % this_file)
            print(e)

            # There's probably a better way to handle this...
            #             return [-1], [-1], [-1]

        this_files_date = this_files_date + dt

    # if pad==True:
    # Add columns of zeros to give data matrix the correct dimensions

    return data, time, attrs


def local_earthquake_quicklook(
    dates,
    datafilt,
    st,
    st2,
    x_max,
    stitle,
    filename=None,
    skip_seismograms=False,
    das_vmax=0.1,
    network_name="",
):
    """
    Make a nice plot of the DAS data and some local seismic stations
    """
    dx = x_max / datafilt.shape[1]
    fig, ax = plt.subplots(figsize=(8, 12))
    date_format = mdates.DateFormatter("%H:%M:%S")

    # Subplot: DAS Data
    ax = plt.subplot(4, 1, 1)
    ax.set_title(f"{network_name}")
    # plt.imshow(datafilt.T,vmin=-0.1,vmax=0.1,cmap='seismic',aspect='auto')
    x_lims = mdates.date2num(dates)
    plt.imshow(
        datafilt.T,
        vmin=-das_vmax,
        vmax=das_vmax,
        cmap="seismic",
        aspect="auto",
        extent=[x_lims[0], x_lims[-1], x_max, 0],
    )
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis_date()
    plt.grid()

    # Subplot: Single DAS Channel
    ax = plt.subplot(4, 1, 2)
    fig.patch.set_facecolor("w")
    #     graph_spacing = -400
    graph_spacing = -20
    for jj in (41, 400, 800, 1400):
        plt.plot(
            dates, datafilt[:, jj] - jj / graph_spacing, label=f"OD = {int(jj*dx)} m"
        )
    plt.legend(loc="upper right")
    ax.set_title(f"{network_name} Individual Channels")
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis_date()
    ax.autoscale(enable=True, axis="x", tight=True)
    plt.grid()

    if not skip_seismograms:
        # Subplot:  station 1
        ax = plt.subplot(4, 1, 3)
        for tr in st:
            times_from_das = np.linspace(x_lims[0], x_lims[-1], len(tr.data))
            plt.plot(times_from_das, tr.data)
        fig.patch.set_facecolor("w")
        ax.set_title("UW NOWS HNN")
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis_date()
        ax.set_xlim((min(times_from_das), max(times_from_das)))
        plt.grid()

        # Subplot:  station 2
        ax = plt.subplot(4, 1, 4)
        for tr in st2:
            times_from_das = np.linspace(x_lims[0], x_lims[-1], len(tr.data))
            plt.plot(times_from_das, tr.data)
        fig.patch.set_facecolor("w")
        ax.set_title("IU COR BH1")
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis_date()
        ax.set_xlim((min(times_from_das), max(times_from_das)))
        plt.grid()

    fig.suptitle(stitle, fontsize=20)
    plt.tight_layout()

    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def data_quicklook(
    dates,
    datafilt,
    x_max,
    stitle,
    filename=None,
    das_vmax=0.1,
    network_name="",
    ylim=None,
):
    """
    Make a nice plot of DAS data
    """
    # dx = x_max / datafilt.shape[1]
    fig, ax = plt.subplots(figsize=(10, 10))
    date_format = mdates.DateFormatter("%H:%M:%S")

    # Subplot: DAS Data

    ax.set_title(f"{network_name}")
    # plt.imshow(datafilt.T,vmin=-0.1,vmax=0.1,cmap='seismic',aspect='auto')
    x_lims = mdates.date2num(dates)
    plt.imshow(
        datafilt.T,
        vmin=-das_vmax,
        vmax=das_vmax,
        cmap="seismic",
        aspect="auto",
        extent=[x_lims[0], x_lims[-1], x_max, 0],
    )
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis_date()
    plt.grid()
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.suptitle(stitle, fontsize=20)
    plt.tight_layout()

    if not filename:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


# def fk_analysis(
#     t0,
#     draw_figure=True,
#     downsamplefactor=5,
#     cable="whidbey",
#     record_length=1,
#     channel_range=[1225, 1600],
# ):
#     """
#     This function takes inputs that describe a subset of a DAS deployment and returns FK data.

#     The default channel range represents the subsea part of the whidbey cable

#     TODO the definition of f assumes d=0.01, i.e., 100 Hz data.  The sampling rate should be read from
#     attrs instead.
#     """

#     prefix, network_name, datastore = data_wrangler(cable, record_length, t0)
#     try:
#         data, dates, attrs = open_sintela_file(
#             prefix, t0, datastore, number_of_files=record_length, verbose=True
#         )
#     except:
#         print("error'ed out")
#         return [np.nan], [np.nan], [np.nan]

#     x1 = channel_range[0]
#     x2 = channel_range[1]

#     subsea_data = detrend(data[:, x1:x2])
#     downsampled_subsea_data = subsea_data[::downsamplefactor, :]

#     ft = fftshift(fft2(downsampled_subsea_data))
#     f = fftshift(
#         fftfreq(
#             downsampled_subsea_data.shape[0],
#             d=1 / (2 * attrs["MaximumFrequency"]) * downsamplefactor,
#         )
#     )
#     k = fftshift(
#         fftfreq(downsampled_subsea_data.shape[1], d=attrs["SpatialSamplingInterval"])
#     )

#     return ft, f, k
