#!/usr/bin/env python

import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from obspy import UTCDateTime

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

def noise_PDF(fdir,flist_,ns,chid,fmin,fmax,ymin,ymax):
    # Load data and calculate spectrum for each file
    data = np.zeros((len(flist_),ns))
    for ii, fname in enumerate(flist_):
        with h5py.File(os.path.join(fdir,fname),'r') as fp:
            if len(fp['Acquisition']['Raw[0]']['RawDataTime'][:]) == ns:
                data[ii,:] = detrend(fp['Acquisition']['Raw[0]']['RawData'][:,chid])
            fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']
    freq = np.tile(np.fft.rfftfreq(ns,d=1./fs),(len(flist_),1)).flatten()
    spec = (20 * np.log10(abs(np.fft.rfft(data,axis=1)))).flatten()
    # Generate noise PDF
    xbins = np.logspace(np.log10(fmin),np.log10(fmax),100)
    ybins = np.linspace(ymin,ymax,100)
    H,xe,ye = np.histogram2d(freq,spec,bins=(xbins,ybins))
    for ix in range(len(xbins)-1):
        if sum(H[ix,:]) > 0:
            H[ix,:] /= sum(H[ix,:])
    xm = (xe[1:] + xe[:-1])/2
    ym = (ye[1:] + ye[:-1])/2
    return H,xm,ym

def noise_stats(H,xm,ym):
    nx = len(xm)
    mn = np.zeros(nx)
    vr = mn.copy()
    for ix in range(nx):
        if sum(H[ix,:])>0:
            mn[ix] = np.average(ym,weights=H[ix,:])
            vr[ix] = np.average((ym-mn[ix])**2,weights=H[ix,:])
        else:
            mn[ix] = np.nan
            vr[ix] = np.nan
    return xm,mn,vr


#fdir = '/mnt/qnap/TERRA_A_test1_125Hz'
#fdir = '/mnt/qnap/KKFL-S_A_test1_125Hz'
fdir = os.path.join('/mnt/qnap',sys.argv[1])
outpref = sys.argv[2]
flist = np.array(os.listdir(fdir))

print(fdir)

ftime = np.array([get_tstamp(fname) for fname in flist])
index = np.argsort(np.array(ftime)-ftime[0])
flist = flist[index]
ftime = ftime[index]

## Print out metadata
fname = flist[0]
with h5py.File(os.path.join(fdir,fname),'r') as fp:
    GL = fp['Acquisition'].attrs['GaugeLength']
    dx = fp['Acquisition'].attrs['SpatialSamplingInterval']
    fs = fp['Acquisition']['Raw[0]'].attrs['OutputDataRate']
    nx = fp['Acquisition']['Raw[0]'].attrs['NumberOfLoci']
    ns = len(fp['Acquisition']['Raw[0]']['RawDataTime'][:])
    #data = fp['Acquisition']['Raw[0]']['RawData'][:]
    print(fname)
    print('Gauge length (m):',GL)
    print('Channel spacing (m):',dx)
    print('Sampling rate (Hz):',fs)
    print('Num channels:',nx)
    print('Num samples:',ns)



## Get mean and variance for all channels
fac = 100
nnx = int(nx//fac)
chs = np.arange(nnx)
x = chs * dx * fac
fmin = 0.01
fmax = 100
ymin = 0
ymax = 90

flist = flist[1:]
if len(flist)>20:
    flist = flist[:20]

for ix, chid in enumerate(chs):
    if np.mod(ix+1,10)==0:
        print('\t',ix+1,nnx)
    xm,mn,vr = noise_stats(*noise_PDF(fdir,flist,ns,chid,fmin,fmax,ymin,ymax))
    if ix == 0:
        mns = np.zeros((nnx,len(mn)))
        vrs = np.zeros((nnx,len(vr)))
    mns[ix,:] = mn
    vrs[ix,:] = vr
'''
plt.figure()
plt.pcolormesh(x,xm,mns.T,cmap='jet',vmin=ymin,vmax=ymax)
plt.gca().set_yscale('log')
plt.ylim([fmin,fmax])
plt.xlim([0,max(x)])
plt.xlabel('Distance (m)')
plt.ylabel('Frequency (Hz)')
plt.title('Mean noise PSD (dB rel. strain)')
plt.colorbar()
plt.savefig('means.png')

plt.figure()
plt.pcolormesh(x,xm,vrs.T,cmap='jet',vmin=1,vmax=100)
plt.gca().set_yscale('log')
plt.ylim([fmin,fmax])
plt.xlim([0,max(x)])
plt.xlabel('Distance (m)')
plt.ylabel('Frequency (Hz)')
plt.title('Noise PSD variance (dB rel. strain)')
plt.colorbar()
plt.savefig('vars.png')
'''
# Rescale to strain rate for better visualization
f_ = np.tile(xm,(len(x),1))
mn_rate = mns + 20*np.log10(f_)
'''
plt.figure()
plt.pcolormesh(x,xm,mn_rate.T,cmap='jet',vmin=10,vmax=60)
plt.gca().set_yscale('log')
plt.ylim([fmin,fmax])
plt.xlim([0,max(x)])
plt.xlabel('Distance (m)')
plt.ylabel('Frequency (Hz)')
plt.title('Mean noise PSD (dB rel. strain rate)')
plt.colorbar()
plt.savefig('rate.png')
'''

outdir = '/mnt/disk1/ethan-scratch/psd_outputs'
np.save(os.path.join(outdir,'xm.npy'),xm)
np.save(os.path.join(outdir,'x.npy'),x)
np.save(os.path.join(outdir,outpref+'_mean.npy'),mns)
np.save(os.path.join(outdir,outpref+'_vars.npy'),vrs)
np.save(os.path.join(outdir,outpref+'_rate.npy'),mn_rate)

