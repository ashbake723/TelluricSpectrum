# Get k factor for spectrum
# opus2py documentation: file:///Users/ashbake/Documents/Research/Projects/TelluricCatalog/read_opus-1.5.0-Linux/doc/opus2py/opus2py-module.html

import astropy.io.fits as fits
import matplotlib.pylab as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import os,sys
sys.path.append('/Users/ashbake/Documents/Research/Projects/Kavli/Code/PB062816/pyratbay/modules/MCcubed/')
import MCcubed as mc3
sys.path.append('/Users/ashbake/Documents/Research/Projects/TelluricCatalog/read_opus/')
import opus2py as op

datapath = '/Users/ashbake/Documents/Research/Projects/TelluricCatalog/__DATA/'
fname = '1504616531908_20150607T102000_ulemke_Si_Quartz_003_SunI2.0057'

# Get Spectrum
spec = np.array(op.read_spectrum(fname))
meta = op.get_spectrum_metadata(fname)

# Construct wavenumber array
vFTS = np.linspace(meta.xAxisStart,meta.xAxisEnd,meta.numPoints)

# Open Tapas
tapas = fits.open(datapath + 'TAPAS/wavenumber/' + 'tapas_oxygen.fits')
vtap = tapas[1].data['wavenumber']
stap = tapas[1].data['transmittance']

# Create a mask for voiding parts don't want to fit
# fit three o2 regions
mask = np.zeros(len(spec))
mask[453333:473444] = 1
mask[361000:374350] = 1
mask[270000:278700] = 1

xl = vFTS[np.where(mask==1)[0]][-1]
xr = vFTS[np.where(mask==1)[0]][0]

iFTS = np.where((vFTS > xl) & (vFTS < xr))[0]
itap = np.where((vtap > xl) & (vtap < xr))[0]
v = vFTS[iFTS]
s = spec[iFTS]*mask[iFTS]
vt = np.abs(vtap[itap])
st = np.abs(stap[itap])

# estimate yl yr starting values from s
yl = np.max(s[0:20])
yr = np.max(s[len(s)-20:])


######################
# Fit
######################
parname  = np.array(['tau','sig','yl','yr','k'])
params   = np.array([15.0, 0.001, yl, yr, 0.000002])
pmin     = np.array([0.8,0.0001,yl*0.99,yr*0.99,-0.00001])
pmax     = np.array([20.0,0.1,yl*1.01,yr*1.01,0.00001])
stepsize = np.array([0.01,0,0.1,0.1,0.000001])
#prior    = np.array([0.2, 0.2, yl, yr, lam0, dl0])
#priorlow = np.array([0.2, 0.2, yl, yr, lam0, dl0])
#priorup  = np.array([0.2, 0.2, yl, yr, lam0, dl0])

def gaussian(x, shift, sig):
    ' Return normalized gaussian with mean shift and var = sig^2 '
    gaus = np.exp(-.5*((x - shift)/sig)**2)/(sig * np.sqrt(2*np.pi))
    return gaus/sum(gaus)


def func(params, v,s,vt,st):
  """
  Minimization function
  params: optical depth tau, broadening sig, yl yr for continuum, k wavelength

  v, s == FTS wavenumber and spectra  (subarray around O2)
  vt,st == tapas wave# and transmission (subarray around O2)

  """
  tau, sig, yl, yr, k = params
  # Tau, convolve tapas
  psf       =  gaussian(vt,vt[int(len(vt)/2.)],sig)
  st_broad  =  fftconvolve(st**tau,psf,'same')
  # Transform v_corr to v wrong
  vt_shift  =  vt/(k+1)
  # Interpolate tapas and evaluate it on v_corr
  interfxn   = interp1d(vt_shift, st_broad, fill_value = 0, bounds_error=False)
  s_int      = interfxn(v)
  # Multiply by line for continuum
  m = (yl - yr)/(0 - len(s_int))
  line = m*v+ yl
  model = line*s_int/np.max(s_int)
  return model

# Put whatever for uncertainties & RUN
unc  = 0.001*s
#bestp, CRlo, CRhi, stdp, posterior, Zchain

out  = mc3.mcmc(s, unc,
             func=func, indparams=[v,s,vt,st], 
             params=params, stepsize=stepsize, pmin=pmin, pmax=pmax,
             nsamples=10**4, burnin=10**3,plots=True)


# Recreate best fit and lambda array
bestfit = func(params, v,s, vt,st)
tau, sig, yl, yr, k = params

from matplotlib import gridspec

params = {'legend.fontsize': 18,
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':18,
         'ytick.labelsize':18}
plt.rcParams.update(params)

f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[2, 1]},sharex=True)
a0.plot(v,s,label='FTS')
a0.plot(v,bestfit, 'k--',label='Model',lw=2)
#yticks0 = [20000,22000,24000,26000,28000,30000,32000]
#a0.set_yticks(yticks0)
residuals = 100 * (s-bestfit)/bestfit
a1.plot(v,residuals)
a1.plot(v,np.zeros(len(v)),'k--',lw=2)
plt.subplots_adjust(bottom=0.14, left = 0.15,hspace=0)
#yticks = [-2,-1,0,1,2]
#a1.set_yticks(yticks)
plt.xlim(min(v),max(v))
a1.set_xlabel('Wavenumber')
