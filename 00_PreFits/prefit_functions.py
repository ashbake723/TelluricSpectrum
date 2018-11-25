import astropy.io.fits as fits
import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import splev, BSpline
from astropy.time import Time
import urllib, json, glob, scipy, os, sys

sys.path.append('../read_opus/')
import opus2py as op

sys.path.append('../tools/')
from hitran_V3 import *


def define_pararr(data,nlines,pstart=None):
    # For each, define name and value
    params = np.concatenate((np.ones(1),          # stau
                             np.ones(1),           # o2_tau
                             np.zeros(2),	      # cont
                             np.zeros(1),          # o2_dnu
                             np.zeros(1),      # dnu
                             np.zeros(1),      # vel
                             np.ones(1),       # taus
                             data['gamma-air'],    # sigs L
                             data['Shift0'],  # linecents
                             np.log10(data['S'])           # line strength
                         ))         
							
    par_lo = np.concatenate((np.ones(1)*0.99,      # stau
                             np.ones(1)*0.1, 	   # o2_tau
                             np.zeros(2)-0.1,       # cont
                             np.zeros(1)-0.5,       # o2_dnu
                             np.zeros(1)-0.5,   # dnu
                             np.zeros(1)-0.8,   # vel
                             np.ones(1)*0.1,    # taus
                             data['gamma-air']*0.5,     # sigs
                             data['Shift0']-0.01,   # linecents
                             np.log10(data['S'])-0.3   # line strength
                         ))         

    
    par_hi = np.concatenate(( 
        np.ones(1)*1.1,          # stau
        np.ones(1)*10,           # o2_tau
        np.zeros(2)+0.1,         # continuum
        np.zeros(1)+0.5,         # o2_dnu
        np.zeros(1)+0.5,     # dnu
        np.zeros(1)+0.8,     # vel
        np.ones(1)*30.0,     # taus
        data['gamma-air']*1.7,   # sigs0
        data['Shift0']+0.01,     # linecents
        np.log10(data['S'])+0.3  # line strength
    ))         



    parnames = np.array(['stau'] + 
                        ['o2_tau'] +
                        ['cont']*2 +
                        ['sigsD']+
                        ['dnu']  + 
                        ['vel']  + 
                        ['taus']  + 
                        ['sigs0'] * nlines + 
                        ['Shift0'] * nlines +
                        ['S'] * nlines
                )
		
    bounds = []
	
    for i in range(len(par_lo)):
        bounds.append([par_lo[i],par_hi[i]])
    
    return parnames, params, bounds


def func2(params, v_shift, s,  unc, hitran_data, stel, o2, continuum,taus,iline = None, mode='sum'):
    """
    Minimization function
    
    Inputs:
    params:  model parameters, see fxn define_pararr
    v     :  wavenumber array
    s     :  array of spectral data
    unc   :  uncertainty array
    hitran_data: dictionary with hitran line parameters to feed hitran.py telluric spectrum fxn generator
    stel_mod:    kurucz stellar model
    hitran_var_data: original data for hitran variable modifying/fitting for
    hitran_var: 'linecenter' 'gamma-air' or 'S'
    
    Optional Inputs:
    mode:  'sum' or 'get_model'
    		'sum' returns reduced chi squared, 'get_model' returns model
    """
    nlines = len(hitran_data['I'])

    # Extract Paramaters from params array
    gamD    = params[0:nlines] # defined as percentage of original value
    gam0    = params[nlines:2*nlines]
    gam2    = params[2*nlines:3*nlines]
    anuVc   = params[3*nlines:4*nlines]
    eta     = params[4*nlines:5*nlines]
    Shift0  = params[5*nlines:6*nlines]
    Shift2  = params[6*nlines:7*nlines]
    S       = params[7*nlines:8*nlines]

    hitran_data['gamma-air']  = gam0      # fill in whichever hitran variable deciding to vary
    hitran_data['gamma-SD']   = gam2     # Speed dependent voigt
    hitran_data['gamD']       = gamD  
    hitran_data['anuVc']      = anuVc   
    hitran_data['eta']        = eta   
    hitran_data['Shift0']     = Shift0     
    hitran_data['Shift2']     = Shift2     
    hitran_data['S']          = 10**S
    
    # Scale density of molecules/scalar the same for everything
    dens  = 5e27
    
    # Get telluric spectrum
    telluric = calculate_hitran_xsec(hitran_data, wavenumarr=v_shift, npts=20001, units='m^2', temp=296.0, pressure=1.0)
    # define model
    model = stel + taus * dens * telluric[1] + continuum + o2
        
    if mode =='sum':
    	return np.sum(((model - s)**2)/unc)/10
    elif mode == 'get_model':
    	return model
    elif mode=='get_line':
        hitran_data['S'][iline] = 0
        telluric_mod = calculate_hitran_xsec(hitran_data, wavenumarr=v_shift, npts=20001, units='m^2', temp=296.0, pressure=1.0)		
        return (s - (stel + taus * dens * telluric_mod[1] + continuum + o2))


	
def define_pararr2(data,nlines,pstart=None):
    # For each, define name and value
    params = np.concatenate((
        data['gamD']*0 + 0.009,         # gamD
        data['gamma-air'],              # gam0
        data['gamma-SD']*0 + 0.012,     # gam2
        np.zeros(nlines)+0.09,          # anuVc
        np.zeros(nlines),           # eta
        data['Shift0'],             # Shift0
        np.zeros(nlines),           # Shift2
        np.log10(data['S'])         # line strength
    ))         
							
    par_lo = np.concatenate(( 
        data['gamD']*0 + 0.001,          # gamD
        data['gamma-air']-0.01,          # gam0
        data['gamma-SD']*0 + 0.001,      # gam2
        np.zeros(nlines) -0.1,           # anuVc
        np.zeros(nlines)-0.1,           # eta
        data['Shift0']-0.01,            # Shift0
        np.zeros(nlines)-0.025,         # Shift2
        np.log10(data['S'])-0.3        # line strength
    ))         


    par_hi = np.concatenate((
        data['gamD']*0 + 0.07,        # gamD
        data['gamma-air']+0.01,             # gam0
        data['gamma-SD']*0 + 0.05,      # gam2
        np.zeros(nlines)+0.15,           # anuVc
        np.zeros(nlines)+0.2,           # eta
        data['Shift0']+0.01,           # Shift0
        np.zeros(nlines)+0.025,           # Shift2
        np.log10(data['S'])+0.3     # line strength
    ))         



    parnames = np.array(
        ['gamD'] * nlines +
        ['gam0'] * nlines +
        ['gam2'] * nlines +
        ['anuVc'] * nlines +
        ['eta'] * nlines +
        ['Shift0']  * nlines + 
        ['Shift2'] * nlines +
        ['S'] * nlines
    )
		
    bounds = []
    
    for i in range(len(par_lo)):
        bounds.append([par_lo[i],par_hi[i]])
        
    return parnames, params, bounds
	
def func(params, v, s,  unc, hitran_data, stel_mod, o2_mod, iline = None, mode='sum'):
    """
    Minimization function
    Inputs:
    params:  model parameters, see fxn define_pararr
    v     :  wavenumber array
    s     :  array of spectral data
    unc   :  uncertainty array
    hitran_data: dictionary with hitran line parameters to feed hitran.py telluric spectrum fxn generator
    stel_mod:    kurucz stellar model
    hitran_var_data: original data for hitran variable modifying/fitting for
    hitran_var: 'linecenter' 'gamma-air' or 'S'
    Optional Inputs:
    mode:  'sum' or 'get_model'
    'sum' returns reduced chi squared, 'get_model' returns model
    """
    nlines = len(hitran_data['I'])

    # Extract Paramaters from params array
    m = 5      # number of fixed-defined variables
    stau = params[0:1]
    o2_tau = params[1:2]
    cont   = params[2:4]
    o2_dnu = params[4:5]
    dnu  = params[m  : m + 1]
    vel  = params[m + 1 : m + 2]
    taus = params[m + 2 : m + 3]
    sigs0 = params[m + 3: m + 3 + nlines] # defined as percentage of original value
    shift0 = params[m + 3 + nlines: m + 3 + 2*nlines]
    strengths  = params[m + 3 + 2*nlines: m + 3 + 3*nlines]

    hitran_data['gamma-air']  = sigs0   # fill in whichever hitran variable deciding to vary
    hitran_data['Shift0'] = shift0
    hitran_data['S'] = 10**strengths
    
    # Scale density of molecules/scalar the same for everything
    dens  = 5e27
	
    tck      = interpolate.splrep(v[::-1]*(1+vel/3.0e5),stel_mod[::-1]*stau, s=0)
    stel     = interpolate.splev(v,tck,der=0)
    # O2 Shift
    tck      = interpolate.splrep(v[::-1]*(1+o2_dnu/3.0e5),o2_mod[::-1]*o2_tau, s=0)
    o2       = interpolate.splev(v,tck,der=0)
    # Telluric component
    v_shift  = v*(1 + dnu/3.0e5)
    telluric = calculate_hitran_xsec(hitran_data, wavenumarr=v_shift, npts=20001, units='m^2', temp=296.0, pressure=1.0)
    # Continuum
    continuum_slope = (cont[0] - cont[1])/(v[0] - v[-1])
    continuum       = continuum_slope * (v - v[0])  + cont[0]
    # define model
    model = stel + taus * dens * telluric[1] + continuum + o2
        
    if mode =='sum':
    	return np.sum(((model - s)**2)/unc)/1000
    elif mode == 'get_model':
    	return model,stel,o2,continuum,v_shift
    elif mode=='get_line':
        hitran_data['S'][iline] = 0
        telluric_mod = calculate_hitran_xsec(hitran_data, wavenumarr=v_shift, npts=20001, units='m^2', temp=296.0, pressure=1.0)		
        return (s - (stel + taus * dens * telluric_mod[1] + continuum + o2))
    	
    	
def plot_fit(so):
    f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.plot(1e7/so.cal.v,so.cal.s,'r.',label='Observed Spectrum')
    ax1.plot(1e7/so.cal.v,so.cal.stel,'g-',label='Stellar Template')
    ax1.plot(1e7/so.cal.v,model,'k--',label='Model Spectrum')
    ax1.plot(1e7/so.cal.v,so.cal.iodine,'m-',label='Iodine Template')
    ax2.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Flux')
    ax1.legend(loc='best',fontsize=9)
    plt.subplots_adjust(hspace=0)
    ax2.plot(1e7/so.cal.v,so.cal.s-model)
    ax2.set_ylabel('Residuals')
    ax2.set_ylim(-0.03,0.03)
    ax1.set_xlim(621.62,621.81)
    plt.xticks([621.65, 621.70,621.75,621.80])



def save_spectrum(fname,tobs,params,pnames,stdp,l,f,res):
    """
    Save final spectrum's wavelength, flux, and residuals
    """
    tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='wavelength', format='E', array=l), fits.Column(name='flux', format='E', array=f), fits.Column(name='residuals', format='E', array=res)])
    prihdr = fits.Header()
    
    for i in range(len(pnames)):
        prihdr[pnames[i]]        = params[i]      # best fit params
        prihdr[pnames[i]+'_std'] = stdp[i]        # 
    prihdr['obs_time']   = tobs
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    if os.path.isfile('../DATA/FTS_processed/%s' %fname):
        os.system('rm ../DATA/FTS_processed/%s' %fname)
    
    thdulist.writeto('../DATA/FTS_processed/%s' %fname)



