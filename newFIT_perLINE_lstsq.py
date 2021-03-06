
 
# Get k factor for spectrum
# opus2py documentation: file:///Users/ashbake/Documents/Research/Projects/TelluricCatalog/read_opus-1.5.0-Linux/doc/opus2py/opus2py-module.html

# look into
# http://iancze.github.io/Starfish/


import astropy.io.fits as fits
import matplotlib.pylab as plt
from matplotlib import gridspec
import numpy as np
import scipy.optimize as opt
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import splev, BSpline
import os,sys
from astropy.time import Time
sys.path.append('read_opus/')
import opus2py as op
import glob
import scipy 
from hitran import *

plt.ion()

from functions import storage_object
so = storage_object()
from functions import gaussian,get_cont

params = {'legend.fontsize': 18,
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':18,
         'ytick.labelsize':18}
plt.rcParams.update(params)


def fit_poly(x,y,xnew,deg=2):
	"""
	Fit a polynomial  to x,y and return polynomial sampled
	at xnew.
	
	deg - degree of polynomial, optional, default 2
	
	output: fit sampled at xnew
	"""
	par = np.polyfit(x,y,deg)
	fit = np.zeros(len(xnew))
	for i in range(len(par)):
		fit += par[i]*xnew**(deg-i)
	return fit


def get_time(time,date):
	"""
	Given the time and date of the FTS file, return sjd as Time object
	"""
	sts = date[6:] + '-' + date[3:5] + '-' + date[0:2] + ' ' + time[0:12]
	gmtplus = float(time[18])
	sjd = Time(sts, format='iso', scale='utc').jd - gmtplus/24.0 # subtract +1 hr
	return sjd


def load_data(so,nfiles='all',stelmod='kurucz'):
    """
    Load Everything to so object
    
    Inputs:
    ------- 
    so  : storage oject defined in functions.py

    nfiles : either integer or 'all' 
             Specifies how many files to read
    
    stelmod: default 'kurucz'
             can choose which stellar model. other option: 'phoenix'
    """
    ### Names
    so.datapath  = 'DATA/'
    so.ftsnames  = glob.glob('%s/*I2.0*' %so.ftsday) 
    if len(so.ftsnames) == 0:
        so.ftsnames  = glob.glob('%s/*0.0*' %so.ftsday) #when I2 not present
    if len(so.ftsnames) == 0:
        so.ftsnames  = glob.glob('%s/*I2*.0*' %so.ftsday) #when no halogens
    
    ###  Load things

    # FTS spectrum
    if nfiles =='all': #define all to be no. of files in folder
        nfiles = len(so.ftsnames)

    s_all   = np.zeros((nfiles,len(op.read_spectrum(so.ftsnames[0]))))
    t_all   = np.zeros(nfiles)
    fnames  = []
    badspec  = []
    for i,specname in enumerate(so.ftsnames[0:5]):#nfiles]):
        s_all[i] = np.array(op.read_spectrum(specname))
        print specname
        
        # Get time of observation
        meta     = op.get_spectrum_metadata(specname)
        t_all[i] = get_time(meta.time,meta.date)

        # Keep list of filenames
        fnames.append(specname.split('/')[-1])
        
        # Keep track of which spectra are full of nans, delete later
        if len(np.where(np.isnan(s_all[i]))[0]) > 0:
            badspec.append(i)
    
    so.nspec    = nfiles - len(badspec)                         # number of spectra in s_all
    so.fts.s    = np.delete(s_all,badspec,axis=0)               # spectra, no nan specs
    so.fts.meta = op.get_spectrum_metadata(so.ftsnames[0])      # meta data for first file
    so.fts.v    = np.linspace(so.fts.meta.xAxisStart,so.fts.meta.xAxisEnd,so.fts.meta.numPoints)
    so.fts.fnames = np.delete(np.array(fnames),badspec,axis=0)  # running list of file names
    so.fts.t_all = np.delete(t_all,badspec)                     #times of all the observations

    # STELLAR MODEL (to fit continuum) 
    if stelmod == 'kurucz':         # http://kurucz.harvard.edu/stars/sun/00asun.readme
        stellar = fits.open('DATA/STELLAR/kurucz_500000_sunall.fits') #kurucz
        so.stel.v = (1/(1e-7 * stellar[1].data['wavelength']))[600000:1010000] # nm to cm^-1
        so.stel.s = 4 * np.pi * stellar[1].data['flux'][600000:1010000] * 1e7 * (1/so.stel.v)**2
        stellar.close()

    elif stelmod == 'kurucz_irr':         # http://kurucz.harvard.edu/stars/sun/
        stellar = fits.open('DATA/STELLAR/irradthuwn.fits') #irradiance
        so.stel.v = stellar[1].data['wavenumber'][0:1200000] 
        so.stel.s = stellar[1].data['irradiance'][0:1200000]
        stellar.close()

    elif stelmod == 'phoenix':
        #phoenix
        stellar = fits.open('DATA/STELLAR/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
        so.stel.v = 1/(1e-8 * stellar[0].data)[300000:1000000] # angstrom to wavenumber
        stellar.close()

        stellar = fits.open('DATA/STELLAR/lte05700.fits')
        so.stel.s = stellar[0].data[300000:1000000] * (1/so.stel.v)**2 # times cm^2 to get erg/s/cm2/cm-1
        stellar.close()

    elif stelmod == 'atlas':
        # Solar ATLAS...this isn't a good option, jsut dont want to load
        # it if i don't have to
        atlas  = fits.open('DATA/solarAtlas/solar_atlas_V1_405-1065.fits')
        so.stel.v  = atlas[1].data['wavenumber']
        so.stel.s  = atlas[1].data['flux_norm']
        atlas.close()
    
    else:
        raise Warning("Didn't choose a proper stellar model option. Choose" 
                        "either phoenix or kurucz or atlas")


    # EPHEM
    vv = fits.open('DATA/EPHEM/%s_perspec.fits' %so.ftsday[9:],dtype=str)
    ephem= vv[1].data
    so.eph.fnames = ephem['specname']
    so.eph.vels   = ephem['velocity']
    vv.close()


    # TAPAS (...2.fits is from Bremen Germany, slightly lower than Gott.)
    so.tapname    = 'TAPAS/tapas_oxygen.fits'
    so.tapname_h2o= 'TAPAS/tapas_water.fits'

    tapas    = fits.open(so.datapath + so.tapname) # oxygen
    so.tap.v = tapas[1].data['wavenumber']
    so.tap.o2 = tapas[1].data['transmittance']
    tapas.close()

    tapas    = fits.open(so.datapath + so.tapname_h2o)
    so.tap.h2o = tapas[1].data['transmittance']
    tapas.close()



def fit_continuum0(subv,spec,filelist):
    """
    Flatten data and apply k shift
    """
    # Open file with wavelength regions to fit
    f = np.loadtxt('fit_wavelengths.txt')
    v_c = f[:,0]
    dv  = f[:,1]


    # Storage for final spectrum
    final_f = np.zeros((len(spec),len(subv)))

    for j in range(len(spec)):
        # Repeat this part
        subf = spec[j]#[isub]

        # Pick regions, flux
        v_cont = []
        f_cont = []
        for i in range(len(v_c)):
            temp_is = np.where((subv > v_c[i]) & (subv < (v_c[i]+dv[i])))[0]
            if len(temp_is) > 0:
                v_cont.append(np.median(subv[temp_is]))
                f_cont.append(np.mean(subf[temp_is]))

        fit = fit_poly(v_cont,f_cont,subv,deg=1)
        final_f[j] = spec[j]/fit

    return subv,final_f


def gaussian(x, shift, sig):
    ' Return normalized gaussian with mean shift and var = sig^2 '
    gaus = np.exp(-.5*((x - shift)/sig)**2)/(sig * np.sqrt(2*np.pi))
    return gaus/sum(gaus)


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
	if os.path.isfile('DATA/FTS_processed/%s' %fname):
		os.system('rm DATA/FTS_processed/%s' %fname)
		
	thdulist.writeto('DATA/FTS_processed/%s' %fname)


def define_pararr(data,nlines,nspec):
	# For each, define name and value
	
	params = np.concatenate((np.ones(1),          # stau
							np.ones(1),           # press
							np.zeros(2),	      # cont
							np.zeros(nspec),      # a
							np.zeros(nspec),      # dnu
							np.zeros(nspec),      # vel
							np.ones(nspec),       # taus
							np.ones(nlines)))     # hitran parameter 

							
	par_lo = np.concatenate((np.ones(1)*0.95,      # stau
							np.ones(1)*0.97, 	   # press
							np.zeros(2)-0.1,       # cont
							np.zeros(nspec)*0.9,      # a
							np.zeros(nspec),       # dnu
							np.zeros(nspec)-0.8,   # vel
							np.ones(nspec)*0.1,    # taus
							np.ones(nlines)*0.6))  # hitran

						
	par_hi = np.concatenate(( 
							np.ones(1)*1.2,        # stau
							np.ones(1)*1.03,        # press
							np.zeros(2)+0.1,       # continuum
							np.zeros(nspec)*1.1,    # a
							np.zeros(nspec),        # dnu
							np.zeros(nspec)+0.8,   # vel
							np.ones(nspec)*15.0,   # taus
							np.ones(nlines)*1.4))   # hitran
							#data['S']*1.1))         # deps


	parnames = np.array(['stau'] + 
    					 ['press'] +
    					 ['cont']*2 +
    					 ['a'] * nspec +
						 ['dnu']  * nspec + 
						 ['vel']  * nspec + 
						 ['taus'] * nspec +
						 ['hitran'] * nlines
    					 )
		
	bounds = []
	
	for i in range(len(par_lo)):
		bounds.append([par_lo[i],par_hi[i]])
		
	return parnames, params, bounds



def func(params, v, s,  unc, hitran_data, stel_mod, hitran_var_arr, hitran_var,mode='sum'):
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
    nlines = len(data['I'])
    nspec  = len(s)

    # Extract Paramaters from params array
    m = 4      # number of fixed-defined variables
    stau = params[0:1]
    pres = params[1:2]
    cont = params[2:4]
    a    = params[m : m + nspec]
    dnu  = params[m + nspec : m + 2*nspec]
    vel  = params[m + 2*nspec : m + 3*nspec]
    taus = params[m + 3*nspec : m + 4*nspec]
    hit  = params[m + 4*nspec: m + 4*nspec + nlines] # defined as percentage of original value
    

    # Apply hitran modification - depends on variable
    if hitran_var == 'gamma-air':
    	hitran_data[hitran_var] = hitran_var_arr * hit  # fill in whichever hitran variable deciding to vary
    elif hitran_var == 'linecenter':
	    hitran_data[hitran_var] = hitran_var_arr + (hit - 1)/100.0  # fill in whichever hitran variable deciding to vary
    elif hitran_var == 'S':
        hitran_data[hitran_var] = hitran_var_arr * hit
    else:
		print 'Pick a proper hitran variable name to vary!'
		return
	
	
	# Scale density of molecules/scalar the same for everything
    dens  = 5e26
    
    # Create model array
    model = np.zeros(np.shape(s))
    
    for i in range(len(s)):
	# Stellar component
		tck      = interpolate.splrep(v[::-1]*(1+vel[i]/3.0e5),stel_mod[::-1]*stau, s=0)
		stel     = interpolate.splev(v,tck,der=0)
		# Telluric component
		v_shift  = v*(1 + dnu[i]/3.0e5)
		telluric = calculate_hitran_xsec(hitran_data, wavenumarr=v_shift, npts=20001, units='m^2', temp=296.0, pressure=pres)
		# Continuum
		continuum_slope = (cont[0] - cont[1])/(v[0] - v[-1])
		continuum       = continuum_slope * (v - v[0])  + cont[0]
        # define model
		model[i] = stel + taus[i] * dens * telluric[1] - continuum

    if mode =='sum':
    	return np.sum(((model - s)**2)/unc)/1000
    elif mode == 'get_model':
    	return model




def setup_data(ifold, i0=531100,i1=531900):
	"""
	Get chunk of data with flatten spectrum ready to fit
	
	inputs:
	ifold - folder index, which folder to load
	i0    - lower index of range
	i1    - higher index of range
	
	outputs:
	vflat
	sflat
	"""
	# START
	so.ftsfolders   = glob.glob('DATA/FTS/*')
	fnames = []
	# Load folder and data
	so.ftsday = so.ftsfolders[ifold]
	load_data(so,nfiles='all',stelmod='kurucz_irr')
	spec = so.fts.s[:,i0:i1]
	v = so.fts.v[i0:i1]
	
	fnames = so.fts.fnames
	print so.ftsday
	nightlist = np.array([so.ftsday]*300)
	# CREATE FLATTENED DATA
	muspecs = np.mean(spec,axis=1)
	# keep spec with high snr
	igood = np.where(muspecs > 3*np.std(muspecs))[0]
	spec = spec[igood]
	goodlist = nightlist[igood]
	fnames = np.array(fnames)[igood]
	nspec = len(igood)
	t_all = so.fts.t_all[igood]
	
	flatspec = fit_continuum0(v,spec,goodlist)
	vflat = flatspec[0]
	sflat = flatspec[1]
	
	return vflat, -1*np.log(sflat)


def stelmod_resid():
    pass


if __name__ == '__main__':
    # User chosen things
    ifold           = 0  #folder index to work on
    
    v,s = setup_data(ifold)
    s=s[0:1]
    data = read_hitran2012_parfile('DATA/HITRAN/hitran_H2O_11984_11999.par')

    # STorage arrays
    nspec = len(s)
    deg_cont = 3
    
    pnames,params_start,bounds = define_pararr(data,len(data['S']),len(s))                             

    residuals = []
    fit_eph = np.zeros(nspec)
    act_eph = np.zeros(nspec)
    chi2    = np.zeros(nspec)
    param_final = np.zeros((nspec,len(pnames)))
    param_std   = np.zeros((nspec,len(pnames)))

	# Get uncertainties
    unc = 0.001* np.sqrt(np.abs(np.exp(s[0])))
    unc[0:2] = 10.0
    unc[len(unc)-3:] = 10.0
    unc[np.where(s[0] < -8)] = 10.0
    unc = (1/np.exp(s[0])) * unc # propogate to log

    #STELLAR MODEL
    ssub = np.where( (so.stel.v < v[0]) & (so.stel.v > v[-1]))[0]
    int_model = interp1d(so.stel.v[ssub],so.stel.s[ssub],kind='linear',
                             bounds_error=False,fill_value=so.stel.s[-1])
    stel_mod  = int_model(v)
    stel_mod  = -1*np.log(stel_mod/np.max(stel_mod))

	# First fit for gamma-air
    hitran_var = 'gamma-air'
    hitran_var_arr = data['gamma-air']
    out = opt.minimize(func,params_start,\
        args=(v, s, unc, data, stel_mod,hitran_var_arr,hitran_var),\
        method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 

	# Analyze outputs
    params = out['x']
    model = func(params, v, s,  unc, data, stel_mod, hitran_var_arr,hitran_var,mode='get_model')
    data[hitran_var] = hitran_var_arr * params[np.where(pnames == 'hitran')]
	
    print x
    
    # Second, fit for line centers
    hitran_var = 'linecenter'
    hitran_var_arr = data[hitran_var]
    out = opt.minimize(func,params_start,\
        args=(v, s, unc, data, stel_mod,hitran_var_arr,hitran_var),\
        method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 

    params = out['x']
    model = func(params, v, s,  unc, data, stel_mod, hitran_var_arr,hitran_var,mode='get_model')
    data[hitran_var] = hitran_var_arr + (params[np.where(pnames == 'hitran')] -1)/100.0

    print x
    
    # Third, fit for line depths
    hitran_var = 'S'
    hitran_var_arr = data[hitran_var]
    out = opt.minimize(func,params_start,\
        args=(v, s, unc, data, stel_mod,hitran_var_arr,hitran_var),\
        method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 

    params = out['x']
    model = func(params, v, s,  unc, data, stel_mod, hitran_var_arr,hitran_var,mode='get_model')
    data[hitran_var] = hitran_var_arr + (params[np.where(pnames == 'hitran')] -1)/100.0


#    out = opt.minimize(func,params_start,\
#    args=(v, s, unc, data, stel_mod,data['gamma-air'],'gamma-air'),\
#    method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 
 
    # Plot Best Fit
    ispec = 0
    f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[2, 1]},sharex=True)
    a0.plot(v,s[ispec],lw=2)
    a0.plot(v,model[ispec],'k--')
    a0.set_xlim([v[-3],v[2]])
    a0.set_ylabel('Absorbance')
    a0.set_title('Spectrum with Best Fit')

    a1.plot(v,s[ispec]-model[ispec],'k-')
    a1.plot(v,s[ispec]*0,'r--')
    a1.set_xlabel('Wavenumber (cm-1)')
    yticks = [-0.1,0,0.1]
    a1.set_yticks(yticks)
    a1.set_ylim([-0.15,0.15])
    plt.subplots_adjust(hspace=0)

    # Plot
    plt.figure(-11)
    p_lo = np.array(bounds)[:,0]
    p_hi = np.array(bounds)[:,1]
    frac =  (params-p_lo)/(p_hi-p_lo)
    plt.plot(np.arange(len(params)), frac,'o')
    plt.xlabel('Parameter Index')
    plt.ylabel('Fraction from Bounds')
    
    
    
    
    
    
    