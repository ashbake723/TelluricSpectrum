
 
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
sys.path.append('../read_opus/')
import opus2py as op
sys.path.append('../')
import glob
import scipy 
from hitran_V2 import *

plt.ion()

from functions import storage_object
so = storage_object()
from functions import gaussian,get_cont

rcparams = {'legend.fontsize': 18,
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':18,
         'ytick.labelsize':18}
plt.rcParams.update(rcparams)


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
    so.datapath  = '../DATA/'
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
    for i,specname in enumerate(so.ftsnames[0:10]):#nfiles]):
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
        stellar = fits.open('../DATA/STELLAR/kurucz_500000_sunall.fits') #kurucz
        so.stel.v = (1/(1e-7 * stellar[1].data['wavelength']))[600000:1010000] # nm to cm^-1
        so.stel.s = 4 * np.pi * stellar[1].data['flux'][600000:1010000] * 1e7 * (1/so.stel.v)**2
        stellar.close()

    elif stelmod == 'kurucz_irr':         # http://kurucz.harvard.edu/stars/sun/
        stellar = fits.open('../DATA/STELLAR/irradthuwn.fits') #irradiance
        so.stel.v = stellar[1].data['wavenumber'][0:1200000] 
        so.stel.s = stellar[1].data['irradiance'][0:1200000]
        stellar.close()

    elif stelmod == 'phoenix':
        #phoenix
        stellar = fits.open('../DATA/STELLAR/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
        so.stel.v = 1/(1e-8 * stellar[0].data)[300000:1000000] # angstrom to wavenumber
        stellar.close()

        stellar = fits.open('../DATA/STELLAR/lte05700.fits')
        so.stel.s = stellar[0].data[300000:1000000] * (1/so.stel.v)**2 # times cm^2 to get erg/s/cm2/cm-1
        stellar.close()

    elif stelmod == 'atlas':
        # Solar ATLAS...this isn't a good option, jsut dont want to load
        # it if i don't have to
        atlas  = fits.open('../DATA/solarAtlas/solar_atlas_V1_405-1065.fits')
        so.stel.v  = atlas[1].data['wavenumber']
        so.stel.s  = atlas[1].data['flux_norm']
        atlas.close()
    
    else:
        raise Warning("Didn't choose a proper stellar model option. Choose" 
                        "either phoenix or kurucz or atlas")


    # EPHEM
    vv = fits.open('../DATA/EPHEM/%s_perspec.fits' %so.ftsday[12:],dtype=str)
    ephem= vv[1].data
    so.eph.fnames = ephem['specname']
    so.eph.vels   = ephem['velocity']
    vv.close()


    # TAPAS (...2.fits is from Bremen Germany, slightly lower than Gott.)
    so.tapname     = 'TAPAS/tapas_oxygen.fits'
    so.tapname_h2o = 'TAPAS/tapas_water.fits'

    tapas     = fits.open(so.datapath + so.tapname) # oxygen
    so.tap.v  = tapas[1].data['wavenumber']
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
    f = np.loadtxt('../fit_wavelengths.txt')
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
	if os.path.isfile('../DATA/FTS_processed/%s' %fname):
		os.system('rm ../DATA/FTS_processed/%s' %fname)
		
	thdulist.writeto('../DATA/FTS_processed/%s' %fname)


def define_pararr(data,nlines,nspec,pstart=None):
	# For each, define name and value
	params = np.concatenate((np.ones(1),          # stau
							np.ones(1),           # o2_tau
							np.zeros(2),	      # cont
							np.zeros(1),          # o2_dnu
							np.zeros(nspec),      # dnu
							np.zeros(nspec),      # vel
							np.ones(nspec),       # taus
							data['gamma-air'],    # sigs L
							data['gamma-SD'],     # sigs 2
							data['linecenter'],  # linecents
							np.log10(data['S'])           # line strength
							))         
							
	par_lo = np.concatenate((np.ones(1)*0.99,      # stau
							np.ones(1)*0.1, 	   # o2_tau
							np.zeros(2)-0.1,       # cont
							np.zeros(1)-0.5,       # o2_dnu
							np.zeros(nspec)-0.5,   # dnu
							np.zeros(nspec)-0.8,   # vel
							np.ones(nspec)*0.1,    # taus
                            data['gamma-air']*0.5,     # sigs
							data['gamma-SD']*0.5,      # sigs
							data['linecenter']-0.01,   # linecents
							np.log10(data['S'])-0.3            # line strength
							))         


	par_hi = np.concatenate(( 
							np.ones(1)*1.1,          # stau
							np.ones(1)*10,           # o2_tau
							np.zeros(2)+0.1,         # continuum
							np.zeros(1)+0.5,         # o2_dnu
							np.zeros(nspec)+0.5,     # dnu
							np.zeros(nspec)+0.8,     # vel
							np.ones(nspec)*30.0,     # taus
							data['gamma-air']*1.5,   # sigs0
							data['gamma-SD']*1.5,    # sigs2
							data['linecenter']+0.01, # linecents
							np.log10(data['S'])+0.3            # line strength
							))         



	parnames = np.array(['stau'] + 
    					 ['o2_tau'] +
    					 ['cont']*2 +
    					 ['sigsD'] * nspec +
						 ['dnu']  * nspec + 
						 ['vel']  * nspec + 
						 ['taus'] * nspec + 
						 ['sigs0'] * nlines + 
						 ['sigs2'] * nlines +
						 ['linecents'] * nlines +
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
    nspec  = len(s)

    # Extract Paramaters from params array
    m = 5      # number of fixed-defined variables
    stau = params[0:1]
    o2_tau = params[1:2]
    cont   = params[2:4]
    o2_dnu = params[4:5]
    dnu  = params[m  : m + nspec]
    vel  = params[m + nspec : m + 2*nspec]
    taus = params[m + 2*nspec : m + 3*nspec]
    sigs0 = params[m + 3*nspec: m + 3*nspec + nlines] # defined as percentage of original value
    sigs2 = params[m + 3*nspec + nlines: m + 3*nspec + 2*nlines]
    linecents  = params[m + 3*nspec + 2*nlines: m + 3*nspec + 3*nlines]
    strengths  = params[m + 3*nspec + 3*nlines: m + 3*nspec + 4*nlines]

    hitran_data['gamma-air']  = sigs0   # fill in whichever hitran variable deciding to vary
    hitran_data['gamma-SD']   = sigs2     # Speed dependent voigt
    hitran_data['linecenter'] = linecents
    hitran_data['S'] = 10**strengths
    
	# Scale density of molecules/scalar the same for everything
    dens  = 5e27
	
	# Create model array 
    model = np.zeros(np.shape(s))
    
    for i in range(len(s)):
	    tck      = interpolate.splrep(v[::-1]*(1+vel[i]/3.0e5),stel_mod[::-1]*stau, s=0)
	    stel     = interpolate.splev(v,tck,der=0)
        # O2 Shift
	    tck      = interpolate.splrep(v[::-1]*(1+o2_dnu/3.0e5),o2_mod[::-1]*o2_tau, s=0)
	    o2       = interpolate.splev(v,tck,der=0)
		# Telluric component
	    v_shift  = v*(1 + dnu[i]/3.0e5)
	    telluric = calculate_hitran_xsec(hitran_data, wavenumarr=v_shift, npts=20001, units='m^2', temp=296.0, pressure=1.0)
		# Continuum
	    continuum_slope = (cont[0] - cont[1])/(v[0] - v[-1])
	    continuum       = continuum_slope * (v - v[0])  + cont[0]
	    # Get kernel to convolve with
	    #		K = kernel(v, 4, a[i])
	    # define model
	    model[i] = stel + taus[i] * dens * telluric[1] + continuum + o2
        
    if mode =='sum':
    	return np.sum(((model - s)**2)/unc)/1000
    elif mode == 'get_model':
    	return model,stel
    elif mode=='get_line':
        hitran_data['S'][iline] = 0
        telluric_mod = calculate_hitran_xsec(hitran_data, wavenumarr=v_shift, npts=20001, units='m^2', temp=296.0, pressure=1.0)		
        return (s[0] - (stel + taus[i] * dens * telluric_mod[1] + continuum + o2))
    	
    	


def setup_data(ifold, center, dk):
	"""
	Get chunk of data with flatten spectrum ready to fit
	
	inputs:
	ifold - folder index, which folder to load
	cent  - center of subregion
	dk    - width of subregion
	
	outputs:
	vflat
	sflat
	"""
	# START
	so.ftsfolders   = glob.glob('../DATA/FTS/*')
	fnames = []
	
	# Load folder and data
	so.ftsday = so.ftsfolders[ifold]
	load_data(so,nfiles='all',stelmod='kurucz_irr')

	# Get i0,i1
	i1 = np.where(so.fts.v > (center - dk))[0][-1]
	i0 = np.where(so.fts.v < (center + dk))[0][0]

    # Apply subregion
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
	
	return vflat, -1*np.log(sflat),fnames



#if __name__ == '__main__':
for ispec in range(0,1):
    # User chosen things
    ifold           = 0  #folder index to work on
    icent           = 7
#    ispec           = 0 
    
    # Load linecents
    f = np.loadtxt('isolated_lines.txt')
    linecenters,indicents = f[:,0],f[:,1]
    
    v,s,fnames = setup_data(ifold,linecenters[icent],5.0)
    s = s[ispec:ispec+1]   
    #data = read_hitran2012_parfile('HITRAN/h2o_15429_15449.par')
    data = read_hitran2012_parfile('HITRAN/h2o_%s.par' %int(linecenters[icent]))
        
    data['sig_D'] = 0.001
    
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
                             bounds_error=False,fill_value=so.stel.s[ssub][-1])
    stel_mod  = int_model(v)
    stel_mod  = -1*np.log(stel_mod/np.max(stel_mod))

    # O2 Model
    o2sub = np.where( (so.tap.v < v[0]) & (so.tap.v > v[-1]))[0]
    int_model = interp1d(so.tap.v[o2sub],so.tap.o2[o2sub],kind='linear',
                             bounds_error=False,fill_value=so.tap.o2[o2sub][-1])
    o2_mod  = int_model(v)
    o2_mod  = -1*np.log(o2_mod/np.max(o2_mod))

    # print
    print 'About to start the fitting process'

	# First fit for speed dependent voigt
    out = opt.minimize(func,params_start,\
        args=(v, s, unc, data, stel_mod,o2_mod),\
        method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 

	# Analyze outputs
    params = out['x']
    model,stel = func(params, v, s,  unc, data, stel_mod, o2_mod,mode='get_model')
    
    # Take out lines from spectrum (subtract b/c absorbance)
    pickline =  np.where(np.abs(data['linecenter'] - linecenters[icent]) < 0.03)[0]
    just_line = func(params, v, s,  unc, data, stel_mod, o2_mod, iline=pickline,mode='get_line')

 ################# SAVE  ###########################    
    
    # Save spectra without any lines except isolated one
    savename = fnames[ispec].replace('.','_spec')+'_line%s'%int(linecenters[icent])

    # Save fit outputs 
    nlines = len(data['I'])
    nspec  = len(s)
    p_lo = np.array(bounds)[:,0]
    p_hi = np.array(bounds)[:,1]
    frac =  (params-p_lo)/(p_hi-p_lo)

    # Save to Fits
    hdu1 = fits.BinTableHDU.from_columns(
       [fits.Column(name='v', format='E', array=v),
        fits.Column(name='s',format='E',array=just_line)
         ])
         
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='pnames', format='10A', array=pnames),
        fits.Column(name='params',format='E',array=params),
        fits.Column(name='params_start',format='E',array=params_start),
        fits.Column(name='frac',format='E',array=frac)
         ])
         
    hdr = fits.Header()
    hdr['nlines'] = nlines
    hdr['nspec']  = nspec
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([primary_hdu, hdu,hdu1])


    if os.path.exists('SavedLines/%s.fits' %(savename)):
         os.system('rm %s' % 'SavedLines/%s.fits' %(savename))
         hdul.writeto('SavedLines/%s.fits' %(savename))
    else:
        hdul.writeto('SavedLines/%s.fits' %(savename))


   
############################################################
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
    getsigs = np.where(pnames == 'sigs0')
    plt.scatter(np.arange(len(params))[getsigs], frac[getsigs], c=(62+np.log(data['S'])))
    getsigs = np.where(pnames == 'sigs2')
    plt.scatter(np.arange(len(params))[getsigs], frac[getsigs], c=(62+np.log(data['S'])))

    plt.xlabel('Parameter Index')
    plt.ylabel('Fraction from Bounds')
    
    #label regions
    for i,pname in enumerate(np.unique(pnames)):
        indices = np.where(pnames==pname)[0]
        plt.axvline(x = indices[0]-0.5,c='r')
        plt.axvline(x = indices[-1]+0.5,c='r')
        
        plt.text(np.mean(indices)-0.1,1.1,pname)
        

    
    
    
