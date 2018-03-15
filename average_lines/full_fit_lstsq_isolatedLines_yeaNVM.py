
 
# Get k factor for spectrum
# opus2py documentation: file:///Users/ashbake/Documents/Research/Projects/TelluricCatalog/read_opus-1.5.0-Linux/doc/opus2py/opus2py-module.html

# note to self: fit a small section for k, tau, yl-yr stuff to get tau and sigma
# then apply tau sigma and reinterpolate then do CC for kappa

import astropy.io.fits as fits
import matplotlib.pylab as plt
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


plt.ion()
#from sklearn.decomposition import PCA


from functions import storage_object
so = storage_object()
from functions import gaussian,get_cont

def fit_poly(x,y,xnew,deg=2):
    par = np.polyfit(x,y,deg)
    fit = np.zeros(len(xnew))
    for i in range(len(par)):
        fit += par[i]*xnew**(deg-i)
    return fit


def get_k(day):
    # Load k
    f = open('../k_factors.txt','r')
    lines = f.readlines()
    f.close()
    
    for l in lines:
        if l.startswith(day):
            k = float(l.split('\t')[1])

    return k


def get_time(time,date):
    sts = date[6:] + '-' + date[3:5] + '-' + date[0:2] + ' ' + \
                   time[0:12]
    gmtplus = float(time[18])
        
    sjd = Time(sts, format='iso', scale='utc').jd - gmtplus/24.0 # subtract +1 hr
    return sjd


def set_up(so,nfiles='all',stelmod='kurucz'):
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
    so.tapname    = 'TAPAS/tapas_oxygen.fits'
    so.tapname_h2o= 'TAPAS/tapas_water.fits'

    tapas    = fits.open(so.datapath + so.tapname) # oxygen
    so.tap.v = tapas[1].data['wavenumber']
    so.tap.o2 = tapas[1].data['transmittance']
    tapas.close()

    tapas    = fits.open(so.datapath + so.tapname_h2o)
    so.tap.h2o = tapas[1].data['transmittance']
    tapas.close()



def fit_continuum0(v,spec,filelist):
    """
    Flatten data and apply k shift
    """
    # Open file with wavelength regions to fit
    f = np.loadtxt('../fit_wavelengths.txt')
    v_c = f[:,0]
    dv  = f[:,1]

    # Define sub grid
    #isub = np.where( (v > v_c[0]) & (v < (v_c[-1]+ 5)) )[0]
    subv = v#[isub]

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

        fit = fit_poly(v_cont,f_cont,subv,deg=3)

        # Now correct shift - compare to 0th
        # Apply k from file
        k = get_k(filelist[j].split('/')[-1])
        tck = interpolate.splrep(subv[::-1]/(1+k),subf[::-1]/fit[::-1], s=0)
        f_new = interpolate.splev(subv,tck,der=0)
        final_f[j] = f_new

    return subv,final_f


def gaussian(x, shift, sig):
    ' Return normalized gaussian with mean shift and var = sig^2 '
    gaus = np.exp(-.5*((x - shift)/sig)**2)/(sig * np.sqrt(2*np.pi))
    return gaus/sum(gaus)



def save_spectrum(fname,tobs,params,pnames,stdp,l,f,res):
    tbhdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='wavelength', format='E', array=l),
         fits.Column(name='flux', format='E', array=f),
         fits.Column(name='residuals', format='E', array=res)])

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


def define_pararr():
	# Define array as name  init_guess lo  hi 
    param_arr   = np.array([
    ['t_h2o',	2.0,	0.1,	20.0],
    ['t_o2',	2.0,	2.0,	2.0],  # not in use, no o2
    ['stau',	1.01,	1.0,	1.05],
    ['sig',		0.01,	0.01,	0.01],  # not in use
    ['vh2o',	0.0,	-1.0,	1.0], 
    ['vo2',		0.0,	0.0,	0.0],    # not in use
    ['vstar',	0.0,	-1.0,	1.0],
    ['a',		1.0,	0.98,	1.03],
    ['b',		1.0,	1.0,	1.0],   # not in use below
    ['c',		1.0,	1.0,	1.0],
    ['d',		1.0,	1.0,	1.0],
    ['e',		1.0,	1.0,	1.0],
    ['f',		1.0,	1.0,	1.0]
    ],dtype=None)
    
    parnames = param_arr[:,0]
    params   = param_arr[:,1].astype('float')
    par_lo   = param_arr[:,2].astype('float')
    par_hi   = param_arr[:,3].astype('float')
    
    bounds = []
    for i in range(len(par_lo)):
    	bounds.append([par_lo[i],par_hi[i]])
    	
    return parnames, params, bounds
    

def lorentz(x,x0,gamma):
	fxn = (1/np.pi) * (0.5 * gamma) / ( (x-x0)**2 + (0.5*gamma)**2 )
	return fxn/sum(fxn)

def func(params, data,  unc, v, stell, h2o, o2,mode='sum'):
    """
    Minimization function
    """
    t_h2o, t_o2, stau, sig, vel_h2o, vel_o2, vel , a,b,c,d,e,f= params
    # Interpolate on shifted grid
    tck      = interpolate.splrep(v[::-1]*(1+vel/3.0e5),stell[::-1]**stau, s=0)
    tck_h2o  = interpolate.splrep(v[::-1]*(1+vel_h2o/3.0e5),h2o[::-1]**t_h2o, s=0)
    tck_o2   = interpolate.splrep(v[::-1]*(1+vel_o2/3.0e5),o2[::-1]**t_o2, s=0)
    # Multiply by line for continuum
    coeff = (a,a,a,a,a,a)
    knots = np.linspace(v[-1],v[0],len(coeff))
    cont  = BSpline(knots,coeff,2,extrapolate=True)
    # Broaden TAPAS
    #psf   = gaussian(v,v[int(len(v)/2.)],sig)
    psf   = lorentz(v,v[int(len(v)/2.)],sig)
    pad_h2o = np.pad(interpolate.splev(v,tck_h2o,der=0),\
    	pad_width=(200,200),mode='constant',constant_values=(1,1))
    wide_h2o = fftconvolve(pad_h2o,psf,'same')
	# Multiply Everything
    model = interpolate.splev(v,tck_o2,der=0)* \
            interpolate.splev(v,tck,der=0)* \
            wide_h2o[200:1200]
    model[np.where(np.isnan(model))] = 1e-10
    model_final = cont(v) * model #fftconvolve(model,psf,'same')	
    model_final[np.where(model_final <= 0)] = 1e-10
    if mode == 'sum':
    	return np.sum(((np.log(model_final) - data)**2)/unc)
    elif mode =='model':
    	return np.log(model_final)
	




if __name__ == '__main__':
    # User chosen things
    ifold           = 0  #folder index to work on
    icent           = 0
    
    # Load linecents
    f = np.loadtxt('isolated_lines.txt')
    linecenters,indicents = f[:,0],f[:,1]
    center = linecenters[icent]
    dk = 5.0
    
    # START
    so.ftsfolders   = glob.glob('../DATA/FTS/*')
    fnames = []

    # Load for each folder
    folder = so.ftsfolders[ifold]
    so.ftsday = folder
    set_up(so,nfiles='all',stelmod='kurucz_irr')

    # Get i0,i1
    i1 = np.where(so.fts.v > (center - dk))[0][-1]
    i0 = np.where(so.fts.v < (center + dk))[0][0]

    spec = so.fts.s[:,i0:i1]
    fnames = so.fts.fnames
    print folder

    nightlist = np.array([folder]*300)

    # CREATE FLATTENED DATA
    v = so.fts.v[i0:i1]
    muspecs = np.mean(spec,axis=1)
    igood = np.where(muspecs > 3*np.std(muspecs))[0]
    spec = spec[igood]
    goodlist = nightlist[igood]
    fnames = np.array(fnames)[igood]
    nspec = len(igood)
    t_all = so.fts.t_all[igood]

    flatspec = fit_continuum0(v,spec,goodlist)
    vflat = flatspec[0]
    sflat = flatspec[1]

    # Sub Set
    isub = np.arange(21000,22000)
    
    # STorage arrays
    deg_cont = 3
    
    pnames,params,bounds = define_pararr()                             

    residuals = []
    fit_eph = np.zeros(nspec)
    act_eph = np.zeros(nspec)
    chi2    = np.zeros(nspec)
    param_final = np.zeros((nspec,len(pnames)))
    param_std   = np.zeros((nspec,len(pnames)))


    #STELLAR MODEL
    ssub = np.where( (so.stel.v < vflat[isub[0]]) & (so.stel.v > vflat[isub[-1]]))[0]
    int_model = interp1d(so.stel.v[ssub],so.stel.s[ssub],kind='linear',
                             bounds_error=False,fill_value=so.stel.s[isub][-1])
    stel_mod  = int_model(vflat[isub])
    stel_mod  = stel_mod/np.max(stel_mod)

    # TAPAS
    ssub = np.where( (so.tap.v < vflat[isub[0]]) & (so.tap.v > vflat[isub[-1]]))[0]
    tck_tap = interpolate.splrep(so.tap.v[ssub],so.tap.h2o[ssub], s=0)
    h2o_mod = interpolate.splev(vflat[isub],tck_tap,der=0)
    h2o_mod[np.where(h2o_mod < 0)[0]]= 0

    tck_tap = interpolate.splrep(so.tap.v[ssub],so.tap.o2[ssub], s=0)
    o2_mod  = interpolate.splev(vflat[isub],tck_tap,der=0)
    o2_mod[np.where(o2_mod < 0)[0]]= 0

    allfinal = np.zeros((len(fnames),len(isub)))
    residuals= np.zeros((len(fnames),len(isub)))
     
    for i, fname in enumerate(fnames):
        fspec = sflat[i][isub]
        fspec[np.where(fspec <= 0)] = 1e-10
        fspec = np.log(fspec)
    	
        # FIT SETUP
        unc = 0.001* np.sqrt(np.abs(np.exp(fspec)))
        unc[0:2] = 10.0
        unc[len(unc)-3:] = 10.0
        unc[np.where(fspec < -8)] = 10.0
        unc = (1/np.exp(fspec)) * unc # propogate to log
            
        out = opt.minimize(func,params,\
        	args=(fspec, unc, vflat[isub], stel_mod, h2o_mod, o2_mod,'sum'),\
        	method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 

    	#Store
    	so.cc.message = out['message']
    	so.cc.nit     = out['nit']
    	so.cc.success = out['success']
    	so.cc.k       = out['x']

        params = out['x']
        
        # Save Residuals
        param_final[i]  = out['x'] #bestp
        param_std[i]    = 0 #out[1] #stdp
        model           = func(params,fspec, unc, vflat[isub],stel_mod,h2o_mod,o2_mod,'model')
        residuals[i]    = model - fspec
        fit_eph[i]      = params[6]
        act_eph[i]      = so.eph.vels[np.where(so.eph.fnames == fname)[0]][0]
        chi2[i]         = np.sum(np.array(residuals)**2)
		
        # get spec_proc
        stell_fit = func(params,fspec, unc, vflat[isub],stel_mod,h2o_mod*0+1,o2_mod*0+1,'model')
        spec_proc = fspec-stell_fit
        
        allfinal[i] = spec_proc

        print i, ' ', fname

        # Save stellar subtracted spectrum, put in header fit results
        #save_spectrum('p' + fname,t_all[i],params,pnames,param_std[i],vflat[isub],\
        #              spec_proc,residuals[i])


plt.figure()
plt.plot(vflat[isub],model,'k--',label='model')
plt.plot(vflat[isub],fspec,label='data')
plt.legend(loc='best')
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.ylim(-5,0.1)

    
mu_res = np.sum(np.abs(residuals[:,10:990]),axis=1) # sum of residuals
keepthese = np.where(mu_res < (np.mean(mu_res) + np.std(mu_res)))[0]
       
plt.figure()
for i in range(len(fit_eph)):
	if i in keepthese:
		plt.plot(vflat[isub],allfinal[i]*(1/param_final[i][0]))
		
		
# Average specs together to get telluric 0
alltaus = np.array([1/param_final[keepthese][:,0]]).T
tell0 = np.mean(allfinal[keepthese]*alltaus,axis=0)
tell0_std = np.std(allfinal[keepthese]*alltaus,axis=0)
		    
# plot tell_0 and tapas
params[0] = 1.0 # dont apply tau
plt.figure()
plt.plot(vflat[isub],np.log(stel_mod)-0.04,label='Stellar Model Fit')

tap_shift = func(params,fspec, unc, vflat[isub],stel_mod*0+1.0,h2o_mod,o2_mod*0+1,'model')
plt.plot(vflat[isub],tell0,label='Telluric 0')
plt.plot(vflat[isub],tap_shift,'k--',label='TAPAS')

plt.plot(vflat[isub],tap_shift-tell0 + 0.05,'green',label='Residuals')

plt.xlabel('Wavenumber')
plt.ylabel('Absorbance')
plt.title('Telluric 0 compared to TAPAS')

plt.legend(loc='best')





