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
import glob
import scipy 
sys.path.append('../')
from hitran_V2 import *
from astropy import modeling


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

def get_time(time,date):
	"""
	Given the time and date of the FTS file, return sjd as Time object
	"""
	sts = date[6:] + '-' + date[3:5] + '-' + date[0:2] + ' ' + time[0:12]
	gmtplus = float(time[18])
	sjd = Time(sts, format='iso', scale='utc').jd - gmtplus/24.0 # subtract +1 hr
	return sjd


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

    # STELLAR MODEL
    # http://kurucz.harvard.edu/stars/sun/
    stellar = fits.open('../DATA/STELLAR/irradthuwn.fits') #irradiance
    so.stel.v = stellar[1].data['wavenumber'][0:1200000] 
    so.stel.s = stellar[1].data['irradiance'][0:1200000]
    stellar.close()

    # EPHEM
    vv = fits.open('../DATA/EPHEM/%s_perspec.fits' %so.ftsday[12:],dtype=str)
    ephem= vv[1].data
    so.eph.fnames = ephem['specname']
    so.eph.vels   = ephem['velocity']
    vv.close()


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


    
def voigt_profile(x, x0, S, gamma, sigma):
    """
    Using    
    """
    V = modeling.functional_models.Voigt1D(x0,S,gamma,sigma)
    return V(x)





if __name__ == '__main__':
    so.ftsfolders   = glob.glob('../DATA/FTS/*')
    fnames = []
    
    ifold=0
    so.ftsday = so.ftsfolders[ifold]
    load_data(so,nfiles='all',stelmod='kurucz_irr')
    hitran_data = read_hitran2012_parfile('HITRAN/h2o_o2_9000_20000.par')



####

# Find isolated lines
cent = hitran_data['linecenter']
gam  = hitran_data['gamma-air']
stre = hitran_data['S']
plt.plot(cent,gam,'o')
plt.hist(np.diff(cent)/gam[1:])

sig = 4
left_dis = (cent - np.roll(cent,1))/gam > sig
right_dis = (np.roll(cent,-1) - cent)/gam > sig

check = np.where(right_dis & left_dis & (stre > 1e-24) & (hitran_data['M']==1))


plt.figure()   
plt.plot(so.fts.v,so.fts.s[0])
plt.plot(cent[check],stre[check],'o')

# Eliminate weak lines 1e-27 and weaker and redo
delme = np.where(stre < 5e-27)[0]
cent= np.delete(cent,delme)
gam  = np.delete(gam,delme)
stre = np.delete(stre,delme)
M    = np.delete(hitran_data['M'],delme)

sig = 5
left_dis = (cent - np.roll(cent,1))/gam > sig
right_dis = (np.roll(cent,-1) - cent)/gam > sig

check = np.where(right_dis & left_dis & (stre > 1e-24) & (M==1))[0]

# Pick good lines to keep (this is from viewing them individually)
keepme = np.array([102,101,100,99,98,96,95,90,88,
                   87,86,83,81,80,77,
                   46,21,20])

check = check[keepme]

# Save keepmes into text file for fitting later
keep_cents = cent[check]
keep_indices = check
np.savetxt('isolated_lines.txt',zip(keep_cents,keep_indices),\
             header = 'wavelength   index from hitran file',
             fmt=('%f','%i'))



# Cut out each line for averaging
snippets = {}
stel_snip= {}

for i in range(0,len(check)):
    crop = np.where( (so.fts.v > cent[check[i]] - 0.7) & (so.fts.v < cent[check[i]] +0.7) )[0]
    snippets[str(i)] = (so.fts.v[crop],1.0 - so.fts.s[0][crop]/np.max(so.fts.s[0][crop]))

    # Stellar part - interpolate
    crop_s = np.where( (so.stel.v > cent[check[i]] - 0.7) & (so.stel.v < cent[check[i]] +0.7) )[0]
    if len(crop_s) > 0:
        stel_snip[str(i)] = (so.stel.v[crop_s],1.0 - so.stel.s[crop_s]/np.max(so.stel.s[crop_s]))
    else:
        stel_snip[str(i)] =  0


# Params: sig, x0, y0
param_out = np.zeros((len(check),6))
minf      = np.zeros(len(check))
p_names = np.array(['sigma', 'gamma',  'dx',    'dy',  'A',    'vel'])
par_lo  =  np.array([0.0001,  0.001,   -1e-1,   0.0,   0.0,     -1.0 ])
p_st  = np.array([   0.03,    0.1 ,    0.0  ,    0.0,   0.3,    0.16  ])
par_hi  = np.array([ 0.5,     0.3,      1e-1,    0.0,  10.0,    1.0])
bounds = []
for j in range(len(par_lo)):
	bounds.append([par_lo[j],par_hi[j]])


def func(p, x, data,unc,intstel,mode='sum'):
    sig, gam, dx, dy, A, vel = p
    # Make gaussian
    gaus = voigt_profile(x, x[int(len(x)/2.0)] + dx, A, gam, sig)* (1+dy)
    # Shift stellar spectrum
    stel = intstel(x * (1 + vel/3.0e5))
    if mode == 'sum':
    	return np.sum(((data - gaus - stel)**2)/unc)
    else:
    	return gaus , stel

fin_dat = {}
for i in range(0,18):#len(check)):
    v, snip = snippets[str(i)]
    
    stel = stel_snip[str(i)]
    intstel = interp1d(stel[0],stel[1],kind='linear',
                             bounds_error=False,fill_value=0)

    logsnip = np.log(1.0000001-snip)
    unc     = np.sqrt(1.0000001-snip)*0.1
    
    out = opt.minimize(func,p_st,\
    	args=(v,snip,unc,intstel),\
    	method="SLSQP",bounds=bounds,options={'maxiter' : 100})
    	
    # Save parameters
    param_out[i] = out['x']
    minf[i]      = out['fun']
    
    # Save data with stellar part removed
    gaus,stel = func(param_out[i],v,snip,unc,intstel,mode='model')
    plt.figure()
    plt.plot(v,snip,'b')
    plt.plot(v,snip - gaus - stel,'r')
    plt.plot(v,gaus + stel,'--',c='darkgray')
#    plt.plot(v,stel)
    plt.title(str(i)  +' or ' + str(keepme[i]))
    
    # Save data (center around 0, take out stellar, normalize)
    xcent = v[int(len(v)/2.0)] + param_out[i][2]
    fin_dat[str(i)] = (v - xcent,(1-snip)/(1-stel))
    fin_dat['r'+ str(i)] = (v - xcent,snip - gaus - stel)


p = param_out
s,g,x,y,a,vel = p[:,0],p[:,1],p[:,2],p[:,3],p[:,4],p[:,5]

# take final data array and center and normalize, remove bad ones
for i in range(18):
    x,y = fin_dat['r'+str(i)]
    plt.plot(x,y**(1/a[i]))

# Interpolate data, line up with a cross correlation, remove bad ones, avg
v_finer  = np.linspace(fin_dat['6'][0][-1],fin_dat['6'][0][0],10000)
tck      = interpolate.splrep(fin_dat['6'][0][::-1],fin_dat['6'][1][::-1])
ref      = interpolate.splev(v_finer,tck,der=0)

# Take subset
v_finer  = v_finer[3000:7000]
ref      = ref[3000:7000]

# Define cc fxn..also minimize residuals and fit for tau
def cc_func(p,x,y,v_finer,ref,mode='sum'):
    dk, tau = p
    x_new    = x * (1+dk)
    tck      = interpolate.splrep(x_new,y**tau, s=0)
    s_new    = interpolate.splev(v_finer,tck)
    print p
    if mode == 'sum':
        return np.sum((ref - s_new)**2)*0.1 + (-1*np.correlate(s_new,ref)[0])*0.001
    else:
        return s_new


# Run
bounds =  ((-0.1,0.1),(0.1,5.0))
dks    = np.zeros(len(check))
taus   = np.zeros(len(check))
s_all  = np.zeros((len(check),len(ref)))

for i in range(0,18):
    x, y = fin_dat[str(i)][0][::-1],fin_dat[str(i)][1][::-1]
    out = scipy.optimize.minimize(cc_func,(0.0,1.0),
           args=(x,y,v_finer,ref),
           bounds=bounds,
           method="SLSQP") # tried a few and TNC seemed best

    p = out['x']
    dks[i], taus[i] = p[0], p[1]
    s_all[i,:] = cc_func(p,x,y,v_finer,ref,mode='func')

plt.figure()
for i in range(18):
    plt.plot(v_finer,s_all[i])
    
#########################################

ref = np.mean(s_all,axis=0)

# Fit again but just centers
def cc_func2(dk,v_finer,s,ref,mode='sum'):
    x_new    = v_finer + dk/1000.0
    tck      = interpolate.splrep(x_new,s, s=0)
    s_new    = interpolate.splev(v_finer,tck)
    if mode == 'sum':
        return (-1*np.correlate(s_new,ref)[0])*0.01
    else:
        return s_new

dks    = np.zeros(len(check))
s_all2  = np.zeros((len(check),len(ref)))
for i in range(0,18):
    s   = s_all[i]
    out = scipy.optimize.minimize(cc_func2,(0.0),
           args=(v_finer,s,ref),
           method="SLSQP",bounds=[(-0.01,0.01)])

    dks[i] = out['x']
    s_all2[i,:] = cc_func2(dks[i],v_finer,s,ref,mode='func')


plt.figure()
for i in range(18):
    plt.plot(v_finer,s_all2[i,:])

plt.plot(v_finer,ref,'k',lw=3)







