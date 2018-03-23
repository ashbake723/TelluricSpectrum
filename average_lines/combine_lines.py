import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import splev, BSpline
import os,sys
import astropy.io.fits as fits
import glob
import scipy
from scipy import interpolate
plt.ion()

# Load files
fnames = glob.glob('./SavedLines/*fits')

# Storage arrays
taus = np.zeros(len(fnames))
specs= np.zeros((len(fnames),1000))
specs_wing = np.zeros((len(fnames),1000))

def bindat(x,y,nbins):
    """
    Bin Data
        
    Inputs:
    ------
    x, y, nbins
    
    Returns:
    --------
    arrays: bins, mean, std  [nbins]
    """
    # Create bins (nbins + 1)?
    n, bins = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    
    # Calculate bin centers, mean, and std in each bin
    bins = (bins[1:] + bins[:-1])/2
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
        
    # Return bin x, bin y, bin std
    return bins, mean, std


for i,fname in enumerate(fnames):
    # Load files
    f = fits.open(fname)
    pardic = f[1].data
    v,s    = f[2].data['v'],f[2].data['s']
    hdr    = f[1].header
    f.close()
    
    # Plot
    #plt.plot(v-v[len(v)/2],s)
    
    # Resample s, take subset to remove noisy wings for matching
    v_finer  = np.linspace(v[430],v[230],1000)
    tck      = interpolate.splrep(v[::-1][200:450],s[::-1][200:450])
    spec      = interpolate.splev(v_finer,tck,der=0)
    
    # Resample s but don't take subset (for fitting the wings)
    v_finer2  = np.linspace(v[500],v[150],1000)
    tck2      = interpolate.splrep(v[::-1][100:550],s[::-1][100:550])
    spec2     = interpolate.splev(v_finer2,tck2,der=0)
 
    # store resampled spectrum
    specs[i]      = spec
    specs_wing[i] = spec2
    
    # Save Taus
    itau = np.where(pardic['pnames'] =='taus')[0]
    taus[i] = pardic['params'][itau]


# Define cc fxn..also minimize residuals and fit for tau
def cc_func(p,x,y,ref,mode='sum'):
    dk, k, tau = p
    x_new    = x * dk + k
    tck      = interpolate.splrep(x_new,y*tau, s=0)
    s_new    = interpolate.splev(x,tck)
    #print p
    if mode == 'sum':
        return np.sum((ref - s_new)**2)*0.1
    else:
        return s_new


# Run
ref = specs[2]
nspec  = 118
#bounds =  ((0.5,1.5), (-10,10), (0.1,5.0))
bounds =  ((0.7,1.3), (-0.1,0.1), (0.1,3.0))
dks    = np.zeros(nspec)
ks     = np.zeros(nspec)
taus   = np.zeros(nspec)
s_all  = np.zeros((nspec,len(ref)))
s_all_wing  = np.zeros((nspec,len(specs_wing[0])))

for i in range(0,nspec):
    #x, y = np.arange(1000)-500, specs[i]
    x, y = v_finer-v_finer[len(v_finer)/2], specs[i]
    y[np.where(y<0)[0]] = 1e-8 
    out = scipy.optimize.minimize(cc_func,(1.0,-0.0007,0.5),
           args=(x,y,ref),
           bounds=bounds,
           method="TNC")

    p = out['x']
    dks[i], ks[i], taus[i] = p[0], p[1], p[2]
    s_all[i,:] = cc_func(p,x,specs[i],ref,mode='func') 
    s_all_wing[i,:] = cc_func(p,v_finer2-v_finer2[len(v_finer2)/2],
                               specs_wing[i],ref,mode='func')
#*** need to rematch s_all_wing with best fit found with subset
# because diff x mapping makes it not possible to just apply p
# maybe if i fit in actual cm^-1 units it will work?
all_res = s_all - np.mean(s_all,axis=0) # store residuals
mean_s  = np.mean(s_all,axis=0)
std_s   = np.std(s_all,axis=0)

# Plott all spec with mean
plt.figure()
for i in range(nspec):
    plt.plot(x,s_all[i],'r',alpha=.2)

plt.plot(x,np.mean(s_all,axis=0),'k--')
plt.xlabel('Wavenumber Index (arb)')
plt.ylabel('Scaled Absorbance')
plt.title('%s Isolated Spectral Lines, Matched' %nspec)

# Plot residuals with binning
plt.figure()
for i in range(nspec):
    plt.plot(x,all_res[i],'r',alpha=.2)

#out = bindat(np.round(np.arange(1000*nspec)/nspec)-500,all_res.T.flatten(),100)
#plt.errorbar(*out,fmt='o',zorder=10)
plt.xlabel('Wavenumber Index (arb)')
plt.ylabel('Residuals')
plt.title('%s Residuals from mean of all aligned, Isolated Spectral Lines' %nspec)


#########################################
# FIT AND ANALYZE
#########################################

# Do a PCA
# Start PCA
A = s_all.T
M = (A-np.mean(A.T,axis=1)).T
[latent,coeff] = np.linalg.eig(np.cov(M))  #latent: eigenvalue, coeff: eig vectors
score = np.dot(coeff.T,M)

plt.figure()
plt.plot(score[0]) # basically the mean, which is good
plt.plot(score[1])
plt.xlabel('index')
plt.title('First two principle components')
plt.ylabel('Absorbance')

#########################################
# Fit wings to smooth it, then save for fitting
# tried spline, not worth it, just use voigt
# Fit with voigt wing
from astropy import modeling

def voigt_fit(p, x, data,err,mode='lstq'):
    """
    """
    S, gamma, sigma, kappa0 = p
    
    V = modeling.functional_models.Voigt1D(kappa0,S,gamma,sigma)    

    if mode=='lstq':
        return np.sum(((V(x) - data)**2))
    else:
        return V(x)

params_start = (.4, 0.1, 0.001, 0.0)
bounds       = ((.1,1.0), (0.01,0.3), (0.0001,0.2), (-0.1,0.1))
out = opt.minimize(voigt_fit,params_start,\
        args=(x,mean_s,std_s),\
        method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 


params = out['x']
V = voigt_fit(params, x, mean_s, std_s, mode='out')

plt.figure()
plt.plot(x,V)
plt.plot(x,mean_s)
plt.plot(x,V-mean_s)

################
# Try most complex fit from the Hartmann Line Profile code
################

sys.path.append('../')
from hartmann_line_profiles import PROFILE_HT

def func(p,x,mean_s,mode='lstq'):
    #      sg0     : Unperturbed line position in cm-1 (Input).
    #      GamD    : Doppler HWHM in cm-1 (Input)
    #      Gam0    : Speed-averaged line-width in cm-1 (Input).       
    #      Gam2    : Speed dependence of the line-width in cm-1 (Input).
    #      anuVC   : Velocity-changing frequency in cm-1 (Input).
    #      eta     : Correlation parameter, No unit (Input).
    #      Shift0  : Speed-averaged line-shift in cm-1 (Input).
    #      Shift2  : Speed dependence of the line-shift in cm-1 (Input)       
    #      sg      : Current WaveNumber of the Computation in cm-1 (Input).
    sg0,GamD,Gam0,Gam2,Shift0,Shift2,anuVC,eta, S = p
    V = PROFILE_HT(sg0,GamD,Gam0,Gam2,Shift0,Shift2,anuVC,eta,x)
    if mode=='lstq':
        return np.sum(((200*S * V[0]/sum(V[0]) - mean_s)**2))
    else:
        return 200*S * V[0]/sum(V[0])

params_start = (-0.03, 0.0002, 0.14, 0.1, 0.001,0.0,0.01,0.01, 0.01)
bounds       = ((-0.1,0.1), # sg0
                (1e-5,0.01), # gamD
                (0.01,0.5),  # gam0
                (0.001,0.5), # gam2
                (0,0.5),    # anuVC
                (-0.1,0.1), # eta
                (0,1e-3), # Shift0
                (0,1e-3), # Shift2
                (0.1,1.0)) # line strength
                
out = opt.minimize(func,params_start,\
        args=(x,mean_s),\
        method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 

params = out['x']
V = func(params, x, mean_s,mode='out')

plt.figure()
plt.plot(x,V,'k--')
plt.plot(x,mean_s)
plt.plot(x,V-mean_s,'r')
plt.plot(x,x*0,'gray')
plt.xlabel('Wavenumber (shifted, cm^-1)')
plt.ylabel('Absorbance')


# ################
# Fit just the wings
################
bounds2       = ((-0.1,0.1), # sg0
                (1e-5,0.01), # gamD
                (0.01,0.5),  # gam0
                (0.001,0.5), # gam2
                (0,0.5),    # anuVC
                (-0.1,0.1), # eta
                (0,1e-3), # Shift0
                (0,1e-3), # Shift2
                (0.1,1.0)) # line strength

def func_err(p,x,mean_s,mode='lstq'):
    #      sg0     : Unperturbed line position in cm-1 (Input).
    #      GamD    : Doppler HWHM in cm-1 (Input)
    #      Gam0    : Speed-averaged line-width in cm-1 (Input).       
    #      Gam2    : Speed dependence of the line-width in cm-1 (Input).
    #      anuVC   : Velocity-changing frequency in cm-1 (Input).
    #      eta     : Correlation parameter, No unit (Input).
    #      Shift0  : Speed-averaged line-shift in cm-1 (Input).
    #      Shift2  : Speed dependence of the line-shift in cm-1 (Input)       
    #      sg      : Current WaveNumber of the Computation in cm-1 (Input).
    sg0,GamD,Gam0,Gam2,Shift0,Shift2,anuVC,eta, S = p
    V = PROFILE_HT(sg0,GamD,Gam0,Gam2,Shift0,Shift2,anuVC,eta,x)
    if mode=='lstq':
        return np.sum((((200*S * V[0]/sum(V[0]) - mean_s)**2)))
    else:
        return 200*S * V[0]/sum(V[0])

params_start = params # Use last fit for params
   
out_left = opt.minimize(func_err,params_start,\
        args=(x[0:500],mean_s[0:500]),\
        #sqrt(np.sqrt(std_s[0:500]))),\
        method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 

out_right = opt.minimize(func_err,params_start,\
        args=(x[450:],mean_s[450:]),\
        #sqrt(np.sqrt(std_s[0:500]))),\
        method="SLSQP",bounds=bounds,options={'maxiter' : 100}) 

params_left = out_left['x']
params_right = out_right['x']

V_left  = func_err(params2, x[0:500], mean_s[0:500], mode='out')
V_right = func_err(params2, x[400:], mean_s[400:], mode='out')

plt.figure()
plt.plot(x,mean_s,zorder=-1)
plt.plot(x[0:400],V_left[0:400],'k--')
plt.plot(x[500:],V_right[100:],'k--')
plt.plot(x[0:400],V_left[0:400] - mean_s[0:400],'r--')
plt.plot(x[500:],V_right[100:] - mean_s[500:],'r--')
plt.plot(x,x*0,c='gray')

# analyze plot and find index where want to keep


# Save Functions if happy with fits


