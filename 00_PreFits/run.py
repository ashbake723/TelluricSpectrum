# should be run on a day basis for each spectrum (so much specify ifold and ispec)
import astropy.io.fits as fits
import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as opt
from scipy.signal import fftconvolve
from scipy.interpolate import splev, BSpline, interp1d
from astropy.time import Time
import glob, ephem, os, sys
import argparse
from distutils.util import strtobool
from joblib import Parallel, delayed
import multiprocessing

plt.ion()
sys.path.append('../tools/')

from objects import load_object
from functions import *




#################################################################
####################### START FITTING ###########################
#################################################################


################
def func0(p,so,knots,mode='sum'):
    """
    Fit for h2o shift, stellar shift
    """
    dnu,vel = p[0:2]
    coeff   = p[2:] 
        
    # Iodine
    tck_iod = interpolate.splrep(so.cal.v[::-1]*(1-dnu/3.0e5),so.cal.iodine[::-1], k=2, s=0)
    iod_int = interpolate.splev(so.cal.v,tck_iod,der=0)
    
    # Stellar
    tck      = interpolate.splrep(so.cal.v[::-1]*(1-vel/3.0e5),so.cal.stel[::-1], k=2,s=0)
    stel_int = interpolate.splev(so.cal.v,tck,der=0)

    # Continuum
    spl = BSpline(knots, coeff, 2)
    continuum = spl(so.cal.v)
    #continuum_slope = (cont[0] - cont[1])/(so.cal.v[0] - so.cal.v[-1])
    #continuum       = continuum_slope * (so.cal.v - so.cal.v[0])  + cont[0]

    model = (stel_int + iod_int + continuum) * so.cal.mask
        
    if mode =='sum':
        return np.sum(((model - so.cal.mask*so.cal.s)**2))
    elif mode == 'get_model':
        return model,iod_int,continuum


# Test with 17281.3
def run_fit(so):	
	# Delete sections with strong tellurics
	lams = 1e7/so.cal.v
	if np.any([np.all([(lams > 813.0),(lams < 833.0)],axis=0)]):
		idel = np.where((lams > 813.0) & (lams < 833.0))[0]
	
	if np.any([np.all([(lams > 760.1),(lams < 773.1)],axis=0)]):
		idel = np.where((lams >760.1) & (lams < 773.1))[0]
	
	if np.any([np.all([(lams > 686.0),(lams < 741.0)],axis=0)]):
		idel = np.where((lams > 686.0) & (lams < 741.0))[0]
		
	if np.any([np.all([(lams > 589.5),(lams < 596.5)],axis=0)]):
		idel = np.where((lams > 589.5) & (lams < 596.5))[0]
	
	if np.any([np.all([(lams > 628.0),(lams < 643.0)],axis=0)]):
		idel = np.where((lams > 628.0) & (lams < 643.0))[0]
	
	try:
		so.cal.v = np.delete(so.cal.v,idel)
		so.cal.s = np.delete(so.cal.s,idel)
		so.cal.iodine = np.delete(so.cal.iodine,idel)
		so.cal.stel = np.delete(so.cal.stel,idel)
	except NameError:
		idel = 0

	# Skip if all tellurics (no wavelengths defined)
	if len(so.cal.v)==0.0:
		return None,None

	so.cal.mask = np.ones(len(so.cal.v))
	# Fill in mask with zeros 1cm-1 around strong and weak water and o2 lines 
	for i,cent in enumerate(so.hit.hit_dic['linecenter']):
		if so.hit.hit_dic['S'][i] > 5e-25:
			so.cal.mask[np.where((so.cal.v > cent - 1.2) & (so.cal.v < cent + 1.2))[0]] = 0.0
		elif so.hit.hit_dic['S'][i] > 5e-26:
			so.cal.mask[np.where((so.cal.v > cent - 0.3) & (so.cal.v < cent + 0.3))[0]] = 0.0


	# Set up continuum
	knots = np.linspace(so.cal.v[-1],so.cal.v[0],4)
	knots = np.concatenate((np.array([so.cal.v[-1] - 87.5]), knots, np.array([so.cal.v[0] + 87.5])))

	# set up fit
	pnames = ['dnu','vel','c1','c2','c3']
	pstart = [0,     0,   0.0 ,  0.0, 0.0]
	bounds = [[-0.01,0.01], [-1.0,1.0], [-0.1,0.1], [-0.1,.1], [-0.1,0.1]]
	
	if lams[0] > 640:
	    bounds = [[0,0], [so.eph.vels[so.run.ispec] - 0.1,so.eph.vels[so.run.ispec]+0.1],[-0.1,0.1], [-0.1,.1], [-0.1,0.1]]
	
	print 'starting fit'
	out  = opt.minimize(func0,pstart,\
                args=(so,knots),\
    			method="TNC",bounds=bounds,options={'maxiter' : 1000}) 

	# Store Outputs
	pout = out['x']
	so.cal.dnu, so.cal.vel = pout[0:2]

	model, iod_int, continuum = func0(out['x'],so,knots,mode='get_model')

	return model, out



def fit_wrapper(icent,ispec):
	#################################################################
	#######################  SETUP NIGHT  ###########################
	#################################################################
	so = load_object('prefit.cfg') # load storage object and load config file
	so.run.icent = icent + 3
	so.run.ispec = ispec
	
	so,success = setup_data(so)  


	print 'working on day %s' %so.fts.fname

	# Setup Night and Savefile name
	so.run.savename = '%s_%s.txt' %(so.run.save_prefix,so.fts.specnum)
	so.run.savepath = so.run.save_folder + so.fts.name + '/' + so.run.savename

	if not success:
		print 'spectrum has nans - cant work like this'
		fsave = open(so.run.savepath,'a')
		fsave.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %(so.run.icent,so.fts.t,-9.9,-9.9,-9.9,so.run.airmass,so.eph.vels[int(so.fts.specnum)]))
		fsave.close()
	else:
		so.run.centv = so.run.centers[so.run.icent]
		so.cal.vlo = so.run.centv - 175.0
		so.cal.vhi = so.run.centv + 175.0
		so.hit.hit_dic,so.cal.v,so.cal.s,tap,so.cal.stel,v_edges,so.cal.iodine = get_subspec(so,threshold=so.cal.Smin,get_hitran=True,dk=175.0)

		####################### START FITTING ###########################
		model,out = run_fit(so)
	
		# Assess if it's a bad fit (if near boundary or fit failed)
	
		plot_on = False
		if plot_on == True:
			if np.all(model != None):
				plt.plot(so.cal.v,so.cal.s)
				plt.plot(so.cal.v,model)
				plt.plot(so.cal.v,model-so.cal.s*so.cal.mask)
		
	
		#################################
		# Save file
		if np.all(model != None): 
			fsave = open(so.run.savepath,'a')
			fsave.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %(so.run.icent,so.fts.t,so.cal.dnu,so.cal.vel,out['fun'],so.run.airmass,so.eph.vels[int(so.fts.specnum)]))
			fsave.close()
		else:
			fsave = open(so.run.savepath,'a')
			fsave.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %(so.run.icent,so.fts.t,-9.9,-9.9,-9.9,so.run.airmass,so.eph.vels[int(so.fts.specnum)]))
			fsave.close()
	

if __name__=='__main__':
	# Setup Night and Savefile name
	so = load_object('prefit.cfg') # load storage object and load config file
	if len(sys.argv) > 1: 
		ispec = int(sys.argv[1]) - 1  # subtract 1 bc 0 isnt valid job id

	so,success = setup_data(so)  

	so.run.savename = '%s_%s.txt' %(so.run.save_prefix,so.fts.specnum)
	so.run.savepath = so.run.save_folder + so.fts.name + '/' + so.run.savename

	# Create folder if not exists
	if not os.path.exists(so.run.save_folder + so.fts.name + '/'):
		os.system('mkdir %s' %(so.run.save_folder + so.fts.name))
			
	if not os.path.isfile(so.run.savepath):
		fsave = open(so.run.savepath,'w')
		fsave.write('#icent\tjd\tdnu\tvel\tfuncval\tairmass\n')
		fsave.write('#specname = %s\n' %so.fts.fname)
		fsave.close()
	elif so.run.overwrite:
		# raise a warning 
		print 'WARNING: Path already exists, you chose to overwrite existing file'
		fsave = open(so.run.savepath,'w')
		fsave.write('#icent\tjd\tdnu\tvel\tfuncval\tairmass\n')
		fsave.write('#specname = %s\n' %so.fts.fname)
		fsave.close()
	else:
		print 'WARNING: Path already exists, you chose to append'

	# Fit
	num_cores = multiprocessing.cpu_count()
	inputs    = range(len(so.run.centers[3:]))
	if so.fts.nspec > num_cores:
		nloop = int(np.ceil(so.fts.nspec/num_cores))
		input_array = [inputs[i:i + nloop] for i in xrange(0, len(inputs), nloop)]
	if input_array:
		for inputs in input_array:
			results   = Parallel(n_jobs=num_cores)(delayed(fit_wrapper)(icent,ispec) for icent in inputs)
	else:
		results   = Parallel(n_jobs=num_cores)(delayed(fit_wrapper)(icent,ispec) for icent in inputs)
