# should be run on a daily basis (specify ifold) and all spectra want to include should have been run by prefit
import astropy.io.fits as fits
import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as opt
from scipy.signal import fftconvolve
from scipy.interpolate import splev, BSpline, interp1d
from astropy.time import Time
import glob, ephem, os, sys
import argparse

plt.ion()

from prefit_functions import *

sys.path.append('../read_opus/')
import opus2py as op

sys.path.append('../tools/')
from hitran_V3 import *
from functions import *

from objects import load_object
so = load_object('analyze.cfg') # load storage object and load config file

so,success = setup_data(so)  


#################################################################
####################### Load File Functions #####################
#################################################################

def load_prefit(filename):
	"""
	load all fit files in prefit output folder
	
	Loads night chosen from analyze.cfg
	"""
	f = np.loadtxt(filename)
	icent, dnu, vel, fval = f[:,0],f[:,2],f[:,3], f[:,4]
	t      = f[:,1][0]
	airmass= f[:,5][0]
	eph    = f[:,6][0]

	return icent,dnu,vel,fval,t,airmass,eph


#################################################################
####################### Prep Saving FITTING #####################
#################################################################
print 'working on day %s' %so.fts.name

loadfiles = glob.glob(so.run.load_folder + so.fts.name + '/' + so.run.save_prefix + '*.txt')

so.run.savename = so.run.save_prefix + '_%s.txt'%so.fts.name
so.run.savepath = so.run.save_folder + so.run.savename

if not os.path.exists(so.run.save_folder):
	os.system('mkdir %s' %(so.run.save_folder + so.fts.name))

if not os.path.isfile(so.run.savepath):
    fsave = open(so.run.savepath,'w')
    fsave.write('#specnum,t,airmass,eph,med_dnu,std_dnu,med_vel,std_vel\n')
    fsave.write('#folder = %s\n' %so.fts.name)
    fsave.close()
elif so.run.overwrite:
	print 'WARNING: Path already exists, you chose to overwrite existing file'
	fsave = open(so.run.savepath,'w')
	fsave.write('#specnum,t,airmass,eph,med_dnu,std_dnu,med_vel,std_vel\n')
	fsave.write('#folder = %s\n' %so.fts.name)
	fsave.close()
else:
	# raise a warning 
	print 'WARNING: Path already exists, you chose to append'


#################################################################
####################### START FITTING ###########################
#################################################################

for i,filename in enumerate(loadfiles):
	# load file contents
	icent,dnu,vel,fval,t,airmass,eph =  load_prefit(filename)
	
	# save the spectrum number
	specnum = filename.split('/')[-1].split('_')[-1][0:4]
	eph = so.eph.vels[int(specnum)] # ephs should be ordered
	
	# Median and std of iodine velocities
	med_dnu = np.median(dnu[np.where(icent>20)[0]]) ##check this where
	std_dnu = np.std(dnu[np.where(icent>20)[0]])
	
	# Median and std of stellar velocities
	med_vel = np.median(vel[np.where(vel > -9.9)[0]])
	std_vel = np.std(vel[np.where(vel > -9.9)[0]])
	
	fxn_val = np.median(fval)
	
	# If med_vel == 0
	if np.isnan(med_vel):
		med_vel = -9.9
		std_vel = -9.9
		med_dnu = -9.9
		std_dnu = -9.9
		fxn_val = -9.9
	
	# Save to file
	fsave = open(so.run.savepath,'a')
	fsave.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %(specnum,t,airmass,eph,med_dnu,std_dnu,med_vel,std_vel,fxn_val))
	fsave.close()

