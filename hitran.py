# -*- coding: UTF-8 -*-

from __future__ import print_function, division
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob

__authors__ = 'Nathan Hagen'
# edited by ashley baker
__license__ = 'MIT/X11 License'
__contact__ = 'Nathan Hagen <and.the.light.shattered@gmail.com>'
__all__     = ['lorentzian_profile', 'read_hitran2012_parfile', 'translate_molecule_identifier',
           'get_molecule_identifier', 'calculate_hitran_xsec']

## ======================================================
def lorentzian_profile(kappa, S, gamma, kappa0):
    '''
    Calculate a Lorentzian absorption profile.
    Parameters
    ----------
    kappa : ndarray
        The array of wavenumbers at which to sample the profile.
    S : float
        The absorption line "strength" (the integral of the entire line is equal to S).
    gamma : float
        The linewidth parameter of the profile.
    kappa0 : float
        The center position of the profile (in wavenumbers).
    Returns
    -------
    L : ndarray
        The sampled absorption profile.
    '''
    L = (S / pi) * gamma / ((kappa - kappa0)**2 + gamma**2)
    return(L)

## ======================================================
def lorentzian_asym(kappa, S, gamma, kappa0, kernel):
    '''
    Calculate a Lorentzian absorption profile.
    Parameters
    ----------
    kappa : ndarray
        The array of wavenumbers at which to sample the profile.
    S : float
        The absorption line "strength" (the integral of the entire line is equal to S).
    gamma : float
        The linewidth parameter of the profile.
    kappa0 : float
        The center position of the profile (in wavenumbers).
    Returns
    -------
    L : ndarray
        The sampled absorption profile.
    '''
    L = (S / pi) * gamma / ((kappa - kappa0)**2 + gamma**2)
    return(fftconvolve(L,kernel,'same'))

def kernel(kappa, sigma, a):  
    '''
    Calculate a convolution kernel that will asymmetrize things
    Parameters
    ----------
    kappa : ndarray
        The array of wavenumbers at which to sample the profile.
    sigma : integer, even
        Width of convolving kernel
    m : integer
        slope of line aka how asymmetric
    Returns
    -------
    K : ndarray
        The sampled kernel.
    '''
    x  = np.arange(len(kappa))
    K  = np.zeros(len(kappa))
    K[len(K)/2 - int(sigma/2.):len(K)/2 + int(sigma/2.)] = a**np.arange(sigma)
    
    K *= 1.0/sum(K)
    
    return K
 
## ======================================================
def read_hitran2012_parfile(filename):
    '''
    Given a HITRAN2012-format text file, read in the parameters of the molecular absorption features.
    Parameters
    ----------
    filename : str
        The filename to read in.
    Return
    ------
    data : dict
        The dictionary of HITRAN data for the molecule.
    '''

    if not os.path.exists:
        raise ImportError('The input filename"' + filename + '" does not exist.')

    if filename.endswith('.zip'):
        import zipfile
        zip = zipfile.ZipFile(filename, 'r')
        (object_name, ext) = os.path.splitext(os.path.basename(filename))
        print(object_name, ext)
        filehandle = zip.read(object_name).splitlines()
    else:
        filehandle = open(filename, 'r')

    data = {'M':[],               ## molecule identification number
            'I':[],               ## isotope number
            'linecenter':[],      ## line center wavenumber (in cm^{-1})
            'S':[],               ## line strength, in cm^{-1} / (molecule m^{-2})
            'Acoeff':[],          ## Einstein A coefficient (in s^{-1})
            'gamma-air':[],       ## line HWHM for air-broadening
            'gamma-self':[],      ## line HWHM for self-emission-broadening
            'Epp':[],             ## energy of lower transition level (in cm^{-1})
            'N':[],               ## temperature-dependent exponent for "gamma-air"
            'delta':[],           ## air-pressure shift, in cm^{-1} / atm
            'Vp':[],              ## upper-state "global" quanta index
            'Vpp':[],             ## lower-state "global" quanta index
            'Qp':[],              ## upper-state "local" quanta index
            'Qpp':[],             ## lower-state "local" quanta index
            'Ierr':[],            ## uncertainty indices
            'Iref':[],            ## reference indices
            'flag':[],            ## flag
            'gp':[],              ## statistical weight of the upper state
            'gpp':[]}             ## statistical weight of the lower state

    print('Reading "' + filename + '" ...')

    for line in filehandle:
        if (len(line) < 160):
            raise ImportError('The imported file ("' + filename + '") does not appear to be a HITRAN2012-format data file.')

        data['M'].append(uint(line[0:2]))
        data['I'].append(uint(line[2]))
        data['linecenter'].append(float64(line[3:15]))
        data['S'].append(float64(line[15:25]))
        data['Acoeff'].append(float64(line[25:35]))
        data['gamma-air'].append(float64(line[35:40]))
        data['gamma-self'].append(float64(line[40:45]))
        data['Epp'].append(float64(line[45:55]))
        data['N'].append(float64(line[55:59]))
        data['delta'].append(float64(line[59:67]))
        data['Vp'].append(line[67:82])
        data['Vpp'].append(line[82:97])
        data['Qp'].append(line[97:112])
        data['Qpp'].append(line[112:127])
        data['Ierr'].append(line[127:133])
        data['Iref'].append(line[133:145])
        data['flag'].append(line[145])
        data['gp'].append(line[146:153])
        data['gpp'].append(line[153:160])

    if filename.endswith('.zip'):
        zip.close()
    else:
        filehandle.close()

    for key in data:
        data[key] = array(data[key])

    return(data)

## ======================================================
def translate_molecule_identifier(M):
    '''
    For a given input molecule identifier number, return the corresponding molecular formula.
    Parameters
    ----------
    M : int
        The HITRAN molecule identifier number.
    Returns
    -------
    molecular_formula : str
        The string describing the molecule.
    '''

    trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     '8':'NO',
              '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
             '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl', '22':'N2',   '23':'HCN',   '24':'CH3Cl',
             '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
             '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4', '39':'CH3OH', '40':'CH3Br',
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3'}
    return(trans[str(M)])

## ======================================================
def get_molecule_identifier(molecule_name):
    '''
    For a given input molecular formula, return the corresponding HITRAN molecule identifier number.
    Parameters
    ----------
    molecular_formula : str
        The string describing the molecule.
    Returns
    -------
    M : int
        The HITRAN molecular identified number.
    '''

    trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     '8':'NO',
              '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
             '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl', '22':'N2',   '23':'HCN',   '24':'CH3Cl',
             '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
             '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4', '39':'CH3OH', '40':'CH3Br',
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3'}
    ## Invert the dictionary.
    trans = {v:k for k,v in trans.items()}
    return(int(trans[molecule_name]))

## ======================================================
def calculate_hitran_xsec(data, wavemin=None, wavemax=None, wavenumarr=None, npts=20001, units='m^2', temp=296.0, pressure=1.0):
    '''
    Given the HITRAN data (line centers and line strengths) for a molecule, digitize the result into a spectrum of
    absorption cross-section in units of cm^2.
    Parameters
    ----------
    data : dict of ndarrays
        The HITRAN data corresponding to a given molecule.
    wavemin : float, optional
        The minimum wavelength os the spectral region of interest.
    wavemax : float, optional
        The maximum wavelength os the spectral region of interest.
    wavenumarr: array, floats, optional
    	Array of wavenumbers to define the data at. If defined, will overwrite wavemin and wavemax using bounds of this array
    units : str, optional
        A string describing in what units of the output cross-section should be given in. Choices available are:
        {'cm^2/mole', 'cm^2.ppm', 'm^2/mole', 'm^2.ppm', 'm^2', cm^2}.
    temp : float
        The temperature of the gas, in Kelvin.
    pressure : float
        The pressure of the gas, in atmospheres.
    Returns
    -------
    waves : ndarray
        The wavenumbers (cm-1) at which the cross-section data is evaluated.
    xsec : array_like
        The mean absorption cross-section (in cm^2) per molecule, evaluated at the wavelengths given by input `waves`.
    '''

    assert (temp > 70.0) and (temp < 3000.0), 'Gas temperature must be greater than 70K and less than 3000K.'
    if (wavemin is None):
    	if (wavenumarr is None):
        	wavemin = amin(10000.0 / data['linecenter']) - 0.1
        else:
        	wavemin = wavenumarr[0]
    if (wavemax is None):
    	if (wavenumarr is None):
        	wavemax = amax(10000.0 / data['linecenter']) + 0.1
        else:
        	wavemax = wavenumarr[-1]

    ## First step: remove any data points that do not correspond to the primary isotope. (If we want to use isotopes,
    ## then we need to figure out mixing ratios.) For most HITRAN gases, the primary isotope is about 99% of the total
    ## atmospheric composition.
    okay = (data['I'] == 1)
    linecenters = array(data['linecenter'][okay])       ## line centers in wavenumbers
    linestrengths = array(data['S'][okay])
    linewidths = array(data['gamma-air'][okay])
    gammas_self = array(data['gamma-self'][okay])
    N_tempexps = array(data['N'][okay])     ## the temperature-dependent exponent for air-broadened linewidths
    nlines = alen(linecenters)
    Qratio = 1.0     # this is a placeholder for the ratio of total partition sums
    Epps = array(data['Epp'][okay])         ## the lower-energy-level energy (in cm^{-1})
    deltas = array(data['delta'][okay])     ## the "air pressure shift" (in cm^{-1} / atm)

    ## Convert the wavelengths (um) to wavenumbers (cm^{-1}). Create a spectrum linearly sampled in wavenumber (and
    ## thus nonuniformly sampled in wavelength).
    if (wavenumarr is None):
    	wavenumbers = linspace(wavemin, wavemax, npts)
    else:
    	wavenumbers = wavenumarr
    waves = 10000.0 / wavenumbers
    xsec = zeros_like(wavenumbers)

    ## Define the list of channel boundary wavelengths.
    dk = wavenumbers[1] - wavenumbers[0]

    for i in arange(nlines):
        linecenter = linecenters[i]
        linestrength = linestrengths[i]
        linewidth = linewidths[i]
        N_tempexp = N_tempexps[i]
        Epp = Epps[i]
        delta = deltas[i]

        ## If the spectral line is well outside our region of interest, then ignore it.
        if (linecenter < amin(wavenumbers-0.5)):
            continue
        elif (linecenter > amax(wavenumbers+0.5)):
            continue

        ## If using a different temperature and pressure than the HITRAN default (296K and 1atm), then scale the
        ## linewidth by the temperature and pressure, adjust the linecenter due to pressure, and scale the
        ## linestrength.
        linecenter += delta * (pressure - 1.0) / pressure
        linewidth *= (pressure / 1.0) * pow(296.0/temp, N_tempexp)
        linestrength *= Qratio * exp(1.43877 * Epp * ((1.0/296.0) - (1.0/temp)))

        ## Note: the quantity sum(L * dk) should sum to "S"!
        
        L = lorentzian_profile(wavenumbers, linestrength, linewidth, linecenter)
        xsec += L

    if units.endswith('/mole'):
        xsec = xsec * 6.022E23
    elif units.endswith('.ppm'):
        xsec = xsec * 2.686E19

    if units.startswith('cm^2'):
        pass
    elif units.startswith('m^2'):
        xsec = xsec / 10000.0

    return(1e4/waves, xsec)