import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys
from decimal import Decimal
import math
import yaml
from scipy import optimize
from scipy.special import kn
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

open_param_file = open('parameters.yaml')
param = yaml.load(open_param_file)
c = param['constants']['c']
eps_b = 1.0

hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
hbar_cgs = param['constants']['hbar_cgs']
prec = param['precision']

n=1
um_to_per_s = 2*np.pi*c/(n)*1e4 # omega = this variable / lambda (in um)
inputs = np.loadtxt(str('../')+param['inputs'],skiprows=1)

which='Ds'
elong_or_chub = 'elong'

def loadData(which):
    if which == 'Dl':
        if elong_or_chub == 'chub':
            data_long = np.loadtxt('/Users/clairewest/werk/research/philips_square/simulations/eels/Spectrum_monomer_long',skiprows=1)
        if elong_or_chub == 'elong':
            data_long = np.loadtxt('/Users/clairewest/werk/research/philips_square/simulations/eels/elongate/Spectrum_elongmono_long',skiprows=1)
        w_long = data_long[:,1] #eV
        effic_long = data_long[:,2]
        allData_long = np.column_stack([w_long, effic_long])
        allData_sortlong = allData_long[allData_long[:,0].argsort()[::-1]]
        idx = np.where(allData_sortlong[:,0] > 1.8)     #1.9
        allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        idx = np.where(allData_sortlong[:,0] <= 0.)  
        allData_sortlong = np.delete(allData_sortlong, idx, axis=0)
        w = np.asarray(allData_sortlong[:,0]) 
        effic_sim = np.asarray(allData_sortlong[:,1])
    
    if which == 'Ds':
        if elong_or_chub == 'chub':
            data_short = np.loadtxt('/Users/clairewest/werk/research/philips_square/simulations/eels/Spectrum_monomer_shortoff',skiprows=1)
        if elong_or_chub == 'elong':
            data_short = np.loadtxt('/Users/clairewest/werk/research/philips_square/simulations/eels/elongate/Spectrum_elongmono_shortoff',skiprows=1)
        w_short = data_short[:,1] # eV
        effic_short = data_short[:,2]
        allData_short = np.column_stack([w_short, effic_short])
        allData_sortshort = allData_short[allData_short[:,0].argsort()[::-1]]
        idx = np.where(allData_sortshort[:,0] >= 2.5)  #2.5   
        allData_sortshort = np.delete(allData_sortshort, idx, axis=0)
        idx = np.where(allData_sortshort[:,0] <= 0.)  
        allData_sortshort = np.delete(allData_sortshort, idx, axis=0)
        w = np.asarray(allData_sortshort[:,0]) 
        effic_sim = np.asarray(allData_sortshort[:,1])   

    return [w, effic_sim/max(effic_sim)]

def gammaEELS(
    w_all, # the range of wavelengths the Gam EELS is taken over, i.e. we need a val of GamEEL for many different wavelenths [1/s]
    w0,
    gamR,
    ebeam_loc,
    amp,
    ):     
    v = 0.48 
    gamL = 1/np.sqrt(1-v**2)
    m = (2.0*e**2)/(3.0*gamR*hbar_eVs*c**3)
    gam = 0.07 + gamR*w_all**2
    alpha = e**2/m * hbar_eVs**2/(-w_all**2 - 1j*gam*w_all + w0**2)
    rod_loc = np.array([0,0])
    magR = np.linalg.norm(ebeam_loc-rod_loc)
    constants = 4.0*e**2/((hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*(w_all/hbar_eVs)**2*(kn(1,(w_all/hbar_eVs)*magR/(v*c*gamL)))**2
   # print constants
    Gam_EELS = constants*np.imag(alpha)
    return amp*Gam_EELS #units of 1/eV

gaml_scale = .05
ampl_scale = 25

def fit_gammaEELSlong(raw_w, w0l, gamRl, ampl):
    ebeam_loc = np.array([-46*2*1E-7,0])
    gammaEEL_long = gammaEELS(w_all=raw_w,w0=w0l, gamR=gamRl*gaml_scale,amp=ampl*ampl_scale,ebeam_loc=ebeam_loc)
    return gammaEEL_long

def plot_n_fit_long():
    #       w_s, g_s, a
    lower = [1.,  1,    1]
    upper = [2.5, 10,  10]
    guess = [1.7,  5,   5]

    w_rawdata = loadData(which='Dl')[0]
    eel_rawdata = loadData(which='Dl')[1]
    params, params_covariance = optimize.curve_fit(fit_gammaEELSlong, w_rawdata, eel_rawdata, bounds=[lower,upper],p0=guess)
    print(params)
    print('w_l = ', '%.2f' % params[0])
    print('m_l = ', '%.3e' % ((2.0*e**2)/(3.0*(params[1]*gaml_scale)*hbar_eVs*c**3)))

    plt.subplot(1,1,1)
    if elong_or_chub == 'chub':
        data_long = np.loadtxt('/Users/clairewest/werk/research/philips_square/simulations/eels/Spectrum_monomer_long',skiprows=1)
        allData = np.column_stack([data_long[:,1], data_long[:,2]]); allData_sort = allData[allData[:,0].argsort()[::-1]]
        title='Fitting Long Axis Dipole (172 nm x 92 nm rod)'
    if elong_or_chub == 'elong':
        data_long = np.loadtxt('/Users/clairewest/werk/research/philips_square/simulations/eels/elongate/Spectrum_elongmono_long',skiprows=1)
        allData = np.column_stack([data_long[:,1], data_long[:,2]]); allData_sort = allData[allData[:,0].argsort()[::-1]]
        title='Fitting Long Axis Dipole (200 nm x 60 nm rod)'

    plt.plot(w_rawdata,  fit_gammaEELSlong(w_rawdata,*params),'k',label='Fit',linewidth=3)
    plt.plot(allData_sort[:,0], allData_sort[:,1]/max(allData_sort[:,1]),color='dodgerblue', linestyle=':',label='Raw data',linewidth=3)

    plt.title(title, fontsize=16)
    plt.xlim([0.5, 2.75])
    plt.xlabel('Energy [eV]', fontsize=16)
    plt.ylabel('Noramlized EEL [eV$^{-1}$]',fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks([0.5, 1.0, 1.5, 2.0, 2.5], fontsize=14)
    plt.legend(fontsize=14, frameon=False, loc='upper left')
    plt.subplots_adjust(bottom=0.2)
    plt.show()

if which == 'Dl':
    plot_n_fit_long()

########################################################################2#######################
################################################################################################

### normal ###
if elong_or_chub == 'chub':
    gams_scale = .02; gaml_scale = .09; 
    amps_scale = 9; ampl_scale = 2; 

### elongate ###
if elong_or_chub == 'elong':
    gams_scale = .04; gaml_scale = .08; 
    amps_scale = 10; ampl_scale = 1; 

def fit_gammaEELSshortnlong(raw_w, w0s, gamRs, amps,w0l, gamRl, ampl):
    ebeam_loc = np.array([0, -26*2*1E-7])
    gammaEEL_short = gammaEELS(w_all=raw_w,w0=w0s, gamR=gamRs*gams_scale,amp=amps*amps_scale,ebeam_loc=ebeam_loc)
    gammaEEL_long = gammaEELS(w_all=raw_w,w0=w0l, gamR=gamRl*gaml_scale,amp=ampl*ampl_scale,ebeam_loc=ebeam_loc)
    return gammaEEL_short + gammaEEL_long

def plot_n_fit_shortnquad():
    #       w_s, w_q, g_s, g_q, m_s, m_q
    lower = [1.9,  1.,  1,  1.4,  1., 1]
    upper = [2.9,  10., 10., 2.9,  10, 10.]
    guess = [2.45, 1.5, 5, 1.6, 5, 5]

    w_rawdata = loadData(which='Ds')[0]
    eel_rawdata = loadData(which='Ds')[1]
    params, params_covariance = optimize.curve_fit(fit_gammaEELSshortnlong, w_rawdata, eel_rawdata, bounds=[lower,upper],p0=guess)

    plt.subplot(1,1,1)
    if elong_or_chub == 'chub':
        data_long = np.loadtxt('/Users/clairewest/werk/research/philips_square/simulations/eels/Spectrum_monomer_shortoff',skiprows=1)
        allData = np.column_stack([data_long[:,1], data_long[:,2]]); allData_sort = allData[allData[:,0].argsort()[::-1]]
        title='Fitting Short Axis Dipole (172 nm x 92 nm rod)'
    if elong_or_chub == 'elong':
        data_long = np.loadtxt('/Users/clairewest/werk/research/philips_square/simulations/eels/elongate/Spectrum_elongmono_shortoff',skiprows=1)
        allData = np.column_stack([data_long[:,1], data_long[:,2]]); allData_sort = allData[allData[:,0].argsort()[::-1]]
        title='Fitting Short Axis Dipole (200 nm x 60 nm rod)'


    plt.plot(w_rawdata,  fit_gammaEELSshortnlong(w_rawdata,*params),'k',label='Fit',linewidth=3)
    plt.plot(w_rawdata, eel_rawdata, color='lime', linestyle=':',label='Raw data',linewidth=3)
   
    plt.title(title, fontsize=16)
    plt.xlim([1.25, 2.5])
    plt.xlabel('Energy [eV]', fontsize=16)
    plt.ylabel('Noramlized EEL [eV$^{-1}$]',fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks([1.5, 2.0, 2.5], fontsize=14)
    plt.legend(fontsize=14, frameon=False, loc='upper left')
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    print(params)
    print('w_s = ', '%.2f' % params[0])
    print('m_s = ', '%.3e' % ((2.0*e**2)/(3.0*(params[1]*gams_scale)*hbar_eVs*c**3)))

    print('w_l = ', '%.2f' % params[3])
    print('m_l = ', '%.3e' % ((2.0*e**2)/(3.0*(params[4]*gams_scale)*hbar_eVs*c**3)))

if which == 'Ds':
    plot_n_fit_shortnquad()
