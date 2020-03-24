import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import yaml
from scipy.special import kn

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

open_param_file = open('parameters.yaml')
param = yaml.load(open_param_file)
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']
hbar_cgs = param['constants']['hbar_cgs']

e = param['constants']['e']
eps_b = np.sqrt(param['n_b'])
n = eps_b**2
nm_to_per_s = 2*np.pi*c/(n)*1e7 # omega = this variable / lambda (in nm)
um_to_per_s = 2*np.pi*c/(n)*1e4 # omega = this variable / lambda (in um)
dim=2

longorshort = 'long'


def loadData_long():
	data = np.loadtxt('/Users/clairewest/werk/research/philips_square/simulation/eels/monomer/Spectrum_long',skiprows=1)
	w = um_to_per_s / data[:,0] # needs to be in 1/s
	effic = data[:,2]
	allData = np.column_stack([w, effic])
	allData_sort = allData[allData[:,0].argsort()[::-1]]
	idx = np.where(allData_sort[:,0] >= 2./hbar_eVs)		
	allData_sort = np.delete(allData_sort, idx, axis=0)
	idx = np.where(allData_sort[:,0] <= 1.5/hbar_eVs)	
	allData_sort = np.delete(allData_sort, idx, axis=0)
	w = np.asarray(allData_sort[:,0]) 
	effic_sim = np.asarray(allData_sort[:,1])#/max(allData_sort[:,1]) 
	return [w, effic_sim/max(effic_sim)]	


def loadData_short():
	data = np.loadtxt('/Users/clairewest/werk/research/philips_square/monomer/Spectrum_short',skiprows=1)
	w = um_to_per_s / data[:,0] # needs to be in 1/s
	effic = data[:,2]
	allData = np.column_stack([w, effic])
	allData_sort = allData[allData[:,0].argsort()[::-1]]
	idx = np.where(allData_sort[:,0] >= 2.5/hbar_eVs)		
	allData_sort = np.delete(allData_sort, idx, axis=0)
	idx = np.where(allData_sort[:,0] <= 2.0/hbar_eVs)	
	allData_sort = np.delete(allData_sort, idx, axis=0)
	w = np.asarray(allData_sort[:,0]) 
	effic_sim = np.asarray(allData_sort[:,1])#/max(allData_sort[:,1]) 
	return [w, effic_sim/max(effic_sim)]	


scalegam = .05

cs = 172./2*1E-7
a = 92./2*1E-7

def prolate_parameters(
		cs, # semi-major axis, units of cm
		a,   # semi-minor axis, units of cm
		which # which dipole excitation 
		): 
	es = (cs**2 - a**2)/cs**2
	Lz = (1-es**2)/es**3*(-es+1./2*np.log((1+es)/(1-es)))
	Ly = (1-Lz)/2	
	D = 3./4*((1+es**2)/(1-es**2)*Lz + 1)
	V = 4./3*np.pi*a**2*cs

	if which == 'long':
		li = cs
		Li = Lz
	if which == 'short':
		li = a
		Li = Ly
	return D,li, Li, V

def alpha_spheriod( #spheriod in modified long wavelength approx.
		w, # 1/s
		w0, # eV
		gamNR_scaled # eV
		): 
	D, li, Li, V = prolate_parameters(cs=cs, a=a, which=longorshort)
	eps_inf = 9.
	m = 4*np.pi*e**2*((eps_inf-1)+1/Li)/((w0/hbar_eVs)**2*V/Li**2) # g 
	m_LW = m + D*e**2/(li*c**2) # g (charge and speed of light)
	w0_LW = (w0/hbar_eVs)*np.sqrt(m/m_LW) # 1/s
	gamNR = gamNR_scaled*scalegam # eV
	gam_LW = gamNR/hbar_eVs*(m/m_LW) + 2*e**2/(3*m_LW*c**3)*w**2 # 1/s
	alpha = e**2/m_LW * 1/ (w0_LW**2 - w**2 - 1j*w*gam_LW) # cm**3
	return m_LW, m, alpha, w0, w0_LW

def gammaEELS(
		w, # 1/s 
		w0, # eV
		gamNR_scaled #eV
		): 
	m_LW, m, alpha, w0, w0_LW = alpha_spheriod(w=w, w0=w0,gamNR_scaled=gamNR_scaled)  
	gamNR = gamNR_scaled*scalegam #eV
	v = 0.48 
	gamL = 1/np.sqrt(1-v**2)
	R = 4.0E-7 # cm
	Gam_EELS = 4.0*e**2/((hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*w**2*(kn(1,w*R/(v*c*gamL))**2)*np.imag(alpha) 
	return m_LW, m, w0, w0_LW, Gam_EELS/max(Gam_EELS) #units of 1/eV

def fitfunc(w,w01, gamNR1_scaled):
	m_LW, m, w0, w0_LW, gameels = gammaEELS(w=w,w0=w01,gamNR_scaled=gamNR1_scaled)
	return gameels


if longorshort == 'long':
	idx = np.where(loadData_long()[1] == max(loadData_long()[1]))
	w0_guess = loadData_long()[0][idx]*hbar_eVs
	guess = [w0_guess,	5.]
	lower = [w0_guess*0.9,	0]
	upper = [w0_guess*1.3, 	10]
	
	params, params_covariance = optimize.curve_fit(fitfunc, loadData_long()[0], loadData_long()[1], p0=guess, bounds=[lower,upper])
	plt.plot(loadData_long()[0]*hbar_eVs, fitfunc(loadData_long()[0],*params),'k',label='Fit',linewidth=3)
	plt.plot(loadData_long()[0]*hbar_eVs, loadData_long()[1], color='dodgerblue', linestyle='--',label='Raw data',linewidth=3)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xlabel('Energy [eV]',fontsize=16)
	plt.ylabel('Noramlized EELS [a.u.]',fontsize=16)
	plt.title('Long axis dipole', fontsize=16)
	plt.legend(fontsize=14)

	print params
	print 'm_LW = ', gammaEELS(loadData_long()[0], *params)[0]
	print 'm = ', gammaEELS(loadData_long()[1], *params)[1]
	print 'w0 = ', gammaEELS(loadData_long()[1], *params)[2]
	print 'w0_LW = ', gammaEELS(loadData_long()[1], *params)[3]*hbar_eVs
	print 'gamNR = ', params[1]*scalegam


if longorshort == 'short':
	idx = np.where(loadData_short()[1] == max(loadData_short()[1]))
	w0_guess = loadData_short()[0][idx]*hbar_eVs
	guess = [w0_guess,	5.]
	lower = [w0_guess*0.9,	0]
	upper = [w0_guess*1.3, 	10]

	params, params_covariance = optimize.curve_fit(fitfunc, loadData_short()[0], loadData_short()[1], p0=guess, bounds=[lower,upper])
	plt.plot(loadData_short()[0]*hbar_eVs, fitfunc(loadData_short()[0],*params),color='k',label='Fit',linewidth=3)
	plt.plot(loadData_short()[0]*hbar_eVs, loadData_short()[1], color='limegreen', linestyle='--',label='Raw data',linewidth=3)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xlabel('Energy [eV]',fontsize=16)
	plt.ylabel('Noramlized EELS [a.u.]',fontsize=16)
	plt.title('Short axis dipole', fontsize=16)
	plt.legend(fontsize=14)

	print params
	print 'm_LW = ', gammaEELS(loadData_short()[0], *params)[0]
	print 'm = ', gammaEELS(loadData_short()[1], *params)[1]
	print 'w0 = ', gammaEELS(loadData_short()[1], *params)[2]
	print 'w0_LW = ', gammaEELS(loadData_short()[1], *params)[3]*hbar_eVs
	print 'gamNR = ', params[1]*scalegam

plt.show()