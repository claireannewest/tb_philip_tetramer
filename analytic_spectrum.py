import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import yaml
from scipy.special import kn

open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']
hbar_cgs = param['constants']['hbar_cgs']

e = param['constants']['e']
eps_b = np.sqrt(param['n_b'])
observable = param['spectra_fitting']['observable']
n = eps_b**2
nm_to_per_s = 2*np.pi*c/(n)*1e7 # omega = this variable / lambda (in nm)
um_to_per_s = 2*np.pi*c/(n)*1e4 # omega = this variable / lambda (in um)
dim=2

load_coordinates = param['load_coords']
coordinates = np.loadtxt(load_coordinates)
x_coords = coordinates[:,0]
numPart = np.float(len(x_coords))
dip_coords = np.zeros((np.int(numPart),np.int(dim)),dtype=np.double)

for row in range(0,np.int(numPart)):
    dip_coords[row,:] = np.array((coordinates[row,0], coordinates[row,1]))
dip_coords = np.append(dip_coords, dip_coords,axis=0)*10**(-7)
y_coords = dip_coords[0:np.int(numPart),0]*10**7
z_coords = dip_coords[0:np.int(numPart),1]*10**7


def loadData():
	dir = param['spectra_fitting']['which_dipole']
	path = param['spectra_fitting']['path']
	data = np.loadtxt(path)#,delimiter=",", dtype=float)
	observable = param['spectra_fitting']['observable']
	code = param['spectra_fitting']['which_code']

	if code == 'edda':
		area_cross = 1
		w = data[:,1] # frequency in eVs
		effic = data[:,2]
	if code == 'edda_exp':
		area_cross = 1
		w = data[0,:]
		effic = data[1,:]

	allData = np.column_stack([w, effic])
	allData_sort = allData[allData[:,0].argsort()[::-1]]

	# idx = np.where(allData_sort[:,0] >= 2.5)		
	# allData_sort = np.delete(allData_sort, idx, axis=0)
	# idx = np.where(allData_sort[:,0] <= 1.3)	
	# allData_sort = np.delete(allData_sort, idx, axis=0)
	
	w = np.asarray(allData_sort[:,0]) 
	effic_sim = np.asarray(allData_sort[:,1])#/max(allData_sort[:,1]) 
	return [w, effic_sim, area_cross]

scalem = 1e-35
scalea=1E-3
scalem1 = 1e-35
scalem2 = 1e-34
def gammaEELS(w,w0,m_scaled, amp_scale): #w & w0 in eVs
	gam0 = 0.069
	m = m_scaled*scalem
	amp = amp_scale*scalea
	v = 0.48
	gamL = 1/np.sqrt(1-v**2)
	R = 9.0E-7
	gamR = hbar_eVs*(w/hbar_eVs)**2*(2.0*e**2)/(3.0*m*c**3)
	gamtot = gam0 + gamR
	return amp*(4.0*e**2/(hbar_eVs*hbar_cgs*np.pi*(v*c)**4*gamL**2)*w**2*(kn(1,(w/hbar_eVs)*R/(v*c*gamL))**2)*(e**2/m*np.imag(1./(w0**2 - w**2 - 1j*w*gamtot))))

def gammaEELS_two(w, w01, w02, m1_scaled, m2_scaled, amp1_scale, amp2_scale):
    gam01 = 0.069; gam02 = 0.069
    m1 = m1_scaled*scalem1
    m2 = m2_scaled*scalem2
    amp1 = amp1_scale*scalea
    amp2 = amp2_scale*scalea
    v = 0.48
    gamL = 1/np.sqrt(1-v**2)
    R = 9.0E-7
    gamR1 = hbar_eVs*(w/hbar_eVs)**2*(2.0*e**2)/(3.0*m1*c**3)
    gamR2 = hbar_eVs*(w/hbar_eVs)**2*(2.0*e**2)/(3.0*m2*c**3)
    gamtot1 = gam01 + gamR1
    gamtot2 = gam02 + gamR2
    return amp1*(4.0*e**4/((m1*hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*w**2*(kn(1,(w/hbar_eVs)*R/(v*c*gamL))**2)*np.imag(1./(w01**2 - w**2 - 1j*w*gamtot1))) + amp2*(4.0*e**4/((m2*hbar_eVs)*hbar_cgs*np.pi*(v*c)**4*gamL**2)*w**2*(kn(1,(w/hbar_eVs)*R/(v*c*gamL))**2)*np.imag(1./(w02**2 - w**2 - 1j*w*gamtot2)))

#FITS:
m_long = 3.41614E-35
m_short = 4.46092E-35
wsp_long = 1.3439#/hbar_eVs
wsp_short = 1.6913#/hbar_eVs

plt.plot(loadData()[0], gammaEELS(loadData()[0], w0=wsp_long, m_scaled=m_long/scalem, amp_scale=0.94367) + gammaEELS(loadData()[0], w0=wsp_short, m_scaled=m_short/scalem, amp_scale=0.0013543),label='Analytic')

plt.plot(loadData()[0], loadData()[1], '--', label='Raw data')
plt.title('Single Particle Gamma EEL Spectrum')	

plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel('Normalized EEL')


plt.show()
loadData()