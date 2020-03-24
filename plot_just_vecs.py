import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure    
import yaml
from just_vecs import calculateEigen

open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

eps_b = np.sqrt(param['n_b'])
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
dim = param['constants']['dim']

prec = param['precision']
base_vector = param['dipole_directions']


def ellipse(particle):
    data_coord = np.loadtxt(param['load_coords'],skiprows=1)
    basevecs = np.loadtxt(param['dipole_directions'])
    a = 170./3*1E-7
    b = 90./3*1E-7
    numPoints = 150
    xval = []
    yval = []
    opp = basevecs[particle,1]
    adj = basevecs[particle,0]
    if adj == 0:
        theta = np.pi/2
    else:
        theta = np.arctan(opp/adj)
    if theta < 0:
        theta = np.pi + theta
    x_cent = data_coord[particle,0]
    y_cent = data_coord[particle,1]
    x_range = np.linspace(-350E-7, 350E-7, numPoints)
    y_range = np.linspace(-350E-7, 350E-7, numPoints)
    for i in range(0,numPoints):
        for j in range(0,numPoints):
            x = x_range[i]
            y = y_range[j]
            if ( x )**2/a**2 + ( y )**2/b**2 <= 1:
                xrot = x*np.cos(theta) - y*np.sin(theta)
                yrot = x*np.sin(theta) + y*np.cos(theta)
                xrotshift = xrot + x_cent
                yrotshift = yrot + y_cent
                xval = np.append(xval, xrotshift)
                yval = np.append(yval, yrotshift)
    return xval, yval



def seeModes(numPart=4):
    wspan, vecmag_at_w  = calculateEigen()
    print vecmag_at_w
    dip_coords = np.zeros((np.int(numPart),np.int(dim)),dtype=np.double) # Initializing array that will contain all dipole coordinates
    coordinates = np.loadtxt(param['load_coords'], skiprows=1)[:,0:2] # location of each particle in cm
 
    base_vectors = np.loadtxt(base_vector)
    fig = plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k') 
    ax = fig.gca()

    for freq_idx in range(0,int(len(wspan))):
        vecs_after_mult = np.multiply(base_vectors, vecmag_at_w[freq_idx,:,np.newaxis])

        tot_evec= vecs_after_mult[:numPart] + vecs_after_mult[:numPart]
        #print tot_evec
        axs = plt.subplot(2,5,freq_idx+1)
        plt.title('%.2f eV' % (wspan[freq_idx]*hbar_eVs), fontsize=14)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

        xval, yval = ellipse(particle=0)
        plt.scatter(xval,yval, color='goldenrod')
        xval, yval = ellipse(particle=1)
        plt.scatter(xval,yval, color='goldenrod')
        xval, yval = ellipse(particle=2)
        plt.scatter(xval,yval, color='goldenrod')
        xval, yval = ellipse(particle=3)
        plt.scatter(xval,yval, color='goldenrod')

        plt.quiver(coordinates[:,0],coordinates[:,1],tot_evec[:,0],tot_evec[:,1], 
            pivot='mid', 
            width=0.1,
            scale=2.0, 
            headlength=4,
            headwidth=5.8,
            minshaft=4.1, 
            minlength=.1
            )

        plt.xlim([min(coordinates[:,0])-1e-5,max(coordinates[:,0])+1e-5])
        plt.ylim([min(coordinates[:,0])-1e-5, max(coordinates[:,0])+1e-5])
        axs.set_aspect('equal', 'box')

seeModes()



plt.show()