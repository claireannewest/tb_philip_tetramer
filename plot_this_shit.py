import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure    
import yaml
from claires_tight import calculateEigen

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
    total, dip_coords  = calculateEigen()
    w = total[:,0]
    base_vectors = np.loadtxt(base_vector)
    numDip = len(base_vectors)
    v = total[:,1:np.int(numDip)+1]
    y_coords = dip_coords[0:np.int(numPart),0]
    z_coords = dip_coords[0:np.int(numPart),1]

    unit_vector = np.zeros((np.int(numDip),2),dtype=np.double)
    vector = np.zeros((np.int(numDip),np.int(numPart)*4),dtype=np.double)
    vec_y_coord_all = np.zeros((np.int(numDip),np.int(numDip)),dtype=np.double)
    vec_z_coord_all = np.zeros((np.int(numDip),np.int(numDip)),dtype=np.double)

    for mode in range(0,np.int(numDip)):
        magnitudes = np.array([v[mode]]).T
        maggys = np.column_stack([magnitudes, magnitudes])
        for particle in range(0,np.int(numDip)):
            unit_vector[particle,:] = base_vectors[particle,:]/np.linalg.norm(base_vectors[particle,:])
        vector = maggys*unit_vector
        vec_y_coord_all[:,mode] = vector[:,0]
        vec_z_coord_all[:,mode] = vector[:,1]     

    fig = plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k') 
    ax = fig.gca()
    for mode in range(0,np.int(numDip)):
        if numPart != numDip:
            evec_ycoord = vec_y_coord_all[0:np.int(numPart),mode] + vec_y_coord_all[np.int(numPart):np.int(numPart*dim),mode]
            evec_zcoord = vec_z_coord_all[0:np.int(numPart),mode] + vec_z_coord_all[np.int(numPart):np.int(numPart*dim),mode]
        if numPart == numDip:
            evec_ycoord = vec_y_coord_all[:,mode] 
            evec_zcoord = vec_z_coord_all[:,mode] 
       
        evalue = round(w[mode], prec)
        evalue_nm = hbar_eVs*2*c*np.pi*10**7/(evalue*eps_b**2)
        axs = plt.subplot(2,4,mode+1)
        plt.title('%.2f eV' % (evalue), fontsize=14)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

        xval, yval = ellipse(particle=0)
        plt.scatter(xval,yval, color='goldenrod')
        xval, yval = ellipse(particle=1)
        plt.scatter(xval,yval, color='goldenrod')
        xval, yval = ellipse(particle=2)
        plt.scatter(xval,yval, color='goldenrod')
        xval, yval = ellipse(particle=3)
        plt.scatter(xval,yval, color='goldenrod')

        plt.scatter(y_coords, z_coords, color='black', s=10)
        plt.quiver(y_coords,z_coords,evec_ycoord,evec_zcoord, 
            pivot='mid', 
            width=0.1,
            scale=2.0, 
            headlength=4,
            headwidth=5.8,
            minshaft=4.1, 
            minlength=.1
            )
        print 'mode = ',mode, 'evec = ', evec_ycoord,evec_zcoord
        plt.xlim([min(y_coords)-1e-5,max(y_coords)+1e-5])
        plt.ylim([min(y_coords)-1e-5, max(y_coords)+1e-5])
        axs.set_aspect('equal', 'box')
    r = z_coords[0]*np.sqrt(2)*1E7
    plt.suptitle('Mode analysis for r = %0.f nm' % r, fontsize=14)

seeModes()



plt.show()