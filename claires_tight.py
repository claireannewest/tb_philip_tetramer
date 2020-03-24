import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys
from decimal import Decimal
import math
import yaml
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
eps_b = np.sqrt(param['n_b'])
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
prec = param['precision']

inputs = np.loadtxt(param['inputs'],skiprows=1)

part_centers = inputs[:,0:2]*1E-7 # particle centers
dip_centers = inputs[:,2:4]*1E-7 # the center of the particle is not necessarily the center of the dipole
L_vecs = inputs[:,4:6]*1E-7
S_vecs = inputs[:,6:8]*1E-7

numParts = len(inputs) # number of particles
numIndModes = (inputs.shape[1]-4)/2

w0_L = 1.62 # w0 of long axis dipole 
w0_S = 2.45 # w0 of short axis dipole 
m_L = .9824E-34 # effective mass of long axis dipole 
m_S = 1.773E-34 # effective mass of short axis dipole

gamNR_L = 0.07
gamNR_S = 0.07

def DL(i): #allows me to grab the dipole centers and directions for each particle
    part_cent_i = part_centers[i : i+1, :]
    dipcent_i = dip_centers[i : i+1, :]
    vecs = L_vecs[i : i+1, :]
    return np.column_stack(( part_cent_i, dipcent_i, vecs ))

def DS(i): #allows me to grab the dipole centers and directions for each particle
    part_cent_i = part_centers[i : i+1, :]
    dipcent_i = dip_centers[i : i+1, :]
    vecs = S_vecs[i : i+1, :]
    return np.column_stack(( part_cent_i, dipcent_i, vecs ))

def make_g(mode_i, mode_j,m,k): #mode 1,2 are four columns: [part_cent_x, part_cent_y, vec_x, vec_y] and four rows corresponding to the four particles
    k = np.real(k)
    r_ij = mode_i[0,2:4]-mode_j[0,2:4]  #distance between the nth and mth dipole
    mag_rij = np.linalg.norm(r_ij)
    if mag_rij == 0: g=0
    else:
        nhat_ij = r_ij / mag_rij
        xi = mode_i[0, 4:6]  
        xj = mode_j[0, 4:6]   
        xi_hat = xi/np.linalg.norm(xi)
        xj_hat = xj/np.linalg.norm(xj)
        xi_dot_nn_dot_xj = np.dot(xi_hat, nhat_ij)*np.dot(nhat_ij, xj_hat)
        nearField = ( 3.*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat) ) / mag_rij**3
        intermedField = 1j*k*(3*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij**2 
        farField = k**2*(xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij
        g =  e**2 * hbar_eVs**2 * ( nearField - intermedField - farField ) * np.exp(1j*k*mag_rij) 
    return -g/(m)

def make_H(k):
    H = np.zeros( (int(numIndModes*numParts),int(numIndModes*numParts)),dtype=complex) 
    w_thisround = k*c/np.sqrt(eps_b)*hbar_eVs #eV

    gam_L = gamNR_L + (np.real(w_thisround))**2*(2.0*e**2)/(3.0*m_L*c**3)/hbar_eVs
    gam_S = gamNR_S + (np.real(w_thisround))**2*(2.0*e**2)/(3.0*m_S*c**3)/hbar_eVs

    for i in range(0, numParts): #handle the on diagonal terms 
        H[ int(numIndModes*i)   , int(numIndModes*i)   ] = w0_L**2 - 1j*gam_L*w_thisround
        H[ int(numIndModes*i+1) , int(numIndModes*i+1) ] = w0_S**2 - 1j*gam_S*w_thisround

    for part_i in range(0 , numParts-1): #handle the off diagonal terms
        for part_j in range(1, numParts): 
            if part_i != part_j:
               
                ### L_i coupled with (L_1, S_1, ... L_N, S_N) ### 
                H[ int(numIndModes*part_i), int(numIndModes*part_j)  ] = make_g(mode_i=DL(i=part_i), mode_j=DL(i=part_j), m=m_L, k=k)
                H[ int(numIndModes*part_i), int(numIndModes*part_j+1) ] = make_g(mode_i=DL(i=part_i), mode_j=DS(i=part_j), m=m_L, k=k)
                
                ### S_i coupled with (L_1, S_1, ... L_N, S_N) ###
                H[ int(numIndModes*part_i+1), int(numIndModes*part_j)  ] = make_g(mode_i=DS(i=part_i), mode_j=DL(i=part_j), m=m_S, k=k)
                H[ int(numIndModes*part_i+1), int(numIndModes*part_j+1)] = make_g(mode_i=DS(i=part_i), mode_j=DS(i=part_j), m=m_S, k=k)

                ########## Now the opposite of the above terms ##########
                H[ int(numIndModes*part_j)  , int(numIndModes*part_i) ] = make_g(mode_i=DL(i=part_j), mode_j=DL(i=part_i), m=m_L, k=k)
                H[ int(numIndModes*part_j+1), int(numIndModes*part_i) ] = make_g(mode_i=DS(i=part_j), mode_j=DL(i=part_i), m=m_S, k=k)
                
                H[ int(numIndModes*part_j)  , int(numIndModes*part_i+1) ] = make_g(mode_i=DL(i=part_j), mode_j=DS(i=part_i), m=m_L, k=k)
                H[ int(numIndModes*part_j+1), int(numIndModes*part_i+1) ] = make_g(mode_i=DS(i=part_j), mode_j=DS(i=part_i), m=m_S, k=k)

        eigval, eigvec = np.linalg.eig(H)
    return eigval, eigvec, H



def interate():
    final_eigvals = np.zeros(np.int(numIndModes*numParts),dtype=complex)
    final_eigvecs = np.zeros( (np.int(numIndModes*numParts), np.int(numIndModes*numParts)), dtype=complex) 
    w_Lstart = -1j*gamNR_L/2. + np.sqrt(-gamNR_L**2/4.+w0_L**2)
    w_Sstart = -1j*gamNR_S/2. + np.sqrt(-gamNR_S**2/4.+w0_S**2)

    for mode in range(0,np.int(numIndModes*numParts)): #converge each mode individually         
        if mode == 0 or mode == 2 or mode == 4 or mode == 6: 
            eigval_hist = np.array([w_Lstart, w_Lstart*1.1],dtype=complex) 
        if mode == 1 or mode == 3 or mode == 5 or mode == 7:
            eigval_hist = np.array([w_Sstart, w_Sstart*1.1],dtype=complex) 

        eigvec_hist = np.zeros((int(numIndModes*numParts), 2))
        eigvec_hist[:,0] = 0.5
        vec_prec = np.zeros((int(numIndModes*numParts), 1))+10**(-prec)

        count = 0
        inercount = 1

        while np.abs((np.real(eigval_hist[0]) - np.real(eigval_hist[1])))  > 10**(-prec) and np.sum(np.abs((eigvec_hist[:,0] - eigvec_hist[:,1]))) > 10**(-prec):
            w_thisround = eigval_hist[0]
            
            if count > 100: 
               denom = ( eigval_hist[2] - eigval_hist[1] ) - ( eigval_hist[1] - eigval_hist[0] )
               w_thisround = eigval_hist[2] - ( eigval_hist[2] - eigval_hist[1] )**2 / denom 
            
            k = w_thisround/hbar_eVs*np.sqrt(eps_b)/c

            val, vec, H = make_H(k=k)

            amp = np.sqrt(np.abs(val))
            phi = np.arctan2(np.imag(val), np.real(val))
            energy = amp*np.cos(phi/2)

            post_sort_val = energy[energy.argsort()]
            post_sort_vec = vec[:,energy.argsort()]

            this_val = post_sort_val[mode]
            this_vec = post_sort_vec[:,mode]
            new_eigvals = this_val

            eigval_hist = np.append(new_eigvals, eigval_hist)
            eigvec_hist = np.column_stack((this_vec, eigvec_hist))

            print(mode, count)
            count = count + 1 

        final_eigvals[mode] = eigval_hist[0]
        final_eigvecs[:,mode] = eigvec_hist[:,0]
    return final_eigvals, final_eigvecs

L_vecs = inputs[:,4:6]
S_vecs = inputs[:,6:8] 

def seeVectors(mode):
    w = np.real(final_eigvals[mode])
    v = np.real(final_eigvecs[:,mode])

    dip_ycoords = dip_centers[:,0]
    dip_zcoords = dip_centers[:,1]  
   
    plt.subplot(1,int(numIndModes*numParts),mode+1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.title('%.2f eV' % (w), fontsize=18)
    plt.scatter(dip_ycoords, dip_zcoords,c='blue',s=50)

    for part_i in range(0, numParts):
        DL_i = L_vecs[part_i : part_i+1, :]
        DS_i = S_vecs[part_i : part_i+1, :]
        mag_mode = (v[int(numIndModes*part_i)]*DL_i + v[int(numIndModes*part_i+1)]*DS_i)

        ymin = min(dip_ycoords)-1E-5; ymax = max(dip_ycoords)+1E-5
        zmin = min(dip_zcoords)-1E-5; zmax = max(dip_zcoords)+1E-5
        plt.quiver(dip_ycoords[part_i : part_i+1], dip_zcoords[part_i : part_i+1], mag_mode[:,0], mag_mode[:,1], pivot='mid', 
            width=.5, #shaft width in arrow units 
            scale=1., 
            headlength=5,
            headwidth=5.,#5.8
            minshaft=4., #4.1
            minlength=.1)
    plt.xlim([ymin, ymax])
    plt.ylim([zmin, zmax])
    plt.yticks([])
    plt.xticks([])
    #plt.show()
    return w, mag_mode

final_eigvals, final_eigvecs = interate()

fig = plt.figure(num=None, figsize=(12, 2), dpi=80, facecolor='w', edgecolor='k')   
for mode in range(0,int(numIndModes*numParts)):
    seeVectors(mode=mode)
plt.show()

def seeFields(mode):
    w = np.real(final_eigvals[mode])
    v = np.real(final_eigvecs[:,mode])

    part_xcoords = 0*dip_centers[:,0]
    part_ycoords = dip_centers[:,0]
    part_zcoords = dip_centers[:,1]  
    part_origins = np.column_stack((part_xcoords, part_ycoords, part_zcoords))
    p = np.zeros((int(numParts), 3))
    for part_i in range(0, int(numParts)):
        L_i = L_vecs[(part_i) : (part_i+1), :]
        S_i = S_vecs[(part_i) : (part_i+1), :]
        p[int(part_i) : int(part_i+1), 1:3] = ( v[int(numIndModes*part_i)]*L_i + v[int(numIndModes*part_i+1)]*S_i )

    ymin = min(part_origins[:,1])-2E-5; ymax = max(part_origins[:,1])+2E-5
    zmin = min(part_origins[:,2])-2E-5; zmax = max(part_origins[:,2])+2E-5

    x = 60e-07; 
    numPoints = 71
    y = np.linspace(ymin, ymax, numPoints ); z = np.linspace(zmin, zmax, numPoints )

    ### Efield for every dipole, [ which dipole, which y point, which z point ] ###
    Ex_field = np.zeros( (int(numParts), int(numPoints), int(numPoints)),dtype=complex)
    Ey_field = np.zeros( (int(numParts), int(numPoints), int(numPoints)),dtype=complex)
    Ez_field = np.zeros( (int(numParts), int(numPoints), int(numPoints)),dtype=complex)
    
    for which_dipole in range(0, int(numParts)):
        for which_y in range(0, int(numPoints)):
            for which_z in range(0, int(numPoints)):
                xval = x
                yval = y[which_y]
                zval = z[which_z]
                k = w/hbar_eVs/c
                point = np.array([xval, yval, zval])
                r = point - part_origins
                nhat = r/np.linalg.norm(r)
                nhat_dot_p = np.sum(nhat*p,axis=1)[:,np.newaxis]
                magr = np.linalg.norm(r,axis=1)[:,np.newaxis]
                nearField = ( 3*nhat * nhat_dot_p - p ) / magr**3
        #         intermedField1 = 1j*k*(3*rhat1 - np.dot(rhat1,p1)) / (np.linalg.norm(r1))**2
        #         farField1 = k**2*(rhat1 - np.dot(rhat1,p1)) / (np.linalg.norm(r1))
                Ex_field[which_dipole, which_z, which_y] = nearField[which_dipole,0]#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))
                Ey_field[which_dipole, which_z, which_y] = nearField[which_dipole,1]#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))
                Ez_field[which_dipole, which_z, which_y] = nearField[which_dipole,2]#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))

    #whichsphere = 1

    Extot = np.real(Ex_field[0,:,:]+Ex_field[1,:,:]+Ex_field[2,:,:]+Ex_field[3,:,:])
    #Eytot = np.real(Ey_field[whichsphere,:,:])#+Ey_field[1,:,:]+Ey_field[2,:,:]+Ey_field[3,:,:]+Ey_field[4,:,:]+Ey_field[5,:,:]+Ey_field[6,:,:]+Ey_field[7,:,:])
    #Eztot = np.real(Ez_field[whichsphere,:,:])

    plt.imshow(Extot, 
        cmap='seismic',
        origin='lower',
        extent=[ymin,ymax,zmin,zmax]
        )

    plt.scatter(part_origins[:,1], part_origins[:,2],c='black',s=30)
    plt.quiver(part_origins[:,1], part_origins[:,2], p[:,1], p[:,2], pivot='mid', 
        width=0.1, #shaft width in arrow units 
        scale=2., 
        headlength=4,
        headwidth=5.8,
        minshaft=4.1, 
        minlength=.1)
    # plt.quiver(part_origins[whichsphere,1], part_origins[whichsphere,2], p[whichsphere,1], p[whichsphere,2], color='green',pivot='mid', width=0.1,scale=2.,headlength=4,headwidth=5.8,minshaft=4.1, minlength=.1)
    plt.show()

#seeFields(mode=0)



