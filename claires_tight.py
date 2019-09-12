import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys
from decimal import Decimal
import math
import yaml

open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
eps_b = np.sqrt(param['n_b'])
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
dim = param['constants']['dim']
wp = param['constants']['w_p'] / hbar_eVs
eps_inf = param['constants']['eps_inf']
gam_NR = param['constants']['gam_NR'] / hbar_eVs
prec = param['precision']


def oscillatorParameters():
    if param['input_parameters'] == 'fit': # Using a fit function to determine oscillator parameters
        input_long = param['characterizing_dipoles']['long']
        input_short = param['characterizing_dipoles']['short']
    if param['input_parameters'] == 'prolate': # Using a prolate spheriod approximation to determine oscillator parameters 
        c_s = param['analytic_prolate']['c_s']
        a_s = param['analytic_prolate']['a_s']
    if param['input_parameters'] == 'oblate': # Using a prolate spheriod approximation to determine oscillator parameters 
        c_s = param['analytic_oblate']['c_s']
        a_s = param['analytic_oblate']['a_s']
    if param['input_parameters'] == 'fit':
        input_trans = np.loadtxt(input_short)
        input_long = np.loadtxt(input_long)
        m_long = input_long[0]
        m_short = input_trans[0]
        wsp_long = input_long[1]/hbar_eVs
        wsp_short = input_trans[1]/hbar_eVs

    if param['input_parameters'] == 'prolate': # y = long axis coded, yet called z in Kevin's notes, z = short axis
        e_s = np.sqrt( (c_s**2-a_s**2) /c_s**2)
        L_z = (1-e_s**2)/(e_s**3)*(-e_s+np.arctanh(e_s))
        L_y = (1.-L_z)/2.0 
        D_z = 3./4.*((1.0+e_s**2)/(1.0-e_s**2)*L_z+1)
        D_y = a_s/(2.*c_s)*(3./e_s*np.arctanh(e_s)-D_z)
        V = 4./3.*np.pi*c_s*(a_s)**2
        wsp_QS_long = np.sqrt(wp**2/((eps_inf-1.0)+1.0/L_z))
        m_QS_long = 4.0*np.pi*e**2*((eps_inf-1.0)+1.0/L_z)/(wsp_QS_long**2*V/L_z**2)
        l_i = c_s
        m_long = m_QS_long + D_z*e**2/(l_i*c**2)
        wsp_long = wsp_QS_long * np.sqrt(m_QS_long/m_long)
        wsp_QS_short = np.sqrt(wp**2/((eps_inf-1.0)+1.0/L_y))
        m_QS_short = 4.0*np.pi*e**2*((eps_inf-1.0)+1.0/L_y)/(wsp_QS_short**2*V/L_y**2)
        l_i = a_s
        m_short = m_QS_short + D_y*e**2/(l_i*c**2)
        wsp_short = wsp_QS_short * np.sqrt(m_QS_short/m_short)

    if param['input_parameters'] == 'oblate':
        e_s = np.sqrt( (a_s**2-c_s**2) /a_s**2)
        l_i = a_s
        L_z = 1.0/e_s**2 * (1.0 - np.sqrt(1.0-e_s**2)/e_s)
        L_x = (1.0-L_z)/2.0
        D_z = 3.0/4.0 *((1.0-2.0*e_s**2)*L_z+1.0)
        D_x = a_s/(2.0*c_s)*(3.0/e_s*np.sqrt(1.0-e_s**2)*np.arcsin(e_s)-D_z)
        V = 4.0/3.0*np.pi*a_s**2*c_s
        wsp_QS = np.sqrt(wp**2/((eps_inf-1.0)+1.0/L_x))
        m_QS = 4.0*np.pi*e**2*((eps_inf-1.0)+1.0/L_x)/(wsp_QS**2*V/L_x**2)
        m = m_QS + D_x*e**2/(l_i*c**2)
        wsp = wsp_QS * np.sqrt(m_QS/m)
        m_long = m
        m_short = m
        wsp_long = wsp
        wsp_short = wsp
    return m_short, m_long, wsp_short, wsp_long

def calculateEigen(): #everything is in CGS until the last step, all frequency is converted to eV

    m_short, m_long, wsp_short, wsp_long = oscillatorParameters()
    base_vectors = np.loadtxt(param['dipole_directions']) # column 0 = x direction; column 1 = y direction; row 0 = first dipole on particle 1; row 1 = second dipole on particle 1; ... units don't matter because these are unit vectors and will be normalized
    coordinates = np.loadtxt(param['load_coords']) # location of each particle in cm
    x_coords = coordinates[:,0] # x coordiantes of each particle 
    y_coords = coordinates[:,1] # y coordiantes of each particle 
    numPart = np.float(len(x_coords)) # total number of particles 
     
    dip_coords = np.zeros((np.int(numPart),np.int(dim)),dtype=np.double) # Initializing array that will contain all dipole coordinates
    for row in range(0,np.int(numPart)):  
        dip_coords[row,:] = np.array((coordinates[row,0], coordinates[row,1]))
    dip_coords = np.append(dip_coords, dip_coords,axis=0)

    H = np.zeros( (np.int(numPart*dim),np.int(numPart*dim)),dtype=complex) 
    vec = np.zeros((np.int(numPart*dim),np.int(numPart*dim)),dtype=np.double) 
    final_eigvals = np.ones(np.int(numPart*dim))*0
    final_eigvecs = np.zeros( (np.int(numPart*dim),np.int(numPart*dim)),dtype=np.double) 
    final_hams = np.zeros( (np.int(numPart*dim),np.int((numPart*dim)**2)),dtype=complex) 
    w = np.ones(np.int(numPart*dim))

    for mode in range(0,np.int(numPart*dim)):
        eigval_hist = np.array([wsp_long, 1, 0],dtype=np.double)*hbar_eVs
        eigVEC_hist = np.zeros( (np.int(numPart*dim),np.int(numPart*dim)),dtype=np.double) 
        ham_hist = np.zeros( (np.int(numPart*dim),np.int((numPart*dim)**2)),dtype=np.double) 
        count = 0
        while (np.abs((eigval_hist[0] - eigval_hist[1]))  > 10**(-prec)):
            denom = ( eigval_hist[2] - eigval_hist[1] ) - ( eigval_hist[1] - eigval_hist[0] )
            eigval_thisround = eigval_hist[2] - ( eigval_hist[2] - eigval_hist[1] )**2 / denom / hbar_eVs
            k = (eigval_thisround*np.sqrt(eps_b))/c

            for i in range(0,np.int(numPart*dim)):
                for j in range(0,np.int(numPart*dim)):
                    r_ij = np.array([ dip_coords[i,0]-dip_coords[j,0] , dip_coords[i,1]-dip_coords[j,1] ])
                    mag_rij = scipy.linalg.norm(r_ij)
                    if i == j and i / numPart < 1: # you're a long dipole
                        H[i,j] = wsp_long**2 
                    if i == j and i / numPart >= 1: # you're a short dipole
                        H[i,j] = wsp_short**2 
                    if mag_rij == 0 and i != j: #if you're on the same particle, you don't couple
                        H[i,j] = 0.0
                    if mag_rij != 0: 
                        xi_hat = base_vectors[i,:]/np.linalg.norm(base_vectors[i,:])
                        xj_hat = base_vectors[j,:]/np.linalg.norm(base_vectors[j,:])
                        if i / numPart < 1:
                            m = m_long
                        if i / numPart >= 1:
                            m = m_short
                        nhat_ij = r_ij / mag_rij
                        xi_dot_nn_dot_xj = np.dot(xi_hat, nhat_ij)*np.dot(nhat_ij, xj_hat)
                        nearField = ( 3.*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat) ) / mag_rij**3
                        intermedField = 1j*k*(3*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij**2 
                        farField = k**2*(xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij
                        g = e**2 * ( ( nearField - intermedField - farField) * np.exp(1j*k*mag_rij))/eps_b
                        H[i,j] = -g / m

            H = H*hbar_eVs**2
            w,v = np.linalg.eig(H) #this solves the eigenvalue problem, producing eigenvalues eval and eigenvectors v. The columns of v, v[:,i] coorespond to the ith mode
            if w[mode] < 0:
                break
            new_eigvals = np.sqrt(np.real(w[mode]))
            eigval_hist = np.append(new_eigvals, eigval_hist)
            ham_hist = np.column_stack((H, ham_hist))
            if count == 6000:
                print('THIS CODE DID NOT CONVERGE GURL </3')
                sys.exit()
            count = count + 1

        final_eigvals[mode] = eigval_hist[0]
        final_eigvecs[:,mode] = np.real(v[:,mode])
        final_hams[:, np.int(numPart*dim*mode) : np.int(numPart*dim*(mode+1))] = ham_hist[:,0 : np.int(numPart*dim)]
    final_eigvecs = final_eigvecs.transpose() #now the rows correspond to the ith mode v[i,:]
    total = np.column_stack((final_eigvals,final_eigvecs)) #arrange a new matrix, where the first column is the evals and the remaining columns are evectors
    total = np.real( total )
    total = total[np.argsort(total[:,0])]
    constants = [eps_b, c, hbar_eVs, m_long, m_short, gam_NR]
    #np.savetxt('total.txt', total)
    #np.savetxt('dip_coords.txt', dip_coords)
    return constants, final_hams, total, dip_coords

calculateEigen()








def check_mode():
    constants, ham_hist, total, dip_coords  = calculateEigen()
    coordinates = np.loadtxt(param['load_coords'])
    x_coords = coordinates[:,0]
    numPart = np.float(len(x_coords))
    dim = param['constants']['dim']
    prec = param['precision']
    for mode in range(0,np.int(numPart*dim)):
        evals = total[mode,0]
        ham = ham_hist[:,mode*np.int(numPart*dim) : mode*np.int(numPart*dim) + np.int(numPart*dim) ]
        vec = total[mode,1:np.int(numPart*dim)+1]
        left_side = np.real(np.matmul(ham, vec))
        right_side = evals**2 * vec
        diff = left_side - right_side
        if diff[0] >= 10**-(1): #used to be 10**-prec
            print 'THIS CODE DID NOT CONVEREGE FOR MODE', mode

#check_mode()




