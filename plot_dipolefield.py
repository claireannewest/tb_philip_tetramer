import numpy as np
import matplotlib.pyplot as plt
from claires_tight import calculateEigen
import yaml

open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)

def seeVectors(mode):
	ham_hist, total, dip_coords  = calculateEigen()    
	w = total[:,0]
	origin = np.loadtxt(param['load_coords'], skiprows=1)[:,0:2]
	base_load = np.loadtxt(param['dipole_directions'])
	base_vectors = np.vstack((base_load[:,0], base_load[:,1]))
	mode1 = np.vstack((base_vectors[:,0], np.zeros((7,2))))
	mode2 = np.vstack((np.zeros((1,2)), base_vectors[:,1], np.zeros((6,2))))
	mode3 = np.vstack((np.zeros((2,2)), base_vectors[:,2], np.zeros((5,2))))
	mode4 = np.vstack((np.zeros((3,2)), base_vectors[:,3], np.zeros((4,2))))
	mode5 = np.vstack((np.zeros((4,2)), base_vectors[:,4], np.zeros((3,2))))
	mode6 = np.vstack((np.zeros((5,2)), base_vectors[:,5], np.zeros((2,2))))
	mode7 = np.vstack((np.zeros((6,2)), base_vectors[:,6], np.zeros((1,2))))
	mode8 = np.vstack((np.zeros((7,2)), base_vectors[:,7]))
	v = total[:,1:]
	y_coords = origin[:,0]
	z_coords = origin[:,1]
	mag_mode = v[mode,0]*mode1 + v[mode,1]*mode2 + v[mode,2]*mode3 + v[mode,3]*mode4 + v[mode,4]*mode5 + v[mode,5]*mode6 + v[mode,6]*mode7 + v[mode,7]*mode8
	return mag_mode
seeVectors(mode=0)

def seeFields(mode):
	just_y1 = []; just_z1 = []; just_E1 = []
	just_y2 = []; just_z2 = []; just_E2 = []
	just_y3 = []; just_z3 = []; just_E3 = []
	just_y4 = []; just_z4 = []; just_E4 = []
	mag_mode = seeVectors(mode)
	origin = np.loadtxt(param['load_coords'], skiprows=1)[:,0:2]
	origin1 = np.array([0, origin[0,0], origin[0,1]])
	origin2 = np.array([0, origin[1,0], origin[1,1]])
	origin3 = np.array([0, origin[2,0], origin[2,1]])
	origin4 = np.array([0, origin[3,0], origin[3,1]])
	origin_all = np.concatenate((origin, origin))

	loadp = str('output/') + str('rhomb_mode') + str(mode) + str('.txt')
	p = np.loadtxt(loadp,skiprows=1)[:,2:4]
	p1 = np.array([0, p[0,0], p[0,1]])
	p2 = np.array([0, p[1,0], p[1,1]])
	p3 = np.array([0, p[2,0], p[2,1]])
	p4 = np.array([0, p[3,0], p[3,1]])	

	ymin = min(origin[:,0])-2E-5; ymax = max(origin[:,0])+2E-5
	zmin = min(origin[:,1])-2E-5; zmax = max(origin[:,1])+2E-5

	x = -20.e-06; y = np.linspace(ymin, ymax, 51 ); z = np.linspace(zmin, zmax, 51 )
	x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
	all_points = np.column_stack((np.ravel(x_grid), np.ravel(y_grid), np.ravel(z_grid)))

	E1_field = np.zeros((len(all_points[:,0]), 3))
	E2_field = np.zeros((len(all_points[:,0]), 3))
	E3_field = np.zeros((len(all_points[:,0]), 3))
	E4_field = np.zeros((len(all_points[:,0]), 3))

	for coord in range(0, len(all_points[:,0])):
		r1 = all_points[coord] - origin1
		rhat1 = r1/np.linalg.norm(r1)
		nearField = ( 3*rhat1*( np.dot(rhat1,p1) ) - p1 ) / (np.linalg.norm(r1))**3
		E1_field[coord, :] = nearField

		r2 = all_points[coord] - origin2
		rhat2 = r2/np.linalg.norm(r2)
		nearField = ( 3*rhat2*( np.dot(rhat2,p2) ) - p2 ) / (np.linalg.norm(r2))**3
		E2_field[coord, :] = nearField
	
		r3 = all_points[coord] - origin3
		rhat3 = r3/np.linalg.norm(r3)
		nearField = ( 3*rhat3*( np.dot(rhat3,p3) ) - p3 ) / (np.linalg.norm(r3))**3
		E3_field[coord, :] = nearField

		r4 = all_points[coord] - origin4
		rhat4 = r4/np.linalg.norm(r4)
		nearField = ( 3*rhat4*( np.dot(rhat4,p4) ) - p4) / (np.linalg.norm(r4))**3
		E4_field[coord, :] = nearField
	
	Etot = E1_field[:,0] + E2_field[:,0] + E3_field[:,0] + E4_field[:,0]

	plt.subplot(1,8,mode+1)
	plt.scatter(all_points[:,1], all_points[:,2], c=Etot, 
        s=10, cmap='seismic',
        vmin = min(Etot),
        vmax = max(Etot)
        )
	plt.scatter(origin[:,0], origin[:,1],c='black')
	plt.quiver(origin_all[:,0], origin_all[:,1], mag_mode[:,0], mag_mode[:,1], pivot='mid', width=0.1,scale=2.0, headlength=4,headwidth=5.8,minshaft=4.1, minlength=.1)
	plt.xlim([ymin, ymax])
	plt.ylim([zmin, zmax])
	plt.axis('off')

for mode in range(0,8):
    seeFields(mode=mode)
    print mode
plt.show()


