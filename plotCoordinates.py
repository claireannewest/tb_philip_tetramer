import numpy as np
import matplotlib.pyplot as plt
import yaml
open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)

load_coordinates = np.loadtxt(param['load_coords'])
base_vectors = np.loadtxt(param['dipole_directions'])
coordinates = np.concatenate((load_coordinates, load_coordinates),axis=0)

dipoles_x = []
dipoles_y = []
dipoles_this_iter_x = []
dipoles_this_iter_y =[]

fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')   

for i in range(0,4):
	#plt.arrow(coordinates[i,0],coordinates[i,1], base_vectors[i,0], 
	#	base_vectors[i,1],head_width=10,color='k')
	y1 = coordinates[i,1] + base_vectors[i,1]
	x1 = coordinates[i,0] + base_vectors[i,0]
	#plt.scatter(coordinates[i,0], coordinates[i,1],color='k')
	#plt.scatter(x1, y1, color='r') 
	if y1-coordinates[i,1] == 0.0:
		m = 1e30
	else:
		m = -((y1-coordinates[i,1])/(x1-coordinates[i,0]))**-1
	print x1
	b = coordinates[i,1]-m*coordinates[i,0]
	refl_x = ((1-m**2)*x1+2*m*y1-2*m*b)/(m**2+1)
	refl_y = ((m**2-1)*y1+2*m*x1+2*b)/(m**2+1)
	#plt.scatter(refl_x, refl_y, color='b')
	dipoles_this_iter_x = [x1, refl_x]
	dipoles_this_iter_y = [y1, refl_y]

	dipoles_x = np.append(dipoles_this_iter_x, dipoles_x)
	dipoles_y = np.append(dipoles_this_iter_y, dipoles_y)
plt.scatter(dipoles_x, dipoles_y, color='r')

plt.xlim([-500,500])
plt.ylim([-500,500])
plt.show()

file = open(str('output/basevecs_1108.txt'),'w')
for j in range(0, len(dipoles_x)):
    file.write( str(dipoles_x[j]) + '\t' + str(dipoles_y[j]) + '\n')
file.close()    
# np.savetxt('long_dipoles.txt', long_dipoles)
# np.savetxt('short_dipoles.txt', short_dipoles)

