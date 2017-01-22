#Parham Adiban 1000639446

'''
Using the Runge-Kutta-Fehlberg method to instate an implicit adaptive method of solving the ode's
related to equations of motion. 
'''

#import required packages
from __future__ import division, print_function
import numpy as np
from matplotlib.pyplot import *


# Define constants
G = 6.67408e-11		# Gravitational constant
ME = 1.024e26		# Mass of the Earth
MS = 1.989e30 		# Mass of Sun
axis = 1.496e11 	# semi-major axis
tol = 1e-3			# Tolerance
ecc = 0.0097 		# Earth orbit eccentricity ** pluto 0.2488

x_0 = axis*(1+ecc) #perihelion
v_0 = np.sqrt(G*MS*(2/x_0 - 1/axis)) # Calculate initial velocity using Kepler's equation1

# Define function where the two 2nd order ODE's are expressed as four 1st order ODE's
def f(r):
	x, y, vx, vy = r #Taking poisition and velocity from array r

	Dx = vx #Derivative of x is equal to the velocity in the x-direction
	Dy = vy #Derivative of y is equal to the velocity in the y-direction
	
	R = np.sqrt(x**2+y**2) 
	
	Dvx = -G*MS*x/R**3
	Dvy = -G*MS*y/R**3
	
	return np.array([Dx,Dy,Dvx,Dvy], float) #return an array of velocity and acceleration in x and y direction

# Define limits for time and timestep
t1 = 0 #Start time
t2 = 1e8 #End time


#Empty lists to store position
xpoints = []
ypoints = []
PE = []
KE = []

# Array r including initial position and initial velocity
r = np.array([x_0, 0.0, 0.0, v_0], float)

# Coefficients used to compute the dependent variable argument of f

b21 =   2.500000000000000e-01  #  1/4
b31 =   9.375000000000000e-02  #  3/32
b32 =   2.812500000000000e-01  #  9/32
b41 =   8.793809740555303e-01  #  1932/2197
b42 =  -3.277196176604461e+00  # -7200/2197
b43 =   3.320892125625853e+00  #  7296/2197
b51 =   2.032407407407407e+00  #  439/216
b52 =  -8.000000000000000e+00  # -8
b53 =   7.173489278752436e+00  #  3680/513
b54 =  -2.058966861598441e-01  # -845/4104
b61 =  -2.962962962962963e-01  # -8/27
b62 =   2.000000000000000e+00  #  2
b63 =  -1.381676413255361e+00  # -3544/2565
b64 =   4.529727095516569e-01  #  1859/4104
b65 =  -2.750000000000000e-01  # -11/40

# Coefficients used to compute local truncation error estimate.  These
# come from subtracting a 4th order RK estimate from a 5th order RK
# estimate.

r1  =   2.777777777777778e-03  #  1/360
r3  =  -2.994152046783626e-02  # -128/4275
r4  =  -2.919989367357789e-02  # -2197/75240
r5  =   2.000000000000000e-02  #  1/50
r6  =   3.636363636363636e-02  #  2/55

# Coefficients used to compute 4th order RK estimate

c1  =   1.157407407407407e-01  #  25/216
c3  =   5.489278752436647e-01  #  1408/2565
c4  =   5.353313840155945e-01  #  2197/4104
c5  =  -2.000000000000000e-01  # -1/5

# Set t and x according to initial condition and assume that h starts
# with a value that is as large as possible.

t = t1
h = 10000

#Implement the runge-kutta-fehlberg method
while t<t2:

	k1 = h*f(r)
	k2 = h*f(r+b21*k1)
	k3 = h*f(r+b31*k1+b32*k2)
	k4 = h*f(r+b41*k1+b42*k2+b43*k3)
	k5 = h*f(r+b51*k1+b52*k2+b53*k3+b54*k4)
	k6 = h*f(r+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5)

	# Calculate the difference between 4th and 5th order runge-kutta
	eps = (r1*k1 + r3*k3 + r4*k4 + r5*k5 + r6*k6)/h

	# Calculate error
	z = np.sqrt(eps[0]**2 + eps[1]**2)

	# Check for error to be below the tolerance level
	if z <= tol:

		# Update time
		t += h

		# Update position
		r += c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5

		#Store position values
		xpoints.append(r[0])
		ypoints.append(r[1])

		# Calculate the velocity
		v = np.sqrt(r[2]**2 + r[3]**2)

		# Calculate the position
		c = np.sqrt(r[0]**2 + r[1]**2)

		# Calculate and store Potential and Kinetic Energies
		PE.append(-G*MS*ME/c)
		KE.append(0.5*ME*v**2)
	
	# Update h value
	h = h * min(max((tol/2/z)**0.25, .1), 4.0)

#Animation of earth orbiting the sun
figure()
gca().set_axis_bgcolor('black')
for i in range(len(xpoints)):
	cla()
	plot(xpoints, ypoints, 'w')
	plot(xpoints[i], ypoints[i], 'bo', markersize = 15)
	plot(0, 0, 'yo', markersize = 50)
	title('Orbit of planet Earth')
	xlabel('x(m)')
	ylabel('y(m)')
	pause(0.001)
show()








