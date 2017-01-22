#Parham Adiban 1000639446
'''
Using adaptive RK4 to solve a basic 3 body problem. Adapted from Exercise 8.16, Newman.
'''

#import required packages
from __future__ import division, print_function
import numpy as np
from matplotlib.pyplot import *

#Define Constants
G = 1							# Gravitational constants
m1, m2, m3 = 150, 200, 250		# Mass of the stars
delta = 1e-3					# Margin of error

# Initial conditions of all 3 stars
r = np.array([[3., 1.], [-1., -2.], [-1., 1.], [0, 0], [0, 0], [0, 0]])

t1 = 0 # Start time
t2 = 2 # End time
N = 10000 # Number of samples
h = (t2-t1)/N #Timestep

# Lists to store position of the three stars
x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []

# Define function where the two 2nd order ODE's are expressed as four 1st order ODE's
def f(r_):

	r1, r2, r3, v1, v2, v3 = r_

	#Derivative of position is equal to the velocity
	dr1 = v1
	dr2 = v2
	dr3 = v3

	# Calculate difference between positions
	diff1 = r1 - r2
	diff2 = r1 - r3
	diff3 = r2 - r3

	# Derivative of velocity is equal to the acceleration
	dv1 = G*m2*(-diff1)/(diff1[0]**2 + diff1[1]**2)**1.5 + G*m3*(-diff2)/(diff2[0]**2 + diff2[1]**2)**1.5
	dv2 = G*m1*(diff1)/(diff1[0]**2 + diff1[1]**2)**1.5 + G*m3*(-diff3)/(diff3[0]**2 + diff3[1]**2)**1.5
	dv3 = G*m1*(diff2)/(diff2[0]**2 + diff2[1]**2)**1.5 + G*m2*(diff3)/(diff3[0]**2 + diff3[1]**2)**1.5

	return np.array([dr1, dr2, dr3, dv1, dv2, dv3], float) #return an array of velocity and acceleration

#Function to implement the rk4 method
def rk4(r, h):
	k1 = h*f(r)
	k2 = h*f(r+0.5*k1)
	k3 = h*f(r+0.5*k2)
	k4 = h*f(r+k3)
	return (k1+2*k2+2*k3+k4)/6



t = t1 		# Initialize t for the while loop
ro = 1		# Start with ro = 1 for our error calculation
time = [t]	# Create list to store the time at every step.

# Function to find the difference between two arrays in order
# to calculate the error
def difference(r1,r2):
	r1 = r1[:3]
	r2 = r2[:3]
	return np.sqrt(sum(sum((r1-r2)**2)))

#Implement the adaptive step method
while t<t2:
	
	if ro<2: #Set a maximum for how large ro can get
		h = h*ro**(1/4) #equation for h'
	else:
		h*=2

	# Prevent h to get too large
	if h > delta:
		h = delta

	# estimating ro
	ri = r + rk4(r, h)
	ri += rk4(ri, h)			
	rj = r + rk4(r, 2*h)

	ro = 30*h*delta/difference(ri, rj)  #calculate new ro

	#Condition for accepted value ro for adaptive timestep
	if ro>1:
		t += 2*h #add time increment
		r = ri
		x1.append(r[0][0])
		y1.append(r[0][1])
		x2.append(r[1][0])
		y2.append(r[1][1])
		x3.append(r[2][0])
		y3.append(r[2][1])
		time.append(t)
		
# #Animation of the 3 stars
# # *************For full animation change t2 value to 10 before running!!!***********
# for i in range(0, len(time), 10):
# 	cla()
# 	plot(x1, y1, 'w')
# 	plot(x2, y2, 'w')
# 	plot(x3, y3, 'w')
# 	plot(x1[i], y1[i], 'bo', label = 'Star 1', markersize = 10)
# 	plot(x2[i], y2[i], 'ro', label = 'Star 2', markersize = 10)
# 	plot(x3[i], y3[i], 'go', label = 'Star 3', markersize = 10)
# 	title('Motion of the Stars', size = 20)
# 	xlabel('x', size = 15)
# 	ylabel('y', size = 15)
# 	legend()
# 	pause(0.01)

#Plotting the trajectory of the 3 stars
figure()
title('Trajectory of the 3 stars for t = 2')
plot(x1, y1, label = 'Star 1')
plot(x2, y2, label = 'Star 2')
plot(x3, y3, label = 'Star 3')
xlabel('x')
ylabel('y')
legend()
show()



