#Parham Adiban 1000639446

'''Use the Adams-Moulton method to predict the motion of a double pendulum'''

#import required packages
from __future__ import division, print_function
from numpy import pi, sin, cos, array, arange, copy
from matplotlib.pyplot import *


# Define constants
m = 1 # Mass of bobs
l = 0.4 # Length of pendulum arm
g = 9.81 # Acceleration due to gravity

a = 0 # Start time
b = 100 # Start time
N = 100000 # Number of samples
h = (b-a)/N # Timestep

# Initial conditions of the pendulum
r = array([pi/2, pi/2, 0, 0], float)

# Constants for the Adams-Bashforth Method
b1 = 3./8
b2 = 19./24
b3 = -5./24
b4 = 1./24

c1 = 55./24
c2 = -59./24
c3 = 37./24
c4 = -9./24

# Function to apply 4th orher Runge-Kutta
def rk4(r_, h=h):
	k1 = h*f(r_)
	k2 = h*f(r_+0.5*k1)
	k3 = h*f(r_+0.5*k2)
	k4 = h*f(r_+k3)
	return (k1+2*k2+2*k3+k4)/6

# Function deifnining motion of the double pendulum
def f(r_):
	theta1, theta2, omega1, omega2  = r_

	dtheta1 = omega1
	dtheta2 = omega2

	domega1 = -(omega1**2*sin(2*theta1 - 2*theta2) +\
			 2*omega2**2*sin(theta1 - theta2) +\
			 g/l*(sin(theta1-2*theta2) + 3*sin(theta1)) ) \
				/(3 - cos(2*theta1 - 2*theta2)) 
	
	domega2 = ( 4*omega1**2*sin(theta1-theta2)\
			+ omega2**2*sin(2*theta1-2*theta2) \
			+ 2*g/l*(sin(2*theta1 - theta2) - sin(theta2)))\
			/(3 - cos(2*theta1 - 2*theta2))

	return array([dtheta1, dtheta2, domega1, domega2], float)

# Function to calculate the energy of the double pendulum
def Energy(r):

	theta1,theta2,omega1,omega2 = r
	V = -m*g*l*(2*cos(theta1) + cos(theta2))
	T = m*l**2*(omega1**2 + 1./2*omega2**2 + omega1*omega2*cos(theta1 - theta2))
	
	return T + V

# List to store position (s) and energy
s = []
energy = []

# Calculate first 3 points using runge-kutta method
for i in range(3):
	s.append(r)
	energy.append(Energy(r))
	r = r + rk4(r)

# Apply the Adams-Moulton method
for j in range(3, N):
	s.append(r)
	energy.append(Energy(r))
	# Calculate weight function using Adams-Bashforth method
	w = r + h*(c1*f(s[j]) + c2*f(s[j-1]) + c3*f(s[j-2]) + c4*f(s[j-3]))
	# Use weight function to calculate Adams-Moulton method
	r = r + h*(b1*f(w) + b2*f(s[j]) + b3*f(s[j-1]) + b4*f(s[j-2]))

# Slice stored value to store the two angles
s = array(s)
t1 = s[:,0]
t2 = s[:,1]

# Plot the angles of the pendulum
figure(1)
plot(t1, label = r'$\theta_1$')
plot(t2, label = r'$\theta_2$')
legend()
title('Angles of the two pendulums')
xlabel('time(ms)')
ylabel('Radians')

# Plot the energy of the system
figure(2)
plot(energy)
title('Fluctuation in Energy')
xlabel('time(ms)')
ylabel('Energy (J)')
show()

# # Animate the motion of the pendulum
# figure(3)
# for i in range(0, N, 50):
# 	cla()
# 	plot(1,1, 'bo', markersize = 15)
# 	ylim([0, 1.3])
# 	xlim([0, 2])
# 	x1 = 1+0.4*sin(t1[i])
# 	y1 = 1-0.4*cos(t1[i])
# 	x2 = x1 + 0.4*sin(t2[i])
# 	y2 = y1 - 0.4*cos(t2[i])
# 	plot(x1, y1, 'ko')
# 	plot([1, x1], [1, y1], 'b-')
# 	plot(x2, y2, 'ko')
# 	plot([x1, x2], [y1, y2], 'b-')
# 	title('Pendulum')
# 	pause(0.001)
# show()

# Plot the trajectory of the bobs
figure(4)
title('Trajectory of the Pendulum')
x1 = 1+0.4*sin(t1)
y1 = 1-0.4*cos(t1)
x2 = x1 + 0.4*sin(t2)
y2 = y1 - 0.4*cos(t2)
plot(x1, y1)
plot(x2, y2)
show()




