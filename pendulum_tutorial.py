#-----Implementation of a pendulum----

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def calculate_trajectory(theta_init,theta_d_init,n_iter=1000,dt=0.01,g=10,l=1):
	phase_traject=np.zeros((n_iter,2))#phase-space trajectory
	phase_traject[0,:]=np.array([theta_init,theta_d_init])

	for i in range(n_iter-1):
		theta_dd=-g/l*np.sin(phase_traject[i,0])
		phase_traject[i+1,1]=phase_traject[i,1]+dt*theta_dd
		phase_traject[i+1,0]=phase_traject[i,0]+dt*phase_traject[i,1]
	return phase_traject

def get_energy(theta,theta_d,m=1,l=1,g=10):
	#the height of the mass
	y=l*np.cos(theta)
	#potential energy
	e_pot=-m*g*y
	#kinetic energy
	e_kin=m/2*(l*theta_d)**2
	return e_pot,e_kin

def draw_pendulum(theta,w=200,h=200,m=1,l=1):
	#create image with width=w and height=h
	img = Image.new("RGB", (w, h), "white")
	#convert the length of the pendulum to some image units
	L=int(0.4*h*l)
	#define a diameter for the penulums mass
	d=int(0.02*h)*m**(1/3)
	#create the draw objects of the image
	draw = ImageDraw.Draw(img)
	#calculate the cartesian coordinates
	x0=int(w/2)
	y0=int(h/2)
	x=x0+L*np.sin(theta)
	y=y0+L*np.cos(theta)
	#draw the pendulum
	draw.line([(x0,y0),(x,y)],fill=(0,0,0),width=1)
	draw.ellipse([(x-d,y-d),(x+d,y+d)], fill=(0,0,255), outline=None)
	return img

def render_traject(phase_traject,m=1,l=1,g=10,save_path='',take_frame_every=1):
	frames=[]#here we clollect the frames
	for i in range(phase_traject.shape[0]):
		if i%take_frame_every==0:
			#get the i-th angle and angular velocity
			theta=phase_traject[i,0]
			theta_d=phase_traject[i,1]
			#draw the corresponding image and add to our frame list
			img=draw_pendulum(theta,w=200,h=200,m=m,l=l)
			frames.append(img)

	frames[0].save(save_path+'pendulum_tutorial.gif',
	               save_all=True,
	               append_images=frames[1:],
	               duration=40,
	               loop=0)

	e_pot_0,e_kin_0=get_energy(phase_traject[0,0],phase_traject[0,1],m,l,g)#initial energy
	e_pot_final,e_kin_final=get_energy(phase_traject[-1,0],phase_traject[-1,1],m,l,g)#final energy
	print('initial energy: '+str(e_pot_0+e_kin_0))
	print('final energy: '+str(e_pot_final+e_kin_final))


#---initial condition ---
theta_init=np.pi/3
theta_d_init=0

dt=0.01
frames_per_second=20
take_frame_every=int(1/(dt*frames_per_second))

#--example of draw_pendulum()----
# img=draw_pendulum(theta_init)
# img.show()

#--example of get_energy()----
# e_pot,e_kin=get_energy(theta_init,theta_d_init)
# print(e_pot)
# print(e_kin)

#---example of calculate_traject()
# phase_traject=calculate_trajectory(theta_init,theta_d_init,dt=dt,n_iter=1000)
# plt.plot(phase_traject[:,0])
# plt.ylabel('angle')
# plt.show()

# #--example: get energy of trajectory
# phase_traject=calculate_trajectory(theta_init,theta_d_init,dt=dt,n_iter=1000)
# total_energy=[]
# for i in range(phase_traject.shape[0]):
# 	e_pot,e_kin=get_energy(phase_traject[i,0],phase_traject[i,1])
# 	total_energy.append(e_pot+e_kin)	
# plt.plot(total_energy)
# plt.ylabel('total energy')
# plt.show()


#---example of render_traject()
phase_traject=calculate_trajectory(theta_init,theta_d_init,dt=dt,n_iter=1000)
render_traject(phase_traject,take_frame_every=take_frame_every)



