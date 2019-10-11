#Implementation of a pendulum
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import os

def get_theta_dd(theta,l,g):
	return -g/l*np.sin(theta)

def calculate_trajectory(theta_init,theta_d_init,n_iter=1000,dt=0.01,g=10,l=1,midpoint=True):
	phase_traject=np.zeros((n_iter,2))#phase-space trajectory
	phase_traject[0,:]=np.array([theta_init,theta_d_init])
	if midpoint:
		print('calculating trajectory using explicite midpoint method...')
	else:
		print('calculating trajectory using explicite Euler method...')
	for i in range(n_iter-1):
		# theta_dd=-g/l*np.sin(phase_traject[i,0])
		theta_dd=get_theta_dd(phase_traject[i,0],l,g)

		if not midpoint:
		#explicite Euler
			phase_traject[i+1,1]=phase_traject[i,1]+dt*theta_dd
			phase_traject[i+1,0]=phase_traject[i,0]+dt*phase_traject[i+1,1]
		else:
		#explicite midpoint method
			theta_d1=phase_traject[i,1]+dt/2*theta_dd
			theta1=phase_traject[i,0]+dt/2*theta_d1
			theta_dd1=get_theta_dd(theta1,l,g)
			phase_traject[i+1,1]=phase_traject[i,1]+dt*theta_dd1
			phase_traject[i+1,0]=phase_traject[i,0]+dt/2*(phase_traject[i+1,1]+phase_traject[i,1])

	return phase_traject


def get_energy(theta,theta_d,m,l,g):
	y=l*np.cos(theta)
	e_pot=-m*g*y
	e_kin=m/2*(l*theta_d)**2
	return e_pot,e_kin

def pil_list_to_cv2(pil_list):
	#converts a list of pil images to a list of cv2 images
	png_list=[]
	for pil_img in pil_list:
		pil_img.save('trash_image.png',format='png')
		png_list.append(cv2.imread('trash_image.png'))
	os.remove('trash_image.png')
	return png_list

def generate_video(cv2_list,path='pendulum.avi',fps=10): 
	#makes a video from a given cv2 image list
	if len(cv2_list)==0:
		raise ValueError('the given png list is empty!')
	video_name = path
	frame=cv2_list[0] 
	# setting the frame width, height width 
	# the width, height of first image 
	height, width, layers = frame.shape   
	video = cv2.VideoWriter(video_name, 0, fps, (width, height))  
	# Appending the images to the video one by one 
	for cv2_image in cv2_list:  
	    video.write(cv2_image) 
	# Deallocating memories taken for window creation 
	cv2.destroyAllWindows()  
	video.release()  # releasing the video generated

def rotation(alpha,v):
	c=np.cos(alpha)
	s=np.sin(alpha)
	R=np.array([[c,-s],[s,c]])
	return np.dot(R,v)

def draw_arrow(draw,start_coord,end_coord,flank_length=2):
	#start_coord and end_coord should be a 2-d numpy vector
	delta=end_coord-start_coord
	flank_1=rotation(3*np.pi/4,delta)
	flank_2=rotation(-3*np.pi/4,delta)
	flank_1=flank_length*flank_1/np.linalg.norm(flank_1)
	flank_2=flank_length*flank_2/np.linalg.norm(flank_2)
	flank_point_1=end_coord+flank_1
	flank_point_2=end_coord+flank_2
	draw.line([(start_coord[0],start_coord[1]),(end_coord[0],end_coord[1]),(int(flank_point_1[0]),int(flank_point_1[1])),(end_coord[0],end_coord[1]),(int(flank_point_2[0]),int(flank_point_2[1]))], fill=(0,0,0))#drawing arrow

def distance_in_phase_space(phase_point):
	phase_point[0]=np.mod(phase_point[0],2*np.pi)
	if phase_point[0]>np.pi:
		phase_point[0]=2*np.pi-phase_point[0]
	return np.linalg.norm(phase_point)

def render_phase_traject(phase_traject,img_res=1,m=1,l=1,g=10,save_path='trash_figures/',take_frame_every=1,second_phase_traject=None,draw_phase=True):
	frames=[]
	e_pot=[]#the potential energy of each of the masses: e_pot=-m*g*y
	e_kin=[]#the kinetic energy of each of the masses: e_kin=m*l**2*theta_d**2/2
	h=int(img_res*200)
	if draw_phase:
		w=2*h
		w_34=int(3*w/4)
	else:
		w=h
	x0=int(h/2)
	y0=int(h/2)
	h_red=int(0.4*h)
	L=h_red
	d=int(0.02*h)
	d_4=d/4
	e_pot_0,e_kin_0=get_energy(phase_traject[0,0],phase_traject[0,1],m,l,g)
	energy=e_pot_0+e_kin_0
	print('initial energy: '+str(energy))
	prev_points=[]
	prev_phase=[]
	if second_phase_traject is not None:
		distance_scale=0.1
		prev_distances=[distance_scale*distance_in_phase_space(second_phase_traject[0,:]-phase_traject[0,:])]
	max_points=500
	# max_theta_d=1.2*np.max(np.abs(phase_traject[:,1]))
	max_theta_d=10

	for i in range(phase_traject.shape[0]):
		if i%take_frame_every==0:
			theta=phase_traject[i,0]
			theta_d=phase_traject[i,1]
			prev_phase.append((theta,theta_d))
			#----transform to cartesian coordinates---
			x=x0+L*np.sin(theta)
			y=y0+L*np.cos(theta)

			prev_points.append(np.array([x,y]))
			#---draw the image ----
			img = Image.new("RGB", (w, h), "white")
			draw = ImageDraw.Draw(img)

			n_prev=min(max_points,len(prev_points))
			if draw_phase:
				draw_arrow(draw,np.asarray([w_34,int(3*h/4)]),np.asarray([w_34,int(1*h/4)]),flank_length=int(h/50))
				draw_arrow(draw,np.asarray([int(5*h/4),int(h/2)]),np.asarray([int(7*h/4),int(h/2)]),flank_length=int(h/50))
				font = ImageFont.truetype("arial.ttf", int(h/20))
				draw.text((w_34-h/7,int(2*h/16)), 'angular velocity', font=font, fill=(0,0,0))
				draw.text((int(14*h/8),int(h/2+h/30)), 'angle', font=font, fill=(0,0,0))
			for k in range(n_prev-1):
				idx=n_prev-k
				point=prev_points[-idx]
				x=point[0]
				y=point[1]
				point=prev_points[-idx+1]
				xx=point[0]
				yy=point[1]
				intensity=int(255*(1-0.99**idx))
				draw.line([(x,y),(xx,yy)],fill=(intensity,intensity,255),width=2)
				if draw_phase:
					if np.abs((prev_phase[-idx][0]+np.pi)%(2*np.pi)-(prev_phase[-idx+1][0]+np.pi)%(2*np.pi))<np.pi:
						phase_x=w_34+x0*((prev_phase[-idx][0]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
						phase_y=y0+y0*prev_phase[-idx][1]/max_theta_d
						phase_xx=w_34+x0*((prev_phase[-idx+1][0]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
						phase_yy=y0+y0*prev_phase[-idx+1][1]/max_theta_d
						draw.line([(phase_x,phase_y),(phase_xx,phase_yy)],fill=(255,intensity,intensity),width=2)
						if idx==2:
							draw.ellipse([(phase_xx-d,phase_yy-d),(phase_xx+d,phase_yy+d)], fill=(0,0,0), outline=None)

			if second_phase_traject is not None:
				x1=x0+L*np.sin(second_phase_traject[i,0])
				y1=y0+L*np.cos(second_phase_traject[i,0])
				draw.line([(x0,y0),(x1,y1)],fill=(255,0,0),width=1)
				draw.ellipse([(x1-d,y1-d),(x1+d,y1+d)], fill=(255,0,0), outline=None)

				distance=distance_scale*distance_in_phase_space(second_phase_traject[i,:]-phase_traject[i,:])
				prev_distances.append(distance)
				text=['Distance in phase space: '+str(distance)[:4]]
				dy=2
				font = ImageFont.truetype("arial.ttf", int(h/20))
				for k in range(len(text)):
					draw.text((int(0.05*h),int(3+0.05*h+dy)), text[k], font=font, fill=(0,0,0))
					dy+=1.1*font.getsize(text[k])[1]
				idx=0
				dx=1
				bias=int(h/10)
				bias_y=-10
				draw.line([(bias,int(bias_y+h/4*2)),(bias,int(h/6)),(bias-2,int(h/6+2)),(bias,int(h/6)),(bias+2,int(h/6+2))], fill=(0,0,0))#drawing arrow
				draw.ellipse([(bias-2,int(bias_y+h/4*(2-prev_distances[-1])-2)),(bias+2,int(bias_y+h/4*(2-prev_distances[-1])+2))], fill=(0,0,0), outline=None)
				while ((bias+dx*idx)<(h-bias) and idx<len(prev_distances)-1):
					intensity=int(255*(1-0.99**idx))
					draw.line([(bias+dx*idx,int(bias_y+h/4*(2-prev_distances[-idx-1]))),(bias+dx*(idx+1),int(bias_y+h/4*(2-prev_distances[-idx-2])))],fill=(intensity,intensity,255),width=2)
					idx+=1


			draw.line([(x0,y0),(x,y)],fill=(0,0,0),width=1)
			draw.ellipse([(x-d,y-d),(x+d,y+d)], fill=(0,0,0), outline=None)
			frames.append(img)

			#----calculate the energies----
			e_pot_i,e_kin_i=get_energy(theta,theta_d,m,l,g)
			e_pot.append(e_pot_i)
			e_kin.append(e_kin_i)

	frames[0].save(save_path+'pendulum.gif',
	               save_all=True,
	               append_images=frames[1:],
	               duration=40,
	               loop=0)
	cv2_list=pil_list_to_cv2(frames)
	generate_video(cv2_list,path=save_path+'pendulum.avi',fps=1000/40)

	e_pot=np.asarray(e_pot)
	e_kin=np.asarray(e_kin)
	print('final energy: '+str(e_pot[-1]+e_kin[-1]))
	return e_pot,e_kin


#---initial condition ---
theta_init=3*np.pi/8
theta_d_init=0
dt=0.005
frames_per_second=20
take_frame_every=int(1/(dt*frames_per_second))
n_iter=3000
#---calculate phase trajectory
phase_traject=calculate_trajectory(theta_init,theta_d_init,dt=dt,n_iter=n_iter)
# second_phase_traject=calculate_trajectory(theta_init+1e-2,theta_d_init,dt=dt,n_iter=n_iter)
e_pot,e_kin=render_phase_traject(phase_traject,img_res=2,take_frame_every=take_frame_every,second_phase_traject=None,draw_phase=True)
# plt.plot(e_pot+e_kin)
# plt.ylabel('energy')
# plt.show()





