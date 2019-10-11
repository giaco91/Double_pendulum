#Let's implement a chaos pendulum (double pendulum)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import os
import librosa
import librosa.display
from routines import *


def pil_list_to_cv2(pil_list):
	#converts a list of pil images to a list of cv2 images
	png_list=[]
	for pil_img in pil_list:
		pil_img.save('trash_image.png',format='png')
		png_list.append(cv2.imread('trash_image.png'))
	os.remove('trash_image.png')
	return png_list

def generate_video(cv2_list,path='double_pendulum.avi',fps=10): 
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

def get_theta_dd(theta1,theta2,theta1_d,theta2_d,m1,m2,l1,l2,g):
	#----theta1_dd-----
	num1=-g*((2*m1+m2)*np.sin(theta1)+m2*np.sin(theta1-2*theta2))
	num2=-2*np.sin(theta1-theta2)*m2*(theta2_d**2*l2+theta1_d**2*l1*np.cos(theta1-theta2))
	denum1=2*m1+m2-m2*np.cos(2*theta1-2*theta2)
	denum=l1*denum1
	theta1_dd=(num1+num2)/denum
	#----theta2_dd----
	num1=2*np.sin(theta1-theta2)
	num2=theta1_d**2*l1*(m1+m2)+g*(m1+m2)*np.cos(theta1)+theta2_d**2*l2*m2*np.cos(theta1-theta2)
	denum=l2*denum1
	theta2_dd=num1*num2/denum
	
	return theta1_dd,theta2_dd

def explicite_euler(dt,theta1,theta2,theta1_d,theta2_d,m1,m2,l1,l2,g):
	theta1_dd,theta2_dd=get_theta_dd(theta1,theta2,theta1_d,theta2_d,m1,m2,l1,l2,g)
	return theta1+dt*theta1_d,theta2+dt*theta2_d,theta1_d+dt*theta1_dd,theta2_d+dt*theta2_dd

def calculate_trajectory(n_iter,dt,theta1_init,theta2_init,theta1_d_init,theta2_d_init,m1=1,m2=1,l1=1,l2=0.5,g=10,add_energy=None):
	phase_traject=np.zeros((n_iter,4))#phase-space trajectory
	phase_traject[0,:]=np.array([theta1_init,theta2_init,theta1_d_init,theta2_d_init])
	for i in range(n_iter-1):
		if (i+1)%100000==0:
			print('progress: '+str(i)+'/'+str(n_iter-1))
		if add_energy is not None:
			phase_traject[i,2]+=np.sign(phase_traject[i,2])*add_energy
		#---explicite Euler ----
		# theta1,theta2,theta1_d,theta2_d=explicite_euler(dt,phase_traject[i,0],phase_traject[i,1],phase_traject[i,2],phase_traject[i,3],m1,m2,l1,l2,g)
		# phase_traject[i+1,:]=np.array([theta1,theta2,theta1_d,theta2_d])

		#---explicite midpoint method ----
		theta1_d_i=phase_traject[i,2]
		theta2_d_i=phase_traject[i,3]
		theta11,theta22,theta11_d,theta22_d=explicite_euler(dt/2,phase_traject[i,0],phase_traject[i,1],theta1_d_i,theta2_d_i,m1,m2,l1,l2,g)
		theta11_dd,theta22_dd=get_theta_dd(theta11,theta22,theta11_d,theta22_d,m1,m2,l1,l2,g)
		theta1_dd,theta2_dd=get_theta_dd(theta11,theta22,theta11_d,theta22_d,m1,m2,l1,l2,g)
		theta1_d=theta1_d_i+dt*theta1_dd
		theta2_d=theta2_d_i+dt*theta2_dd
		phase_traject[i+1,:]=np.array([phase_traject[i,0]+dt/2*(theta1_d_i+theta1_d),phase_traject[i,1]+dt/2*(theta2_d_i+theta2_d),theta1_d,theta2_d])
	return phase_traject

def get_energy(theta1,theta2,theta1_d,theta2_d,m1,m2,l1,l2,g):
	y1=l1*np.cos(theta1)
	y2=y1+l2*np.cos(theta2)
	e_pot=np.array([-m1*g*y1,-m2*g*y2])
	e_kin_1=m1/2*(l1*theta1_d)**2
	e_kin_2=(l1*theta1_d)**2
	e_kin_2+=(l2*theta2_d)**2
	e_kin_2+=2*l1*l2*theta1_d*theta2_d*(np.cos(theta1)*np.cos(theta2)+np.sin(theta1)*np.sin(theta2))
	e_kin_2*=m2/2
	e_kin=np.array([e_kin_1,e_kin_2])
	return e_pot,e_kin

def get_corrected_theta2_d(energy,theta1,theta2,theta1_d,theta2_d,m1,m2,l1,l2,g):
	y1=l1*np.cos(theta1)
	y2=y1+l2*np.cos(theta2)
	a=-l2**2
	b=-2*l1*l2*theta1_d*(np.cos(theta1)*np.cos(theta2)+np.sin(theta1)*np.sin(theta2))
	c=2*(energy+m1*g*y1+m2*g*y2-m1/2*(l1*theta1_d)**2)/m2-(l1*theta1_d)**2
	sqrt_term=b**2-4*a*c
	if sqrt_term<0:
		print(sqrt_term)
		#raise ValueError('the sqrt_term is negative!')
		sqrt_term=0
	else:
		sqrt_term=np.sqrt(sqrt_term)
	theta2_d_corrected=(-b+sqrt_term)/(2*a)
	if np.sign(theta2_d_corrected)!=np.sign(theta2_d):
		theta2_d_corrected=(-b-sqrt_term)/(2*a)
	return theta2_d_corrected

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
	phase_point[0:2]=np.mod(phase_point[0:2],2*np.pi)
	if phase_point[0]>np.pi:
		phase_point[0]=2*np.pi-phase_point[0]
	if phase_point[1]>np.pi:
		phase_point[1]=2*np.pi-phase_point[1]
	return np.linalg.norm(phase_point)

def render_phase_traject(phase_traject,img_res=1,m1=1,m2=1,l1=1,l2=0.5,g=10,save_path='trash_figures/',take_frame_every=1,second_phase_traject=None,draw_phase=True,draw_marker=True,max_points=500,frames_per_second=20,show_energy=False):
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
	l_tot=l1+l2
	l1_ratio=l1/l_tot
	l2_ratio=l2/l_tot
	L1=l1_ratio*h_red
	L2=l2_ratio*h_red
	d=int(0.02*h)
	d1=d*m1**(1/3)
	d2=d*m2**(1/3)
	d_4=d/4
	e_pot_0,e_kin_0=get_energy(phase_traject[0,0],phase_traject[0,1],phase_traject[0,2],phase_traject[0,3],m1,m2,l1,l2,g)
	energy=np.sum(e_pot_0)+np.sum(e_kin_0)
	print('initial energy: '+str(energy))
	prev_points=[]
	prev_phase=[]
	if second_phase_traject is not None:
		distance_scale=0.05
		prev_distances=[distance_scale*distance_in_phase_space(second_phase_traject[0,:]-phase_traject[0,:])]
	# max_theta2_d=1.2*np.max(np.abs(phase_traject[:,3]))
	max_theta2_d=20
	base=np.exp(np.log(0.01)/max_points)
	for i in range(phase_traject.shape[0]):
		if i%10000==0:
			print('rendering iteration: '+str(i)+'/'+str(phase_traject.shape[0]))   
		if i%take_frame_every==0:
			theta1=phase_traject[i,0]
			theta2=phase_traject[i,1]
			theta1_d=phase_traject[i,2]
			theta2_d=phase_traject[i,3]
			prev_phase.append((theta1,theta1_d,theta2,theta2_d))
			# theta2_d=get_corrected_theta2_d(energy,theta1,theta2,theta1_d,theta2_d,m1,m2,l1,l2,g)
			#----transform to cartesian coordinates---
			x1=x0+L1*np.sin(theta1)
			y1=y0+L1*np.cos(theta1)
			x2=x1+L2*np.sin(theta2)
			y2=y1+L2*np.cos(theta2)
			prev_points.append([np.array([x1,y1]),np.array([x2,y2])])
			#---draw the image ----
			img = Image.new("RGB", (w, h), "white")
			draw = ImageDraw.Draw(img)
			n_prev=min(max_points,len(prev_points))
			if draw_phase:
				draw_arrow(draw,np.asarray([int(9*h/8),int(h/2)]),np.asarray([int(15*h/8),int(h/2)]),flank_length=int(h/50))#horizontal
				draw_arrow(draw,np.asarray([w_34,int(7*h/8)]),np.asarray([w_34,int(1*h/8)]),flank_length=int(h/50))#vertical
				font = ImageFont.truetype("arial.ttf", int(h/20))
				draw.text((w_34-h/7,int(1*h/16)), 'angular velocity', font=font, fill=(0,0,0))
				draw.text((int(14*h/8),int(h/2+h/30)), 'angle', font=font, fill=(0,0,0))
			for k in range(n_prev-1):
				idx=n_prev-k
				point=prev_points[-idx]
				xx2=point[1][0]
				yy2=point[1][1]
				point=prev_points[-idx+1]
				xxx2=point[1][0]
				yyy2=point[1][1]
				intensity=int(255*(1-base**idx))
				if draw_marker:
					draw.line([(xx2,yy2),(xxx2,yyy2)],fill=(intensity,intensity,255),width=2)
				if draw_phase:
					if np.abs((prev_phase[-idx][0]+np.pi)%(2*np.pi)-(prev_phase[-idx+1][0]+np.pi)%(2*np.pi))<np.pi:
						phase_x=w_34+x0*((prev_phase[-idx][0]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
						phase_y=y0+y0*prev_phase[-idx][1]/max_theta2_d
						phase_xx=w_34+x0*((prev_phase[-idx+1][0]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
						phase_yy=y0+y0*prev_phase[-idx+1][1]/max_theta2_d
						draw.line([(phase_x,phase_y),(phase_xx,phase_yy)],fill=(255,intensity,intensity),width=2)
						if idx==2:							
							draw.ellipse([(phase_xx-d1,phase_yy-d1),(phase_xx+d1,phase_yy+d1)], fill=(255,0,0), outline=None)
					if np.abs((prev_phase[-idx][2]+np.pi)%(2*np.pi)-(prev_phase[-idx+1][2]+np.pi)%(2*np.pi))<np.pi:
						phase_x=w_34+x0*((prev_phase[-idx][2]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
						phase_y=y0+y0*prev_phase[-idx][3]/max_theta2_d
						phase_xx=w_34+x0*((prev_phase[-idx+1][2]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
						phase_yy=y0+y0*prev_phase[-idx+1][3]/max_theta2_d
						draw.line([(phase_x,phase_y),(phase_xx,phase_yy)],fill=(intensity,intensity,255),width=2)
						if idx==2:
							draw.ellipse([(phase_xx-d2,phase_yy-d2),(phase_xx+d2,phase_yy+d2)], fill=(0,0,255), outline=None)
			if second_phase_traject is not None:
				x11=x0+L1*np.sin(second_phase_traject[i,0])
				y11=y0+L1*np.cos(second_phase_traject[i,0])
				x22=x11+L2*np.sin(second_phase_traject[i,1])
				y22=y11+L2*np.cos(second_phase_traject[i,1])
				draw.line([(x0,y0),(x11,y11)],fill=(255,0,0),width=1)
				draw.ellipse([(x11-d1,y11-d1),(x11+d1,y11+d1)], fill=(255,0,0), outline=None)
				draw.line([(x11,y11),(x22,y22)],fill=(255,0,0),width=1)
				draw.ellipse([(x22-d2,y22-d2),(x22+d2,y22+d2)], fill=(255,0,0), outline=None)

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

			draw.line([(x0,y0),(x1,y1)],fill=(0,0,0),width=2)		
			draw.line([(x1,y1),(x2,y2)],fill=(0,0,0),width=2)
			if second_phase_traject is None:
				draw.ellipse([(x1-d1,y1-d1),(x1+d1,y1+d1)], fill=(255,0,0), outline=None)
				draw.ellipse([(x2-d2,y2-d2),(x2+d2,y2+d2)], fill=(0,0,255), outline=None)
			else:
				draw.ellipse([(x1-d1,y1-d1),(x1+d1,y1+d1)], fill=(0,0,0), outline=None)
				draw.ellipse([(x2-d2,y2-d2),(x2+d2,y2+d2)], fill=(0,0,0), outline=None)
			#----calculate the energies----
			e_pot_i,e_kin_i=get_energy(theta1,theta2,theta1_d,theta2_d,m1,m2,l1,l2,g)
			if show_energy:
				font = ImageFont.truetype("arial.ttf", int(h/20))
				draw.text((int(h/20),int(h/10)), 'energy: '+str(np.sum(e_pot_i)+np.sum(e_kin_i))[:4], font=font, fill=(0,0,0))

			e_pot.append(e_pot_i)
			e_kin.append(e_kin_i)
			frames.append(img)
	#frames[0].save(save_path+'double_pendulum.gif',
	               #save_all=True,
	               #append_images=frames[1:],
	               #duration=40,
	               #loop=0)
	cv2_list=pil_list_to_cv2(frames)
	generate_video(cv2_list,path=save_path+'double_pendulum.avi',fps=frames_per_second)

	e_pot=np.asarray(e_pot)
	e_kin=np.asarray(e_kin)
	print('final energy: '+str(np.sum(e_pot[-1,:])+np.sum(e_kin[-1,:])))
	return e_pot,e_kin

def represent_set_of_trajectories(trajectory_list,img_res=1,m1=1,m2=1,l1=1,l2=0.5,g=10,save_path='trash_figures/',take_frame_every=1,frames_per_second=20):
	frames=[]
	h=int(img_res*200)
	w=2*h
	w_34=int(3*w/4)
	x0=int(h/2)
	y0=int(h/2)
	h_red=int(0.4*h)
	d=int(0.02*h)
	# max_theta2_d=1.2*np.max(np.abs(phase_traject[:,3]))
	max_theta2_d=10
	for i in range(1,trajectory_list[0].shape[0]):
		if i%10000==0:
			print('rendering iteration: '+str(i)+'/'+str(trajectory_list[0].shape[0]))   
		if i%take_frame_every==0:
			#---draw the image ----
			img = Image.new("RGB", (w, h), "white")
			draw = ImageDraw.Draw(img)
			if i<400:
				#--theta1 submanifold
				draw_arrow(draw,np.asarray([int(1*h/8),int(h/2)]),np.asarray([int(7*h/8),int(h/2)]),flank_length=int(h/50))#horizontal
				draw_arrow(draw,np.asarray([int(h/2),int(7*h/8)]),np.asarray([int(h/2),int(1*h/8)]),flank_length=int(h/50))#vertical
				font = ImageFont.truetype("arial.ttf", int(h/24))
				draw.text((h/2-3*h/14,int(1*h/16)), 'inner angular velocity', font=font, fill=(0,0,0))
				draw.text((int(6*h/8),int(h/2+h/30)), 'inner angle', font=font, fill=(0,0,0))
				#--theta2 submanifold
				draw_arrow(draw,np.asarray([int(9*h/8),int(h/2)]),np.asarray([int(15*h/8),int(h/2)]),flank_length=int(h/50))#horizontal
				draw_arrow(draw,np.asarray([w_34,int(7*h/8)]),np.asarray([w_34,int(1*h/8)]),flank_length=int(h/50))#vertical
				font = ImageFont.truetype("arial.ttf", int(h/24))
				draw.text((w_34-3*h/14,int(1*h/16)), 'outer angular velocity', font=font, fill=(0,0,0))
				draw.text((int(14*h/8),int(h/2+h/30)), 'outer angle', font=font, fill=(0,0,0))
			for phase_traject in trajectory_list:
				phase_x1=x0+x0*((phase_traject[i,0]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
				phase_y1=y0+y0*phase_traject[i,2]/max_theta2_d
				if np.abs((phase_traject[i,0]+np.pi)%(2*np.pi)-(phase_traject[i-1,0]+np.pi)%(2*np.pi))<np.pi:
					phase_xx1=x0+x0*((phase_traject[i-1,0]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
					phase_yy1=y0+y0*phase_traject[i-1,2]/max_theta2_d
					draw.line([(phase_x1,phase_y1),(phase_xx1,phase_yy1)],fill=(255,150,150),width=1)
				phase_x2=w_34+x0*((phase_traject[i,1]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
				phase_y2=y0+y0*phase_traject[i,2]/max_theta2_d
				if np.abs((phase_traject[i,1]+np.pi)%(2*np.pi)-(phase_traject[i-1,1]+np.pi)%(2*np.pi))<np.pi:
					phase_xx2=w_34+x0*((phase_traject[i-1,1]+np.pi)%(2*np.pi)-np.pi)/(2*np.pi)
					phase_yy2=y0+y0*phase_traject[i-1,2]/max_theta2_d
					draw.line([(phase_x2,phase_y2),(phase_xx2,phase_yy2)],fill=(150,150,255),width=1)
				draw.point((phase_x1,phase_y1), fill=(255,0,0))
				draw.point((phase_x2,phase_y2), fill=(0,0,255))
			frames.append(img)

	frames[0].save(save_path+'point_cloud.gif',
	               save_all=True,
	               append_images=frames[1:],
	               duration=40,
	               loop=0)
	cv2_list=pil_list_to_cv2(frames)
	generate_video(cv2_list,path=save_path+'point_cloud.avi',fps=frames_per_second)


def draw_fractal(dt=0.005,theta1_lower=-3,theta1_higher=3,theta2_lower=-3,theta2_higher=3,img_res=1,m1=1,m2=1,l1=1,l2=0.5,g=10,save_path='trash_figures/'):
	H=int(200*img_res)
	W=H
	delta1=(theta1_higher-theta1_lower)/W
	delta2=(theta2_higher-theta2_lower)/H
	max_iter=10000
	img = Image.new("RGB", (W, H), "white")
	px=img.load()
	p=0
	for w in range(W):
		for h in range(H):
			print('calculate pixel '+str(p)+'/'+str(W*H))
			p+=1
			theta1=theta1_lower+w*delta1
			theta2=theta2_lower+h*delta2
			theta1_d=0
			theta2_d=0
			flip=False
			n_iter=0
			while not flip and n_iter<max_iter:
				n_iter+=1
				#---explicite midpoint method ----
				theta11,theta22,theta11_d,theta22_d=explicite_euler(dt/2,theta1,theta2,theta1_d,theta2_d,m1,m2,l1,l2,g)
				theta11_dd,theta22_dd=get_theta_dd(theta11,theta22,theta11_d,theta22_d,m1,m2,l1,l2,g)
				theta1_dd,theta2_dd=get_theta_dd(theta11,theta22,theta11_d,theta22_d,m1,m2,l1,l2,g)
				theta1_d_new=theta1_d+dt*theta1_dd
				theta2_d_new=theta2_d+dt*theta2_dd
				theta1+=dt/2*(theta1_d+theta1_d_new)
				theta2+=dt/2*(theta2_d+theta2_d_new)
				theta1_d=theta1_d_new
				theta2_d=theta2_d_new				
				if not -np.pi<theta2<np.pi:
					flip=True
			intensity=int(255*n_iter/max_iter)
			px[w,h]=(intensity,intensity,intensity)
	img.show()

theta1_init=4*np.pi/8
theta2_init=1*np.pi/8
theta1_d_init=0
theta2_d_init=0
dt=0.005
frames_per_second=50
take_frame_every=int(1/(dt*frames_per_second))
n_iter=4000
m2=0.7


#---draw star_dance---
# phase_traject_list=[]
# for i in range(1000):
# 	if i%20==0:
# 		print('trajectory: '+str(i))
# 	r=np.random.rand(4)*2e-2
# 	phase_traject=calculate_trajectory(n_iter,dt,theta1_init+r[0],theta2_init+r[1],theta1_d_init+r[2],theta2_d_init+r[3],m2=m2,l1=1,l2=0.5)
# 	phase_traject_list.append(phase_traject)
# represent_set_of_trajectories(phase_traject_list,img_res=2,m2=m2,l1=1,l2=0.5,save_path='trash_figures/',take_frame_every=take_frame_every,frames_per_second=int(frames_per_second/2))


#---draw fractal
draw_fractal(dt=0.005,theta1_lower=-3,theta1_higher=3,theta2_lower=-3,theta2_higher=3,img_res=0.1,m1=1,m2=1,l1=1,l2=0.5,g=10,save_path='trash_figures/')


# phase_traject=calculate_trajectory(n_iter,dt,theta1_init,theta2_init,theta1_d_init,theta2_d_init,m2=m2,l1=1,l2=0.5)
# phase_traject_2=calculate_trajectory(n_iter,dt,theta1_init+1e-2,theta2_init+1e-2,theta1_d_init,theta2_d_init,m2=m2,l1=1,l2=0.5,add_energy=None)
# e_pot,e_kin=render_phase_traject(phase_traject,img_res=2,take_frame_every=take_frame_every,m2=m2,l1=1,l2=0.5,second_phase_traject=None,draw_phase=False,draw_marker=True,max_points=500,frames_per_second=frames_per_second,show_energy=False)
# plt.plot(np.sum(e_pot,axis=1)+np.sum(e_kin,axis=1))
# plt.ylabel('energy')
# plt.show()

#---audio---
# phase_traject=calculate_trajectory(n_iter,dt,theta1_init,theta2_init,theta1_d_init,theta2_d_init,m2=m2)
# render_phase_traject(phase_traject,img_res=0.5,take_frame_every=int(2/dt/100),m2=m2)
# samples=np.sin(phase_traject[0::30,0])
# plt.plot(samples)
# plt.show()
# fs=20000
# librosa.output.write_wav('chaos_sound.mp3', samples, fs,norm=True)
# time_step=0.05#ms
# ns=time_step*fs
# STFT=classic_STFT(ns=ns,N=256)
# spec=STFT.get_energy_spec(samples)
# plt.imshow(np.log(spec+1),origin='lower')
# plt.show()

