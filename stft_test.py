import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import IPython.display as ipd
import librosa.display

from routines import *


N=256
f_0=60
f_s=8000
c=50#determines the amount of frequency bins, where two consecutive bins are
#shifted version by "c"-cents of one an other
alpha=1#a hyperparameter for the frequency resolution on cost of time resolution
#the larger alpha the larger one should choose the ns-parameter since the time uncertainty
#grows. If one does not change ns, a larger alpha means more computational cost
x, sr = librosa.load('pfiife.mp3',sr=f_s,duration=30)
xl=x[10000:50000]
xs=x[20000:20000+N]

#DFT_obj=DFT_cosine_sine(N)
# #y_hat,z_hat=DFT_obj.forward_dft(xs,positive_frequencies_only=False)
#y_hat,z_hat=DFT_obj.forward_fft(xs)
#e=y_hat**2+z_hat**2
#e=e[0:int(N/2)+1]

time_step=0.05#ms
ns=time_step*f_s
#---classical STFT
c_STFT=classic_STFT(ns=ns,N=256)
c_spec=c_STFT.get_energy_spec(xl)*10**2#conversion factor (experimental), we want the audible amplitude to go from 10-4 to 10^2
# plt.imshow(np.log(spec+1),origin='lower')
# plt.show()
#--morlet STFT
STFT=Morlet_spec(ns=ns,f_0=f_0,f_s=f_s,N_max=1000,cent=c,alpha=alpha)
spec=STFT.get_energy_spec(xl)*5*10**2#conversion factor (experimental)
print('spectrogram size: '+str(spec.shape))


t=np.linspace(0,time_step*(c_spec.shape[1]-1),c_spec.shape[1])
k_m=np.linspace(0,STFT.K-1,STFT.K)
f_m=f_0*2**(k_m*c/1200)
k_fft=np.linspace(0,int(N/2),int(N/2)+1)
f_fft=k_fft*f_s/N
X_fft,Y_fft=np.meshgrid(t,f_fft)
X_m,Y_m=np.meshgrid(t,f_m)
plt.figure(0)
plt.subplot(121)
plt.pcolor(X_fft,Y_fft,np.log(c_spec+1), cmap='gray')
plt.subplot(122)
print(np.max(c_spec))
plt.pcolor(X_m,Y_m,np.log(spec+1), cmap='gray')
print(np.max(spec))
plt.show()


# plt.figure(1)
# t=np.linspace(0,time_step*(c_spec.shape[1]-1),c_spec.shape[1])
# k_fft=np.linspace(0,int(N/2),int(N/2)+1)
# f=k_fft*f_s/N
# X,Y=np.meshgrid(t,f)
# im = plt.pcolor(X,Y,c_spec, cmap='gray')
# plt.show()

# plt.figure(1)
# t=np.linspace(0,time_step*(spec.shape[1]-1),spec.shape[1])
# k_m=np.linspace(0,spec.shape[0]-1,spec.shape[0])
# f=f_0*2**(k_m*c/1200)
# X,Y=np.meshgrid(t,f)
# im = plt.pcolor(X,Y,spec, cmap='gray')
# plt.show()


# k_m=np.linspace(0,STFT.K-1,STFT.K)
# k_fft=np.linspace(0,int(N/2),int(N/2)+1)
# plt.figure(0)
# plt.subplot(121)
# plt.plot(f_0*2**(k_m*c/1200),spec[:])
# plt.subplot(122)
# plt.plot(k_fft*f_s/N,e)
# plt.show()





# K=[0,int(STFT.K*1/3),int(STFT.K*2/3)]
# plt.subplot(221)
# k=0
# plt.plot(STFT.wavelets[k])
# plt.title('N_k='+str(STFT.N_k[k]))
# plt.grid(True)

# plt.subplot(222)
# k=int(STFT.K*1/3)
# plt.plot(STFT.wavelets[k])
# plt.title('N_k='+str(STFT.N_k[k]))
# plt.grid(True)

# plt.subplot(223)
# k=int(STFT.K*2/3)
# plt.plot(STFT.wavelets[k])
# plt.title('N_k='+str(STFT.N_k[k]))
# plt.grid(True)

# plt.subplot(224)
# plt.plot(STFT.wavelets[K[0]])
# plt.plot(STFT.wavelets[K[1]])
# plt.plot(STFT.wavelets[K[2]])
# plt.title('all in one')
# plt.grid(True)

# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.3,
#                     wspace=0.35)
# plt.show()


# y=fft(xs)
# DFT_obj=DFT_cosine_sine(N)
# # #y_hat,z_hat=DFT_obj.forward_dft(xs,positive_frequencies_only=False)
# y_hat,z_hat=DFT_obj.forward_fft(xs)
# e=y_hat**2+z_hat**2
# e=e[0:int(N/2)+1]
# plt.figure(2)
# plt.plot(e)
# plt.show()

# print(np.sum(np.abs(np.real(y)-y_hat*np.sqrt(N))))
# print(np.sum(np.abs(np.imag(y)-z_hat*np.sqrt(N))))
# xs_test,_=DFT_obj.inverse_dft([y_hat,z_hat],orig_signal_is_real=False)
# xs_test_fft,s=DFT_obj.inverse_fft([y_hat,z_hat])
# print(np.sum(np.abs(xs-xs_test)))
# print(np.sum(np.abs(xs-xs_test_fft)))


# librosa.output.write_wav('data/test.mp3', xl, sr)