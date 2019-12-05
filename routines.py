#here I collect signalprocessing routines
import numpy as np

def fft(x):
	return np.fft.fft(x)

class DFT_cosine_sine():

    def __init__(self,N):
        self.N=N
        self.log_N=int(np.log2(N))
        #if x is real we can use symmetry property and only calculate
        #N_eff signal elements
        self.N_eff=int(np.floor(self.N/2)+1)

        #calculate the matrix elements of the DFT
        self.C=np.zeros([N,N])
        self.S=np.zeros([N,N])

        self.sqrt2=np.sqrt(2)
        self.sqrtN=np.sqrt(N)

        c=np.zeros(N)
        s=np.zeros(N)

        for l in range(N):
            c[l]=np.cos(2*np.pi*l/N)/self.sqrtN
            s[l]=np.sin(2*np.pi*l/N)/self.sqrtN
        for k in range(N):
            for n in range(N):
                idx=np.mod(k*n,N)
                self.C[k,n]=c[idx]
                self.S[k,n]=s[idx]
        #note that the matrices C and S are symmetric

        #For the FFT we also need all logarithmic submatrices



    def forward_real(self,x,positive_frequencies_only=False):
        #this is a subroutine that does not check the the input. Plase do not use from outside
        if np.iscomplexobj(x):
            raise Exception('the signal must be real valued')        
        if positive_frequencies_only:
            y_hat=np.zeros(self.N_eff)#the real part 
            z_hat=np.zeros(self.N_eff)#the imaginary part
        else:
            y_hat=np.zeros(self.N)#the real part 
            z_hat=np.zeros(self.N)#the imaginary part

        for k in range(self.N_eff):
            for n in range(self.N):
                y_hat[k]+=self.C[k,n]*x[n]
                z_hat[k]-=self.S[k,n]*x[n]
        if not positive_frequencies_only:
            #fill the symmetric elements
            for k in range(self.N_eff,self.N):
                y_hat[k]=y_hat[self.N-k]
                z_hat[k]=-z_hat[self.N-k]
        return y_hat,z_hat        

    def forward_complex(self,y,z):
        #this is a subroutine that does not check the the input. Please do not use from outside
        y_hat=np.zeros(self.N)#the real part 
        z_hat=np.zeros(self.N)#the imaginary part
        for k in range(self.N):
            for n in range(self.N):
                y_hat[k]+=self.C[k,n]*y[n]+self.S[k,n]*z[n]
                z_hat[k]-=self.S[k,n]*x[n]+self.C[k,n]*z[n]
        return y_hat,z_hat 

    def inverse_(self,y_hat,z_hat,orig_signal_is_real=False):
        y=np.zeros(self.N)
        z=None
        if y_hat.shape[0]==self.N_eff:
            #we assume that the original signal is real valued in this case
            y_hat_full=np.zeros(self.N)
            z_hat_full=np.zeros(self.N)
            y_hat_full[:self.N_eff]=y_hat
            z_hat_full[:self.N_eff]=z_hat
            for k in range(self.N_eff,self.N):
                #fill the symmetric elements
                y_hat_full[k]=y_hat[self.N-k]
                z_hat_full[k]=-z_hat[self.N-k]
            for n in range(self.N):
                for k in range(self.N):
                    y[n]+=self.C[n,k]*y_hat_full[k]-self.S[n,k]*z_hat_full[k]
            return y,z
        if orig_signal_is_real:
            for n in range(self.N):
                for k in range(self.N):
                    y[n]+=self.C[n,k]*y_hat[k]-self.S[n,k]*z_hat[k]
            return  y,z
        else:
            z=np.zeros(self.N)
            for n in range(self.N):
                for k in range(self.N):
                    y[n]+=self.C[n,k]*y_hat[k]-self.S[n,k]*z_hat[k]
                    z[n]+=self.S[n,k]*y_hat[k]+self.C[n,k]*y_hat[k]
            return  y,z


    def check_signal(self,u):
        shape=u.shape
        if len(shape)!=1:
            raise Exception('the signal must be one-dimensional')
        elif u.shape[0]!=self.N and u.shape[0]!=self.N_eff:
            raise Exception('the signal must have length '+str(self.N)+'or in case of inverse trafo also '+str(self.N_eff))


    def forward_dft(self,x,positive_frequencies_only=False):
        #x is either a real valued signal of length N
        #or a list of two real valued signals of length N
        #corresponding to the real and imaginary part of a complex signal
        if isinstance(x, list):
            L=len(x)
            if L==1:
                check_signal(x[0])
                if np.iscomplexobj(x[0]):
                    y_hat,z_hat=self.forward_complex(np.real(x[0]),np.Im(x[0]))                    
                else:
                    y_hat,z_hat=self.forward_real(x[0],positive_frequencies_only=positive_frequencies_only)
            elif L==2:
                self.check_signal(x[0])
                self.check_signal(x[1])
                y_hat,z_hat=self.forward_complex(x[0],x[1])              
            else:
                raise Exception('the list must have length 1 or 2')
        else:
            self.check_signal(x)
            y_hat,z_hat=self.forward_real(x,positive_frequencies_only=positive_frequencies_only)
        return y_hat,z_hat

    def inverse_dft(self,x_hat,orig_signal_is_real=False):
        #analog to forward_dft()
        if isinstance(x_hat, list):
            L=len(x_hat)
            if L==1:
                self.check_signal(x_hat[0])
                y,z=self.inverse_(np.real(x_hat[0]),np.imag(x_hat[1]),orig_signal_is_real=orig_signal_is_real)
            if L==2:
                self.check_signal(x_hat[0])
                self.check_signal(x_hat[1])                 
                y,z=self.inverse_(x_hat[0],x_hat[1],orig_signal_is_real=orig_signal_is_real)            
            else:
                raise Exception('the list must have length 1 (for a complex signal) or 2')
        else:
            self.check_signal(x_hat)
            y,z=self.inverse_(np.real(x_hat),np.imag(x_hat),orig_signal_is_real=orig_signal_is_real)
        return y,z

    def fft_forward_complex(self,y,z,n,log_n):
        y_hat=np.zeros(n)
        z_hat=np.zeros(n)
        n_2=int(n/2)
        
        #core function
        if n==2:
            y_hat[0]=(y[0]+y[1])
            y_hat[1]=(y[0]-y[1])
            z_hat[0]=(z[0]+z[1])
            z_hat[1]=(z[0]-z[1])
            return y_hat/self.sqrt2,z_hat/self.sqrt2

        #recursive phase adding
        else:
            g_y_hat,g_z_hat=self.fft_forward_complex(y[0::2],z[0::2],n_2,log_n-1)
            h_y_hat,h_z_hat=self.fft_forward_complex(y[1::2],z[1::2],n_2,log_n-1)
            for k in range(n_2):
                N_n=int(self.N/n)
                k_p_n2=int(k+n_2)
                y_hat[k]=g_y_hat[k]+self.sqrtN*self.C[k,N_n]*h_y_hat[k]+self.sqrtN*self.S[k,N_n]*h_z_hat[k]
                y_hat[k_p_n2]=g_y_hat[k]+self.sqrtN*self.C[k_p_n2,N_n]*h_y_hat[k]+self.sqrtN*self.S[k_p_n2,N_n]*h_z_hat[k]
                z_hat[k]=g_z_hat[k]-self.sqrtN*self.S[k,N_n]*h_y_hat[k]+self.sqrtN*self.C[k,N_n]*h_z_hat[k]
                z_hat[k_p_n2]=g_z_hat[k]-self.sqrtN*self.S[k_p_n2,N_n]*h_y_hat[k]+self.sqrtN*self.C[k_p_n2,N_n]*h_z_hat[k]
            return y_hat/self.sqrt2,z_hat/self.sqrt2

    def fft_inverse_complex(self,y_hat,z_hat,n,log_n):
        y=np.zeros(n)
        z=np.zeros(n)
        n_2=int(n/2)
        
        #core function
        if n==2:
            y[0]=(y_hat[0]+y_hat[1])
            y[1]=(y_hat[0]-y_hat[1])
            z[0]=(z_hat[0]+z_hat[1])
            z[1]=(z_hat[0]-z_hat[1])
            return y/self.sqrt2,z/self.sqrt2

        #recursive phase adding
        else:
            g_y,g_z=self.fft_inverse_complex(y_hat[0::2],z_hat[0::2],n_2,log_n-1)
            h_y,h_z=self.fft_inverse_complex(y_hat[1::2],z_hat[1::2],n_2,log_n-1)
            for k in range(n_2):
                N_n=int(self.N/n)
                k_p_n2=int(k+n_2)
                y[k]=g_y[k]+self.sqrtN*self.C[k,N_n]*h_y[k]-self.sqrtN*self.S[k,N_n]*h_z[k]
                y[k_p_n2]=g_y[k]+self.sqrtN*self.C[k_p_n2,N_n]*h_y[k]-self.sqrtN*self.S[k_p_n2,N_n]*h_z[k]
                z[k]=g_z[k]+self.sqrtN*self.S[k,N_n]*h_y[k]+self.sqrtN*self.C[k,N_n]*h_z[k]
                z[k_p_n2]=g_z[k]+self.sqrtN*self.S[k_p_n2,N_n]*h_y[k]+self.sqrtN*self.C[k_p_n2,N_n]*h_z[k]
            return y/self.sqrt2,z/self.sqrt2


    def check_if_pot(self,x):
        L=np.log2(x.shape)
        return L%1==0


    def forward_fft(self,x):
        #x is either a real valued signal of length N
        #or a list of two real valued signals of length N
        #corresponding to the real and imaginary part of a complex signal
        if isinstance(x, list):
            L=len(x)
            if not self.check_if_pot(x[0]):
                ValueError('signal length must be a power of two')
            if L==1:
                check_signal(x[0])
                log_N=int(np.log2(x[0].shape))
                if np.iscomplexobj(x[0]):
                    y_hat,z_hat=self.fft_forward_complex(np.real(x[0]),np.Im(x[0]),np.real(x[0]).shape[0],self.log_N)                    
                else:
                    #y_hat,z_hat=self.fft_forward_real(x[0],positive_frequencies_only=positive_frequencies_only)
                    y_hat,z_hat=self.fft_forward_complex(x[0],np.zeros(x[0].shape),x[0].shape[0],self.log_N) 
            elif L==2:
                self.check_signal(x[0])
                self.check_signal(x[1])
                y_hat,z_hat=self.fft_forward_complex(x[0],x[1],x[0].shape,log_N)              
            else:
                raise Exception('the list must have length 1 or 2')
        else:
            if not self.check_if_pot(x):
                ValueError('signal length must be a power of two')
            self.check_signal(x)
            #y_hat,z_hat=self.fft_forward_real(x,positive_frequencies_only=positive_frequencies_only)
            y_hat,z_hat=self.fft_forward_complex(np.real(x),np.zeros(x.shape),np.real(x).shape[0],self.log_N)
        return y_hat,z_hat


    def inverse_fft(self,x):
        #
        #x is a list of two real valued signals of length N
        #corresponding to the real and imaginary part of a complex signal
        if isinstance(x, list):
            L=len(x)
            if not self.check_if_pot(x[0]):
                ValueError('signal length must be a power of two')
            # if L==1:
            #     check_signal(x[0])
            #     log_N=int(np.log2(x[0].shape))
            #     if np.iscomplexobj(x[0]):
            #         y_hat,z_hat=self.fft_forward_complex(np.real(x[0]),np.Im(x[0]),np.real(x[0]).shape[0],log_N)                    
            #     else:
            #         #y_hat,z_hat=self.fft_forward_real(x[0],positive_frequencies_only=positive_frequencies_only)
            #         y_hat,z_hat=self.fft_forward_complex(x[0],np.zeros(x[0].shape),x[0].shape[0],log_N) 
            elif L==2:
                self.check_signal(x[0])
                self.check_signal(x[1])
                y,z=self.fft_inverse_complex(x[0],x[1],self.N,self.log_N)              
            else:
                raise Exception('the input list must have length 2')
        else:
            raise Exception('The input must be a list of two real valued signals corresponding to real and complex part respectively')
            # if not self.check_if_pot(x):
            #     ValueError('signal length must be a power of two')
            # self.check_signal(x)
            # log_N=int(np.log2(x.shape))
            # #y_hat,z_hat=self.fft_forward_real(x,positive_frequencies_only=positive_frequencies_only)
            # y_hat,z_hat=self.fft_forward_complex(np.real(x),np.zeros(x.shape),np.real(x).shape[0],log_N)
        return y,z

class classic_STFT():
    def __init__(self,ns,N):
        self.ns=int(ns)
        self.N=N
        self.w=self.get_gaussian_window(N)
        self.N_eff=int(N/2+1)
        self.DFT_obj=DFT_cosine_sine(N)

    def get_energy_spec(self,x,verbose=True):
        L_x=x.shape[0]
        L_s=int(L_x/self.ns)+1
        buffer_x=np.zeros(int((self.N-1)/2+1))
        x=np.concatenate((buffer_x,x,buffer_x))
        energy_spec=np.zeros((self.N_eff,L_s))
        for l in range(L_s):
            if verbose:
                print('progress:'+str(l+1)+'/'+str(L_s))
            position=l*self.ns
            y_hat,z_hat=self.DFT_obj.forward_fft(x[position:position+self.N])
            energy_spec[:,l]=(y_hat**2+z_hat**2)[0:self.N_eff]
        return energy_spec


    def get_gaussian_window(self,N):
        n=np.linspace(0,N-1,N)-(N-1)/2
        return np.exp(-(8/N**2)*(n*n))

class Morlet_spec():
    def __init__(self,ns,f_0,f_s,cent=100,N_max=2000,alpha=1):
            #actually we are only using the ration f_0/f_s
            self.ns=int(ns)#the step size of the STFT
            self.c=cent#the frequency resolution in cent
            self.f_0=f_0#the smalles frequency we are considering
            self.f_s=f_s#the sample frequency
            self.N_max=N_max#we can fix the largest window size to save computational cost 
                        #while paying with frequency resolution for the lowest frequencies
            self.alpha=alpha
            rel_freq_ratio=cent/1200
            self.K=int(np.log2(f_s/(2*f_0))/rel_freq_ratio)#the number of frequency bins
            k=np.linspace(0,self.K-1,self.K)
            N_k=(2*alpha*f_s/f_0)/(np.pi*2**(k*rel_freq_ratio)*(2**rel_freq_ratio-1))
            self.N_k=np.minimum(N_k,N_max).astype(int)
            print(self.N_k)
            self.w=[]#here we store the frequency dependent window functions
            self.wavelets=[]#here we store the wavelets
            for k in range(self.K):
                w_k=self.get_gaussian_window(self.N_k[k])
                self.w.append(w_k)
                n_k=np.linspace(0,self.N_k[k]-1,self.N_k[k])
                f_k=2**(k*rel_freq_ratio)*f_0/f_s
                oscillator_k=np.exp(-2j*np.pi*f_k*n_k)
                self.wavelets.append(oscillator_k*w_k/np.sqrt(self.N_k[k]))

            
    def get_complex_spec(self,x):
        L_x=x.shape[0]
        L_s=int(L_x/self.ns)+1
        buffer_x=np.zeros(int((self.N_k[0]-1)/2+1))
        x=np.concatenate((buffer_x,x,buffer_x))
        complex_spec=np.zeros((self.K,L_s),dtype=np.complex_)
        for l in range(L_s):
            print('progress:'+str(l+1)+'/'+str(L_s))
            position=l*self.ns
            complex_spec[:,l]=self.STFT(x[position:position+self.N_k[0]])
        return complex_spec

    def get_energy_spec(self,x):
        c_spec=self.get_complex_spec(x)
        return (c_spec*np.conjugate(c_spec)).real

    def get_gaussian_window(self,N):
        n=np.linspace(0,N-1,N)-(N-1)/2
        return np.exp(-(8/N**2)*(n*n))

    def STFT(self,x):
        x_hat=np.zeros(self.K,dtype=np.complex_)
        for k in range(self.K):
            for n in range(self.N_k[k]):
                shifted_idx=int(n+(self.N_k[0]-self.N_k[k])/2)
                x_hat[k]+=x[shifted_idx]*self.wavelets[k][n]
        return x_hat






















