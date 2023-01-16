import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d,interp2d
from mpl_toolkits.mplot3d import Axes3D
from FPA import optic_data_proc
# fine structure constant
alpha=137.03599976
Hatree=0.511*np.power(10,6)/(np.power(alpha,2))

if __name__ == "__main__":
    Folderpath='.'
    filepath=Folderpath+'/optic_Data/Graphite.txt'
    energy,epsilon=optic_data_proc(filepath)
    inter_ep=interp1d(energy,epsilon,kind='cubic')
    energy=energy[0:-1]
    
    

    qmax=40.0
    qmin=0.01
    delta_q=1.0/100.0
    delta_omegap=1.0/5000.0

    q=np.linspace(np.log10(qmin),np.log10(qmax),num=int(1.0/delta_q))
    q=np.power(10,q)
    omegap=np.linspace(np.log10(energy[0]),np.log10(energy[-4]),num=int(1.0/delta_omegap))
    omegap=np.power(10,omegap)
    el=-(np.power(inter_ep(omegap),-1)).imag

    '''
    energy_loss=-np.power(inter_ep(omega),-1).imag
    plt.plot(omega,energy_loss)
    '''

    #ep_fpa=FPA(delta0*omegamax,omega,delta0*qmax,q,delta1*delta0*omegamax,omegap,inter_ep(omegap))
    ep_fpa=np.load(Folderpath+'/energy_loss/C.npy')
    #ep_fpa=cv2.GaussianBlur(ep_fpa,(15,15),0)
    #ep_fpa=cv2.bilateralFilter(ep_fpa,5,100,100)

    fig=plt.figure()
    ax=Axes3D(fig)
    Omega,Q=np.meshgrid(energy,q)
    '''
    inter_p2=interp2d(omega,q,ep_fpa)
    
    delta0=1.0/1000.0
    delta1=0.005

    omega=np.linspace(delta0*omegamax,1.0*omegamax,num=int(1.0/delta0))
    q=np.linspace(delta0*qmax,1.0*qmax,num=int(1.0/delta0))
    ep_fpa=inter_p2(omega,q)
    Omega,Q=np.meshgrid(omega,q)
    print(ep_fpa.shape)

    ep_fpa1000=np.zeros((1001,1001),dtype=np.float64)
    ep_fpa1000[:,0]=0.0
    ep_fpa1000[0,1:]=-(np.power(inter_ep(omega),-1)).imag
    ep_fpa1000[1:,1:]=ep_fpa

    omega=np.linspace(0,1.0*omegamax,num=int(1.0/delta0)+1)
    q=np.linspace(0,1.0*qmax,num=int(1.0/delta0)+1)
    Omega,Q=np.meshgrid(omega,q)
    np.save(Folderpath+'/energy_loss/Cu_test_1000.npy',ep_fpa1000)
    '''
    ax.plot_surface(Omega[0:70,0:100],Q[0:70,0:100],ep_fpa[0:70,0:100],cmap=plt.get_cmap('viridis'),rcount=200,ccount=200)
    plt.show()