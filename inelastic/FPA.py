import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d,interp2d
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

# fine structure constant
alpha=137.03599976
Hatree=0.511*np.power(10,6)/(np.power(alpha,2))


# read optic data from files
def optic_data_proc(filepath):
    refrac=np.loadtxt(filepath,dtype=float,delimiter='\t')
    
    energy=refrac[:,0]/Hatree

    epsilon=refrac[:,1]+1j*refrac[:,2]
    #epsilon=np.power(refrac[:,1]+1j*refrac[:,2],2)

    return energy,epsilon

# definition of Lindhard functions
def F_func(x):
    if x==1:
        return 0.0
    else:
        y=(1.0-np.power(x,2))*np.log(np.abs((1.0+x)/(1.0-x)))
        return y

def F_func_dev(x):
    if x==1:
        return -float('inf')
    else:
        y=2.0-2.0*x*np.log((x+1.0)/(1.0-x))
        return y

def Lindhard_func_real(q,omega,omegap):
    kF=np.power(3*np.pi*np.power(omegap,2)/4.0,1/3)
    EF=kF*kF/2.0
    X=omega/EF
    Z=q/(2.0*kF)
    temp=1.0/2.0+(F_func(Z-X/(4.0*Z))+F_func(Z+X/(4.0*Z)))/(8.0*Z)
    return 1.0+2.0*temp/(np.pi*q*Z)    

def Lindhard_func_real_dev(q,omega,omegap):
    kF=np.power(3*np.pi*np.power(omegap,2)/4.0,1/3)
    EF=kF*kF/2.0
    X=omega/EF
    Z=q/(2.0*kF)
    C1=Z-X/(4.0*Z)
    C2=Z+X/(4.0*Z)
    
    temp=0.0
    temp=temp+C1*np.log(np.abs((C1+1.0)/(C1-1.0)))-C2*np.log(np.abs((C2+1.0)/(C2-1.0)))
    temp=temp*X/(4.0*np.power(Z,2))
    temp=temp+2.0-C1*np.log(np.abs((C1+1.0)/(C1-1.0)))-C2*np.log(np.abs((C2+1.0)/(C2-1.0)))
    temp=temp/(2.0*np.pi*q*Z)
    temp=temp+2.0+1.0/(np.pi*q*Z)
    temp=-2.0*temp/(3.0*omegap)
    return temp

def Lindhard_func_imag(q,omega,omegap):
    kF=np.power(3*np.pi*np.power(omegap,2)/4.0,1/3)
    EF=kF*kF/2.0
    X=omega/EF
    Z=q/(2.0*kF)
    if (X>=0) and (X<=4*Z*(1.0-Z)):
        return X/(8.0*kF*np.power(Z,3))
    elif (X>=np.abs(4*Z*(1-Z))) and (X<=4*Z*(1+Z)):
        return (1.0-np.power(Z-X/(4*Z),2))/(8.0*kF*np.power(Z,3))
    else:
        return 0.0

def gomega(omega,inter_omega):
    return 2*inter_omega/(np.pi*omega)

# full Penn
def FPA(q,omega,omegap,energy_loss):
    kF0=np.power(3*np.pi*np.power(omegap,2)/4.0,1/3)

    Omega,Q,Kf0=np.meshgrid(omega,q,kF0)


    ep_L_imag=Lindhard_imag(Q,Omega,Kf0)
    print('step2')
    ep_L_real=Lindhard_real(Q,Omega,Kf0)
    print('step3')
    a1=ep_L_imag/(np.power(ep_L_real,2)+np.power(ep_L_imag,2))
    Omega,Q,a2=np.meshgrid(omega,q,energy_loss)
    Omegap=np.power(4.0*np.power(Kf0,3)/(3.0*np.pi),1/2)
    a2=a2*2/(np.pi*Omegap)
    ep_fpa=integrate.simps(a1*a2,x=omegap,axis=-1)

    '''
    for k in range(len(omegap)):
        ep_L_imag=Lindhard_imag(Q,Omega,omegap[k])
        ep_L_real=Lindhard_real(Q,Omega,omegap[k])
        delta=ep_L_imag/(np.power(ep_L_real,2)+np.power(ep_L_imag,2))
        #print('omegap:',omegap[k],' L_real_min:',np.min(np.abs(ep_L_real)),' g:',gomega(omegap[k],inter_omega[k]))
        ep_fpa=ep_fpa+delta_omegap*gomega(omegap[k],inter_omega[k])*delta
    '''

    Omega,Q=np.meshgrid(omega,q)
    delta0=1.0
    Delta0=1e-8
    kF_bond=Omega/Q-Q/2.0
    Plasmon_cond1=(kF_bond>0).astype(float)
    Plasmon_cond2=Plasmon_cond(Q,Omega,Delta0)
    Plasmon_condt=Plasmon_cond1*Plasmon_cond2
    kF=(kF_bond-Delta0)*Plasmon_condt+1.0*(1.0-Plasmon_condt)+0.0*1j
    print(np.min(kF))
    for k in range(1000):
        temp=kF
        #Omegap1=np.power(4.0*np.power(kF,3)/(3.0*np.pi),1/2)
        y0=Lindhard_real(Q,Omega,kF)*Plasmon_condt
        #print('y0')
        y1=Lindhard_real_dev(Q,Omega,kF)
        #print('y1')
        kF=kF-y0/y1
        #print(kF)
        delta0=np.max(np.abs(np.abs(kF)-np.abs(temp)))
        print(delta0,np.max(np.abs(kF.imag)))
    #kF=kF-1000.0*(1.0-Plasmon_condt)
    kF2=np.abs(kF)
    Omegap1=np.power(4.0*np.power(kF2,3)/(3.0*np.pi),1/2)
    Omegap2=Omegap1*(Omegap1<=np.max(omegap))+np.max(omegap)*(Omegap1>np.max(omegap))
    inter_omega=inter_ep(Omegap2)
    delta=(gomega(Omegap1,inter_omega)*Plasmon_condt)*np.power(3*np.pi*kF2,1/2)/np.abs(Lindhard_real_dev(Q,Omega,kF))
    ep_fpa=delta+ep_fpa

    return ep_fpa

def Fx(x):
    y=(1.0-np.power(x,2))*np.log(np.abs((1.0+x)/(1.0-x)))
    return y

def Lindhard_real(Q,Omega,kF):
    #kF=np.power(3*np.pi*np.power(Omegap,2)/4.0,1/3)
    Ef=np.power(kF,2)/2.0
    X=Omega/Ef
    Z=Q/(2.0*kF)
    temp=1.0/2.0+(Fx(Z-X/(4.0*Z))+Fx(Z+X/(4.0*Z)))/(8.0*Z)
    return 1.0+temp/(kF*np.pi*np.power(Z,2))

def Plasmon_cond(Q,Omega,Delta0):
    kF=Omega/Q-Q/2.0
    X=1.0+(4*kF/(np.pi*np.power(Q,2)))*(0.5+kF*Fx((Omega/Q+Q/2.0)/kF)/(4.0*Q))
    return (X<0).astype(float)

def Lindhard_real_dev(Q,Omega,kF):
    #kF=np.power(3*np.pi*np.power(omegap,2)/4.0,1/3)
    #temp1=4*kF*Q/(-2*kF*Q+np.power(Q,2)+2*Omega)
    #temp2=4*kF*Q/(2*kF*Q+np.power(Q,2)-2*Omega)
    #temp=np.log((1.0+temp1)/(1.0-temp2))
    #temp=2*kF*temp/(np.pi*np.power(Q,3))

    Ef=np.power(kF,2)/2.0
    X=Omega/Ef
    Z=Q/(2.0*kF)
    Y_minus=Z-X/(4.0*Z)
    Y_plus=Z+X/(4.0*Z)
    a=np.log(np.abs((1.0+Y_minus)/(1.0-Y_minus)))
    b=np.log(np.abs((1.0+Y_plus)/(1.0-Y_plus)))
    temp=(a+b)/(2*np.pi*kF*Q*np.power(Z,2))
    return temp

def Lindhard_imag(Q,Omega,kF):
    #kF=np.power(3*np.pi*np.power(Omegap,2)/4.0,1/3)
    Ef=np.power(kF,2)/2.0
    X=Omega/Ef
    Z=Q/(2.0*kF)
    #ep_L=np.zeros((len(q),len(omega)),dtype=np.float64)
    Part1=((X>0)*(X<4*Z*(1.0-Z))).astype(float)*(X/(8.0*kF*np.power(Z,3)))
    Part2=((X>np.abs(4*Z*(1.0-Z)))*(X<4*Z*(1.0+Z))).astype(float)*((1.0-np.power(Z-X/(4*Z),2))/(8.0*kF*np.power(Z,3)))
    return Part1+Part2

if __name__ == "__main__":
    Folderpath='.'
    filepath=Folderpath+'/optic_Data/CNT.txt'
    energy,epsilon=optic_data_proc(filepath)
    el1=-(np.power(epsilon,-1)).imag
    inter_ep=interp1d(energy,el1,kind='cubic')
    energy=energy[0:-1]

    qmax=40.0
    qmin=0.01
    delta_q=1.0/200.00

    delta_omegap=1.0/7000.0

    a0=np.linspace(0,1,num=int(1/delta_q))
    q=qmin*(np.power(qmax/qmin,np.power(a0,2)))
    #q=np.linspace(np.log10(qmin),np.log10(qmax),num=int(1.0/delta_q))
    #q=np.power(10,q)
    omegap=np.linspace(np.log10(1.0/Hatree),np.log10(400/Hatree),num=int(1.0/delta_omegap))
    omegap=np.power(10,omegap)
    el=inter_ep(omegap)


    '''
    energy_loss=-np.power(inter_ep(omega),-1).imag
    energy_loss=np.abs(energy_loss)
    plt.plot(omega,energy_loss)
    '''
    ep_fpa=FPA(q,energy,omegap,el)

    
    fig=plt.figure()
    ax=Axes3D(fig)
    Omega,Q=np.meshgrid(energy[0:126],q[0:900])
    ax.plot_surface(Omega,Q,np.abs(ep_fpa[0:900,0:126]),cmap=plt.get_cmap('viridis'),rcount=200,ccount=200)
    plt.show()
    np.save(Folderpath+'/energy_loss/CNT.npy',ep_fpa)
    