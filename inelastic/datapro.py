import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from FPA import optic_data_proc

if __name__ == "__main__":
    Folderpath='.'
    filepath=Folderpath+'/optic_Data/Cu.csv'
    energy,epsilon=optic_data_proc(filepath)
    inter_ep=interpolate.interp1d(energy,epsilon,kind='cubic')

    d=1/10000.0
    omegamax=100.0/27.2114

    omega=np.linspace(d*omegamax,omegamax,num=int(1.0/d))
    el1=inter_ep(omega)
    el=-(np.power(el1,-1)).imag

    print(np.sum(omega*el)*omegamax*d/(2*np.pi*np.pi))

    plt.plot(omega,el)
    plt.show()