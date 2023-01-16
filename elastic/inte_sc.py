import numpy as np
from scipy import special,interpolate,integrate
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

Folderpath='.'

sigma=np.load(Folderpath+'sigma.npy')
PE=np.load(Folderpath+'PE.npy')
theta=np.linspace(0,180-0.1,num=1800)

theta0=np.pi*theta/180.0

sigma0=sigma[:,0]
sigmasum=integrate.cumtrapz(sigma*np.sin(np.pi*theta/180.0),x=theta0,axis=1,initial=0)
sigmasumf=sigmasum[:,-1]
#sigma_total=sigmasumf*np.pi/1800.0
sigma_total=integrate.simps(sigma*np.sin(theta0),theta0)
# fine structure constant

alpha=137.03599976
Hatree=0.511*np.power(10,6)/(np.power(alpha,2))
rho=2.25
Ze=6.0
Me=12.0
N0=6.02214076*np.power(5.2917721067,3)*0.0001/Me
N1=rho*N0

print(N1)
print(1.0/(2*np.pi*sigma_total[594]*N1))
np.save(Folderpath+'elastic_mfp.npy',np.power(2*np.pi*sigma_total*N1,-1))
sigmasumf=sigmasumf.reshape((len(sigmasumf),1))
sigmasum=sigmasum/sigmasumf

#E=float(input('please input the target energy(eV):'))

def hash_E(E_input):
    if E_input<=100:
        a=E_input-1.0
    else:
        a=99.0+500.0*((np.log10(E_input)-2.0)/2.3)
    return a

np.save(Folderpath+'sigmasum.npy',sigmasum)
prob=np.linspace(0,1,num=10001)
elastic_sc=np.zeros((len(PE),len(prob)),dtype=np.float64)
for i in range(len(PE)):
    inter_p=interpolate.interp1d(sigmasum[i,:],theta)
    elastic_sc[i,:]=inter_p(prob)
#plt.plot(elastic_sc[int(hash_E(1000)),:])
#plt.plot(elastic_sc[int(hash_E(19000)),:])
#plt.show()
np.save(Folderpath+'elastic_sc.npy',elastic_sc)
'''
N=1000000
sc=elastic_sc[int(hash_E(E)),:]
theta2=np.zeros(N,dtype=np.float64)
for j in range(N):
    theta2[j]=sc[np.random.randint(10001)]
#theta2=sc[R]
print(theta2)
a,edge=np.histogram(theta2,bins=np.linspace(0,180,num=181),density=True)
'''
#plt.plot(a)
#plt.plot(theta,np.log10(10*sigma[int(hash_E(1000)),:]/sigmasumf[int(hash_E(1000))]))
#plt.plot(theta,np.log10(10*sigma[int(hash_E(10000)),:]/sigmasumf[int(hash_E(19000))]))
#plt.plot(prob,elastic_sc[int(hash_E(1000)),:])
#plt.plot(prob,elastic_sc[int(hash_E(19000)),:])
#plt.show()
'''

plt.show()
'''
