import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate,integrate
from mpl_toolkits.mplot3d import Axes3D
from FPA import optic_data_proc

alpha=137.03599976
Hatree=0.511*np.power(10,6)/(np.power(alpha,2))
#rho=2.7
#Ze=13.0
#Me=27.0
#N0=6.02214076*np.power(5.2917721067,3)*0.0001/Me
#N1=rho*N0*Ze
#print(N1)
#N1=0.3410820861806174
#kF=np.power(3.0*np.pi*np.pi*N1,1.0/3.0)
#Ef=np.power(kF,2)/2.0
Ef=21.4/Hatree
kF=np.sqrt(2.0*Ef)
print(Ef*Hatree)
#print(np.power(4*np.pi*N1,1/2)*Hatree)

def PE_initial():
    PE=np.zeros(600,dtype=np.float64)
    for i in range(100):
        PE[i]=(i+1.0)
    for i in range(500):
        PE[i+100]=100.0*np.power(10,2.3*(i+1)/500.0)
    #PE=1.0+PE/(0.511*(10**6))
    return PE/Hatree

def mfp_generate(Elf,omega,delta_omega,q,delta_q,Folderpath):
    Omega,Q=np.meshgrid(omega,q)
    d_mfp=Elf/(np.pi*Q)
    
    E=PE_initial() 
    Ei=((np.log10(E/omega[0])/(np.log10(omega[-1])-np.log10(omega[0])))/delta_omega).astype(np.int64)
    #Ei=Ei+1

    inelastic_mfp=np.zeros(600,dtype=np.float64)
    inelastic_sp=np.zeros(600,dtype=np.float64)

    elf_sumq=np.zeros((600,int(1.0/delta_omega)),dtype=np.float64)
    for j in range(600):
        print(j)
        dmfp=d_mfp/(E[j]+Ef)
        low_bq=np.sqrt(2.0)*(np.sqrt(E[j]+Ef)-np.sqrt((E[j]+Ef-Omega)*(E[j]+Ef-Omega>=0)))
        high_bq=np.sqrt(2.0)*(np.sqrt(E[j]+Ef)+np.sqrt((E[j]+Ef-Omega)*(E[j]+Ef-Omega>=0)))

        cond1=(Q>=low_bq).astype(np.float64)
        cond2=(Q<=high_bq).astype(np.float64)
        condf=cond1*cond2

        Elf_sum=integrate.trapz(dmfp*condf,x=q,axis=0)

        E_index=Ei[j]

        elf_sumq[j,0:E_index]=Elf_sum[0:E_index]

        inelastic_mfp[j]=integrate.trapz(Elf_sum[0:E_index],x=omega[0:E_index])#*x+integrate.simps(Elf_sum[0:E_index_next],x=omega[0:E_index_next])*(1.0-x)
        inelastic_sp[j]=integrate.trapz((omega*Elf_sum)[0:E_index],x=omega[0:E_index])#*x+integrate.simps((omega*Elf_sum)[0:E_index_next],x=omega[0:E_index_next])*(1.0-x)
    np.save(Folderpath+'/inelastic/PE.npy',E)
    np.save(Folderpath+'/inelastic/inelastic_mfp.npy',np.power(inelastic_mfp,-1))
    np.save(Folderpath+'/inelastic/elf_sumq.npy',elf_sumq)
    #plt.plot(np.log10(E*Hatree),Hatree*inelastic_sp/0.53)
    
    plt.plot(np.log10(E*Hatree),np.log10((0.53*np.power(inelastic_mfp,-1))))
    print((0.053*np.power(inelastic_mfp,-1))[0:100])
    plt.show()
    pass

def el_generate(omega,delta_omega,q,delta_q,Folderpath):
    E=PE_initial() 
    Ei=((np.log10(E/omega[0])/(np.log10(omega[-1])-np.log10(omega[0])))/delta_omega).astype(np.int64)
    #Ei=Ei+1

    elf_sumq=np.load(Folderpath+'/inelastic/elf_sumq.npy')
    inelastic_mfp=np.load(Folderpath+'/inelastic/inelastic_mfp.npy')
    inelastic_el=np.zeros((600,10001),dtype=np.float64)
    plt.plot(omega[0:Ei[590]],elf_sumq[590,0:Ei[590]]*inelastic_mfp[590])
    elf_sumq=integrate.cumtrapz(elf_sumq,x=omega,initial=0)
    prob=np.linspace(0,1,num=10001)
    for i in range(len(E)):
        p=elf_sumq[i,0:Ei[i]]*inelastic_mfp[i]
        max_error=np.max(p)
        inter_p=interpolate.interp1d(p/max_error,omega[0:Ei[i]])
        inelastic_el[i,:]=inter_p(prob)
    np.save(Folderpath+'/inelastic/inelastic_le.npy',inelastic_el)
    #plt.plot(inelastic_el[400,:])
    plt.show()
    pass

def sc_generate(Elf,omega,delta_omega,q,delta_q,Folderpath):
    Omega,Q=np.meshgrid(omega,q)
    E=PE_initial() 
    #Ei=((np.log10(E/omega[0])/(np.log10(omega[-1])-np.log10(omega[0])))/delta_omega).astype(np.int64)
    #E=[100.0,1000.0,10000.0]/Hatree

    d_sc=Elf/np.power(np.pi*Q,2)
    prob=np.linspace(0,1,num=1001)
    inelastic_sc=np.zeros((len(E),len(omega),len(prob)),dtype=np.float64)
    for j in range(len(E)):
        dsc=d_sc/(E[j]+Ef)
        print(j)
        #cond_omega=(E[j]+Ef-Omega>=0).astype(np.float64)
        
        low_bq=np.sqrt(2.0)*(np.sqrt(E[j]+Ef)-np.sqrt((E[j]+Ef-Omega)*(E[j]+Ef-Omega>=0)))
        high_bq=np.sqrt(2.0)*(np.sqrt(E[j]+Ef)+np.sqrt((E[j]+Ef-Omega)*(E[j]+Ef-Omega>=0)))
        cond1=(Q>=low_bq).astype(np.float64)
        cond2=(Q<high_bq).astype(np.float64)
        condq=cond1*cond2
        
        Theta=np.arccos((-np.power(Q,2)*condq/4.0+(E[j]+Ef)-0.5*Omega*condq)/(np.sqrt((E[j]+Ef)*(E[j]+Ef-Omega)*condq)+(1e12)*(1.0-condq)))
        Theta=Theta*condq
        dsc=dsc*np.sqrt((E[j]+Ef-Omega)*condq)*np.sin(Theta)
        dsc_total=integrate.cumtrapz(dsc,x=Theta,axis=0,initial=0)
        dsc=condq*(dsc_total/np.maximum(np.max(dsc_total,axis=0),1e-12))
        for k in range(len(omega)):
            p=dsc[:,k]
            if np.max(p)==0.0:
                inelastic_sc[j,k,:]=0.0
            else:
                p=p/np.max(p)
                inter_p=interpolate.interp1d(p,Theta[:,k])
                inelastic_sc[j,k,:]=inter_p(prob)
        #fig=plt.figure()
        #ax=Axes3D(fig)
        #ax.scatter(np.log10(Omega),Theta,condq)
        #plt.show()

    np.save(Folderpath+'/inelastic/inelastic_sc.npy',inelastic_sc)
    pass


def main():
    Folderpath='.'
    filepath=Folderpath+'/optic_Data/Graphite.txt'
    energy,epsilon=optic_data_proc(filepath)
    energy=energy[0:-1]
    Elf=np.abs(np.load(Folderpath+'/energy_loss/Graphite.npy'))
    omega=energy
    
    qmax=40.0
    qmin=0.01
    delta_q=1.0/200.0
    a0=np.linspace(0,1,num=int(1/delta_q))
    q=qmin*(np.power(qmax/qmin,np.power(a0,2)))
    #q=np.linspace(np.log10(qmin),np.log10(qmax),num=int(1.0/delta_q))
    #q=np.power(10,q)
    print(Elf.shape)
    
    inter_elf=interpolate.interp2d(omega,q,Elf,kind='cubic')

    delta_q=1.0/1000.0
    q=np.linspace(np.log10(qmin),np.log10(qmax),num=int(1.0/delta_q))
    q=np.power(10,q)
    delta_omega=1.0/1000.0    
    omega=np.linspace(np.log10(0.0001/Hatree),np.log10(omega[-1]),num=int(1.0/delta_omega))
    omega=np.power(10,omega)

    Elf=inter_elf(omega,q)

    ge_mode=int(input('input generate mode:'))

    if ge_mode==0:
        mfp_generate(Elf,omega,delta_omega,q,delta_q,Folderpath)
    elif ge_mode==1:
        el_generate(omega,delta_omega,q,delta_q,Folderpath)
    elif ge_mode==2:
        sc_generate(Elf,omega,delta_omega,q,delta_q,Folderpath)
    else:
        print('enter 0,1 or 2')
    




if __name__=='__main__':
    main()