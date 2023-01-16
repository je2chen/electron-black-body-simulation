import numpy as np
from scipy import special,integrate
from tfd import TFD
#from matplotlib import pyplot as plt

# fine structure constant
alpha=137.03599976

def PE_initial():
    PE=np.zeros(600,dtype=np.float64)
    for i in range(100):
        PE[i]=(i+1.0)
    for i in range(500):
        PE[i+100]=100.0*np.power(10,2.3*(i+1)/500.0)
    PE=1.0+PE/(0.511*(10**6))
    return PE

def main():
    # basic input
                            # atomic number
    lmax=250                     # maximun number of partial waves <= 200
    rmax=1.0/np.exp(5)          # expansion length in center, with unit hbar/mc
    
    # tolerance for numerical integration
    rtol=1e-8
    atol=1e-8
    
    Ze=14
    # TFD parameters
    TFD_potential=TFD(Ze)
    #B=TFD_potential.B
    #b=TFD_potential.b
    Z=TFD_potential.Z
    rinf=TFD_potential.r0
    
    # initialize energy of primary electrons
    PE=PE_initial()
    #PE=np.array([100.0])
    #E=1.0+PE/(0.511*(10**6))
    # elastic cross section
    sigma=np.zeros((len(PE),1800),dtype=np.float64)

    for i in range(len(PE)):
        W=PE[i]
        print('step:',i,'  energy:',int((W-1.0)*(0.511*(10**6))+0.00001))    
        K1=np.power(np.power(W,2)-1,1/2)
        phase_plus=np.zeros(lmax)
        phase_minus=np.zeros(lmax)
        phase_minus[0]=np.pi
        
        # integration for partial waves
        # spin up part
        l=0
        delta1=1e-12
        delta=1.0
        while l<lmax and delta>delta1:
            k=-(l+1)
            #print(l,delta) # for debug

            # Taylor expansion in center
            # Proc.phys.soc, 1965, 85(3):455-462 https://doi.org/10.1088/0370-1328/85/3/306
            phi0=(np.arcsin(-Z[0]/k))*0.5
            phi1=(W+Z[1]-np.cos(2.0*phi0))/(1.0-2.0*k*np.cos(2.0*phi0))
            phi2=(2.0*phi1*np.sin(2.0*phi0)*(1.0-k*phi1)+Z[2])/(2.0-2.0*k*np.cos(2.0*phi0))
            phi3=(2.0*phi2*np.sin(2.0*phi0)*(1.0-2.0*k*phi1)+2*np.power(phi1,2)*np.cos(2.0*phi0)*(1.0-2.0*k*phi1/3.0)+Z[3])/(3.0-2.0*k*np.cos(2.0*phi0))
            phimax=phi0+phi1*rmax+phi2*np.power(rmax,2)+phi3*np.power(rmax,3)

            # RK45 numerical integration
            Dirac_func=lambda t, y: k*np.sin(2*y)/t+Z[0]*TFD_potential.TFD_xi(t)/t+(W-np.cos(2*y))
            sol=integrate.solve_ivp(Dirac_func,[rmax,rinf],[phimax],rtol=rtol,atol=atol)
            r=sol.t[-1]
            phimax=sol.y[0,-1]

            # calculate phase shifts
            t1=K1*special.spherical_jn(l+1,K1*r)-special.spherical_jn(l,K1*r)*((W+1)*np.tan(phimax)+(1+l+k)/r)
            t2=K1*special.spherical_yn(l+1,K1*r)-special.spherical_yn(l,K1*r)*((W+1)*np.tan(phimax)+(1+l+k)/r)
            phase_plus[l]=np.arctan(t1/t2)
            delta=np.abs(phase_plus[l])
            #print(phase_plus[l]) # for debug
            l=l+1
        l_plus_max=l
        # spin down part
        l=1
        delta2=1e-12
        delta=1.0
        while l<lmax and delta>delta2: 
            k=l
            #print(l,delta) # for debug

            # Taylor expansion in center
            # Proc.phys.soc, 1965, 85(3):455-462; https://doi.org/10.1088/0370-1328/85/3/306
            phi0=(np.pi-np.arcsin(-Z[0]/k))*0.5
            phi1=(W+Z[1]-np.cos(2.0*phi0))/(1.0-2.0*k*np.cos(2.0*phi0))
            phi2=(2.0*phi1*np.sin(2.0*phi0)*(1.0-k*phi1)+Z[2])/(2.0-2.0*k*np.cos(2.0*phi0))
            phi3=(2.0*phi2*np.sin(2.0*phi0)*(1.0-2.0*k*phi1)+2*np.power(phi1,2)*np.cos(2.0*phi0)*(1.0-2.0*k*phi1/3.0)+Z[3])/(3.0-2.0*k*np.cos(2.0*phi0))
            phimax=phi0+phi1*rmax+phi2*np.power(rmax,2)+phi3*np.power(rmax,3)
            
            # RK45 numerical integration
            Dirac_func=lambda t, y: k*np.sin(2*y)/t+Z[0]*TFD_potential.TFD_xi(t)/t+(W-np.cos(2*y))
            sol=integrate.solve_ivp(Dirac_func,[rmax,rinf],[phimax],rtol=rtol,atol=atol)
            r=sol.t[-1]
            phimax=sol.y[0,-1]

            # calculate phase shifts
            t1=K1*special.spherical_jn(l+1,K1*r)-special.spherical_jn(l,K1*r)*((W+1)*np.tan(phimax)+(1+l+k)/r)
            t2=K1*special.spherical_yn(l+1,K1*r)-special.spherical_yn(l,K1*r)*((W+1)*np.tan(phimax)+(1+l+k)/r)
            phase_minus[l]=np.arctan(t1/t2)
            delta=np.abs(phase_minus[l])
            #print(phase_minus[l]) # for debug
            l=l+1
        l_minus_max=l
        # sum over all partial waves and use the non-polarization approximation
        # with accuracy 0.1 degree per step
        f=np.zeros(1800,dtype=np.complex)
        g=np.zeros(1800,dtype=np.complex)
        for theta in range(1800):
            f[theta]=0.0+0.0*1j
            g[theta]=0.0+0.0*1j
            for l in range(min(l_plus_max,l_minus_max)):
                f[theta]=f[theta]+(1/(2*1j*K1))*((l+1)*np.exp(2*1j*phase_plus[l])+l*np.exp(2*1j*phase_minus[l])-(2*l+1))*(special.lpmv(0,l,np.cos(theta*np.pi/1800.0)))
            
            
            for l in range(min(l_plus_max,l_minus_max)-1):
               g[theta]=g[theta]+(1/(2*1j*K1))*(np.exp(2*1j*phase_plus[l+1])-np.exp(2*1j*phase_minus[l+1]))*(special.lpmv(1,l+1,np.cos(theta*np.pi/1800.0)))
        f=f/alpha
        #g=f
        g=g/alpha
        sigma[i,:]=np.power(np.abs(f),2)+np.power(np.abs(g),2)
    
    np.save('sigma.npy',sigma)
    np.save('PE.npy',PE)


    #plt.plot(np.log10(np.power(np.abs(f),2)+np.power(np.abs(g),2)))
    #plt.show()


if __name__ == "__main__":
    main()