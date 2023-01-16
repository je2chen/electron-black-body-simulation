import numpy as np
import scipy as sci
import numba as nb
import sys

# local files
from scattering_ac_cross_energy import Scattering
from scattering_ac_cross_energy import Ef,Wf,U,a0
from multiprocessing.pool import Pool

N_electron=3125
n_pool=32

@nb.jit(nopython=True)
def barrier_func(U0,E,cos_theta):
    if E*cos_theta**2<=U0:
        return 0.0
    else:
        k=np.sqrt(1.0-U0/(E*cos_theta**2))
        return 4.0*k/np.power(1.0+k,2)
        #return 1

@nb.njit(parallel=True,nogil=True)
def MC(theta0):
    #e0=[0.6]
    #e0=[0.1,0.13,0.16,0.2,0.23,0.26,0.3,0.33,0.36,0.4]
    e0=nb.typed.List([1,1.5,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,19.95])
    #e0=nb.typed.List([1,1.5,2,3,4,5,7,9,11,13,15,17,19])
    #e0=nb.typed.List([1,2])
    ratio=np.zeros(len(e0),dtype=np.float64)
    #count_l=np.zeros(20,dtype=np.float64)

    for i in nb.typed.List(range(len(e0))):
        count=0
        E=1000.0*e0[i]+U
        for k in nb.typed.List((range(N_electron))):
            phi0=0.0
            x0=a0*(101*np.random.rand()-50.5)
            y0=a0*(101*np.random.rand()-50.5)
            theta1=theta0*np.pi/180
            electron=nb.typed.List([0.0,x0,y0,165.0,np.cos(theta1),np.sin(theta1),np.cos(phi0),np.sin(phi0),E,0.0])
            
            electron_list_l=nb.typed.List([electron[0]])
            electron_list_x=nb.typed.List([electron[1]])
            electron_list_y=nb.typed.List([electron[2]])
            electron_list_z=nb.typed.List([electron[3]])
            electron_list_v1=nb.typed.List([electron[4]])
            electron_list_v2=nb.typed.List([electron[5]])
            electron_list_v3=nb.typed.List([electron[6]])
            electron_list_v4=nb.typed.List([electron[7]])
            electron_list_E=nb.typed.List([electron[8]])
            electron_list_state=nb.typed.List([electron[9]])

            #for l1 in nb.typed.List(range(1)):
            while len(electron_list_l)!=0:
                electron_list_new_l=nb.typed.List.empty_list(nb.float64)
                electron_list_new_x=nb.typed.List.empty_list(nb.float64)
                electron_list_new_y=nb.typed.List.empty_list(nb.float64)
                electron_list_new_z=nb.typed.List.empty_list(nb.float64)
                electron_list_new_v1=nb.typed.List.empty_list(nb.float64)
                electron_list_new_v2=nb.typed.List.empty_list(nb.float64)
                electron_list_new_v3=nb.typed.List.empty_list(nb.float64)
                electron_list_new_v4=nb.typed.List.empty_list(nb.float64)
                electron_list_new_E=nb.typed.List.empty_list(nb.float64)
                electron_list_new_state=nb.typed.List.empty_list(nb.float64)

                for j in nb.typed.List(range(len(electron_list_l))):
                    A=Scattering(nb.typed.List([electron_list_l[j],electron_list_x[j],electron_list_y[j],electron_list_z[j],\
                    electron_list_v1[j],electron_list_v2[j],electron_list_v3[j],electron_list_v4[j],electron_list_E[j],electron_list_state[j]]))
                    A.forward()
                    electron_list_new_l.extend(A.secondary_electron_l)
                    electron_list_new_x.extend(A.secondary_electron_x)
                    electron_list_new_y.extend(A.secondary_electron_y)
                    electron_list_new_z.extend(A.secondary_electron_z)
                    electron_list_new_v1.extend(A.secondary_electron_v1)
                    electron_list_new_v2.extend(A.secondary_electron_v2)
                    electron_list_new_v3.extend(A.secondary_electron_v3)
                    electron_list_new_v4.extend(A.secondary_electron_v4)
                    electron_list_new_E.extend(A.secondary_electron_E)
                    electron_list_new_state.extend(A.secondary_electron_state)

                    if A.electron_state==1:
                        count=count+barrier_func(U,A.electron_E,A.electron_v[0])
                        #count_l[int(A.electron_l)]=count_l[int(A.electron_l)]+1

                electron_list_l=electron_list_new_l
                electron_list_x=electron_list_new_x
                electron_list_y=electron_list_new_y
                electron_list_z=electron_list_new_z
                electron_list_v1=electron_list_new_v1
                electron_list_v2=electron_list_new_v2
                electron_list_v3=electron_list_new_v3
                electron_list_v4=electron_list_new_v4
                electron_list_E=electron_list_new_E
                electron_list_state=electron_list_new_state

            #if k%(N_electron/10)==0:
            #    print(k/(N_electron/10),(count+0.0)/(k+1.0))
        ratio[i]=1-(count+0.0)/(N_electron+0.0)
        print('voltage ',e0[i],' keV absorption:',ratio)
        #print(count_l)
    return ratio

def dec_MC(N_electron):
    e_path=MC(N_electron)
    return e_path


def ratiocal():
    mypool=Pool(processes=n_pool)
    input1=(0.0*np.ones(n_pool).astype(np.int64)).tolist()
    #input1=(0*np.ones(n_pool).astype(np.int64)).tolist()
    result=mypool.map_async(dec_MC,input1)
    mypool.close()
    mypool.join()
    absorb_c=sum(result.get())/n_pool
    print('incident angle 0 result with',N_electron*n_pool,'electrons:',absorb_c)
    #print(absorb_ae)
    #np.save('absorb_ae.npy',absorb_ae)
    return absorb_c

def material_scan():
    absorbc=ratiocal()
    material_num=sys.argv[1]
    print(material_num)
    np.savetxt('energy_scan_of_material_'+material_num+'.txt',absorbc)


if __name__ == "__main__":
    material_scan()
    pass
