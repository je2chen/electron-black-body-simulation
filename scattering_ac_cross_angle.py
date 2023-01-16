import numpy as np
#from matplotlib import pyplot as plt
import numba as nb
from material import material_data
from material import buddle_judge

'''
physics constants
'''
alpha=material_data.alpha                              # fine structure constant
Hatree=material_data.Hatree                            # atomic energy,about 27.2114 eV
omega_b0=np.log10(0.0001/Hatree)                       # minimum energy for dielectric function
omega_a0=(np.log10(30000.0/Hatree)-omega_b0)/999.0     # maximum energy for dielectric function


'''
structure constants
'''
r0=5/0.0529                    # radius of CNT
a0=2*r0*np.sqrt(np.pi*2.25/0.062)  # size of cell
d0=(800.0*1000.0)/0.0529          # length of CNT
step_f=10.0                       # step for free motion

'''
material constants
'''
material_num=2 #可以改的数字[0,3,5,6]                                         # material table number
Work_path='C:/Users/dell/Documents/CG/EB-simu/database/'  # work path

# load the data of material
rho=material_data.matertial[material_num].rho
Ze=material_data.matertial[material_num].Ze
Me=material_data.matertial[material_num].Me
Ef=material_data.matertial[material_num].Ef
Wf=material_data.matertial[material_num].Wf 
Eion=material_data.matertial[material_num].Eion
U=Ef+Wf
Eion1=Eion[0]
Eion2=Eion[1]
Eion3=Eion[2]

kf=material_data.matertial[material_num].kf
rm=material_data.matertial[material_num].rm
penetration_depth=material_data.matertial[material_num].penetration_depth

Folderpath=Work_path+material_data.matertial[material_num].material_name

# elastic scattering parameters
elastic_mfp=np.load(Folderpath+'/elastic/elastic_mfp.npy')
elastic_sc=np.load(Folderpath+'/elastic/elastic_sc.npy')

# inelastic scattering parameters
inelastic_mfp=np.load(Folderpath+'/inelastic/inelastic_mfp.npy')
inelastic_le=np.load(Folderpath+'/inelastic/inelastic_le.npy')
inelastic_sc=np.load(Folderpath+'/inelastic/inelastic_sc.npy')

@nb.jit
def hash_omega(energy_loss):
    '''
    find corresponding index in table for given energy loss 
    '''
    return (np.log10(energy_loss/Hatree)-omega_b0)/omega_a0

@nb.jit
def hash_E(E_input):
    '''
    find corresponding index in table for given energy
    '''
    if E_input<=100:
        a=E_input-1.0
    else:
        a=99.0+500.0*((np.log10(E_input)-2.0)/2.3)
    return a

@nb.jit
def electron_v_update(electron_v,theta_par,phi_par):
    '''
    electron direction update function
    '''
    cos_theta_old=electron_v[0]
    sin_theta_old=electron_v[1]
    cos_phi_old=electron_v[2]
    sin_phi_old=electron_v[3]

    electron_v_new=nb.typed.List([0.0,0.0,0.0,0.0])

    electron_v_new[0]=cos_theta_old*np.cos(theta_par)-sin_theta_old*np.sin(theta_par)*np.cos(phi_par)
    electron_v_new[1]=np.sqrt(1.0-np.power(electron_v_new[0],2))

    # singular case
    if electron_v_new[0]==1.0:
        electron_v_new[1]=0.0
        sin_phi_minus_phi=np.sin(phi_par)
        cos_phi_minus_phi=np.cos(phi_par)
    else:
        if sin_theta_old==0.0:
            sin_phi_minus_phi=np.sin(phi_par)
            cos_phi_minus_phi=np.cos(phi_par)
        else:
            sin_phi_minus_phi=np.sin(theta_par)*np.sin(phi_par)/electron_v_new[1]
            cos_phi_minus_phi=(np.cos(theta_par)-cos_theta_old*electron_v_new[0])/(sin_theta_old*electron_v_new[1])
    electron_v_new[2]=cos_phi_old*cos_phi_minus_phi-sin_phi_old*sin_phi_minus_phi
    electron_v_new[3]=sin_phi_old*cos_phi_minus_phi+cos_phi_old*sin_phi_minus_phi

    return electron_v_new

@nb.jit(nopython=True)
def barrier_func(U0,E,cos_theta):
    '''
    probability of tunneling
    '''
    if E*cos_theta**2<=U0:
        return 0.0
    else:
        k=np.sqrt(1.0-U0/(E*cos_theta**2))
        return 4.0*k/np.power(1.0+k,2)


spec=[
    ('electron_l',nb.float64),
    ('electron_r',nb.types.ListType(nb.float64)),
    ('electron_v',nb.types.ListType(nb.float64)),
    ('electron_E',nb.float64),
    ('electron_state',nb.float64),
    ('electron_path_x',nb.types.ListType(nb.float64)),
    ('electron_path_y',nb.types.ListType(nb.float64)),
    ('electron_path_z',nb.types.ListType(nb.float64)),
    ('secondary_electron_x',nb.types.ListType(nb.float64)),
    ('secondary_electron_y',nb.types.ListType(nb.float64)),
    ('secondary_electron_z',nb.types.ListType(nb.float64)),
    ('secondary_electron_v1',nb.types.ListType(nb.float64)),
    ('secondary_electron_v2',nb.types.ListType(nb.float64)),
    ('secondary_electron_v3',nb.types.ListType(nb.float64)),
    ('secondary_electron_v4',nb.types.ListType(nb.float64)),
    ('secondary_electron_E',nb.types.ListType(nb.float64)),
    ('secondary_electron_l',nb.types.ListType(nb.float64)),
    ('secondary_electron_state',nb.types.ListType(nb.float64)),
]
@nb.experimental.jitclass(spec)
class Scattering(object):
    def __init__(self,electron):
        
        self.electron_l=electron[0]                               # electron cascade level
        self.electron_r=electron[1:4]                             # electron position
        self.electron_v=electron[4:8]                             # direction of electron velocity, cos theta, sin theta, cos phi, sin phi
        self.electron_E=electron[8]                               # electron energy
        self.electron_state=electron[9]                           # electron state: 0: ready for scattering; 1: in vaccum 3: transmitted 2: absorbed

        # electron path
        self.electron_path_x=nb.typed.List.empty_list(nb.float64)
        self.electron_path_y=nb.typed.List.empty_list(nb.float64)
        self.electron_path_z=nb.typed.List.empty_list(nb.float64)

        # secondary electrons
        self.secondary_electron_x=nb.typed.List.empty_list(nb.float64)
        self.secondary_electron_y=nb.typed.List.empty_list(nb.float64)
        self.secondary_electron_z=nb.typed.List.empty_list(nb.float64)
        self.secondary_electron_v1=nb.typed.List.empty_list(nb.float64)
        self.secondary_electron_v2=nb.typed.List.empty_list(nb.float64)
        self.secondary_electron_v3=nb.typed.List.empty_list(nb.float64)
        self.secondary_electron_v4=nb.typed.List.empty_list(nb.float64)
        self.secondary_electron_E=nb.typed.List.empty_list(nb.float64)
        self.secondary_electron_l=nb.typed.List.empty_list(nb.float64)
        self.secondary_electron_state=nb.typed.List.empty_list(nb.float64)

        (self.electron_path_x).append(self.electron_r[0])
        (self.electron_path_y).append(self.electron_r[1])
        (self.electron_path_z).append(self.electron_r[2])

    # buddle judge function
    def buddle_j(self):
        #return buddle_judge.cyliner_buddle(self.electron_r,r0,a0,d0)
        #return 1
        return buddle_judge.cross_buddle(self.electron_r,r0,a0,d0)

    def forward(self):
        while 1:
            # electron with energy less than work function
            if self.electron_E<U and self.buddle_j()==1:
                self.electron_state=3.0
                break
            in_buddle=self.buddle_j()
            if in_buddle==1:
                next_step=self.sample_step()
                self.update_position(next_step)
                if self.electron_r[2]<0.0:
                    self.electron_state=1.0
                    break
                if self.electron_r[2]>penetration_depth:
                    #print('penetrate!')
                    self.electron_state=2.0
                    break
                if self.sample_scattering_type():
                    self.elastic_scattering()
                else:
                    self.inelastic_scattering()
            elif in_buddle==0:
                while self.buddle_j()==0 and self.electron_r[2]>=0.0 and self.electron_r[2]<=penetration_depth:
                    self.update_position(step_f)
                if self.electron_r[2]<0:
                    self.electron_state=1.0
                    break
                if self.electron_r[2]>penetration_depth:
                    #print('penetrate!')
                    self.electron_state=2.0
                    break   

    def sample_step(self):
        index_e=hash_E(self.electron_E)
        index_int=int(index_e)*((index_e>0))

        index=hash_E(self.electron_E-Ef)
        index=index*(index>0)
        index_int_in=int(index)
        
        t_mfp=1.0/(1.0/elastic_mfp[index_int]+1.0/inelastic_mfp[index_int_in])
        return -t_mfp*np.log(np.random.rand())

    def update_position(self,s_step):
        self.electron_r[0]=self.electron_r[0]+s_step*self.electron_v[1]*self.electron_v[2]
        self.electron_r[1]=self.electron_r[1]+s_step*self.electron_v[1]*self.electron_v[3]
        self.electron_r[2]=self.electron_r[2]+s_step*self.electron_v[0]
        pass
        
    def sample_scattering_type(self):
        index_e=hash_E(self.electron_E)
        index_int=int(index_e)*(index_e>=0)

        index=hash_E(self.electron_E-Ef)
        index=index*(index>=0)
        index_int_in=int(index)
        t_mfp=1.0/(1.0/elastic_mfp[index_int]+1.0/inelastic_mfp[index_int_in])
        scattering_type=t_mfp/elastic_mfp[index_int]
        return np.random.rand()<scattering_type
            
    def elastic_scattering(self):
        index=hash_E(self.electron_E)
        index_int=int(index)*(index>=0.0)
        theta_par=np.pi*elastic_sc[index_int,np.random.randint(10001)]/180.0
        phi_par=2.0*np.pi*np.random.rand()

        self.electron_v=electron_v_update(self.electron_v,theta_par,phi_par)
        pass

    def inelastic_scattering(self):
        index=hash_E(self.electron_E-Ef)
        index=index*(index>=0)
        index_int=int(index)

        #sample loss energy
        le=inelastic_le[index_int,np.random.randint(0,10001)]*Hatree
        le_index=hash_omega(le)
        le_index_int=int(le_index)*(le_index>=0)

        #sample angle
        theta_par=inelastic_sc[index_int,le_index_int,np.random.randint(0,1001)]
        phi_par=2.0*np.pi*np.random.rand()
        electron_v_old=self.electron_v
        # update electron direction
        self.electron_v=electron_v_update(electron_v_old,theta_par,phi_par)
        
        # SE generation
        # ionization judgement
        if le>Eion1:
            se_E=le-Eion1
            se_theta_par=np.pi/2.0-theta_par
            se_phi_par=np.pi+phi_par
        elif le>Eion2:
            se_E=le-Eion2
            se_theta_par=np.pi/2.0-theta_par
            se_phi_par=np.pi+phi_par
        elif le>Eion3:
            se_E=le-Eion3            
            se_theta_par=np.pi/2.0-theta_par
            se_phi_par=np.pi+phi_par
        else:
            q=np.sqrt((4*self.electron_E-2*le-4*np.sqrt(self.electron_E*(self.electron_E-le))*np.cos(theta_par))/Hatree)
            q_minus=-kf+np.sqrt(kf**2+2.0*le/Hatree)
            q_plus=kf+np.sqrt(kf**2+2.0*le/Hatree)          
            # single state
            if q>=q_minus and q<=q_plus:
                kz=(2*(le/Hatree)-q**2)/(2.0*q)
                if kz+q<kf:
                    a=(kz+q)**2-kz**2
                    b=kf**2-(kz+q)**2
                else:
                    a=kf**2-kz**2
                    b=0.0
                se_E=0.0
                while se_E<Ef:
                    kr=np.sqrt(a*np.random.rand()+b)
                    se_E=(kr**2+(kz+q)**2)*Hatree/2.0
                    #SE_theta_par=np.arccos(2.0*np.random.rand()-1.0)
                    #SE_phi_par=2*np.pi*np.random.rand()
                k1=np.sqrt(2*self.electron_E/Hatree)
                k2=np.sqrt(2*(self.electron_E-le)/Hatree)
                qx=-k2*np.cos(phi_par)*np.sin(theta_par)
                qy=-k2*np.sin(phi_par)*np.sin(theta_par)
                qz=k1-k2*np.cos(theta_par)
                cos_theta1=qz/q
                sin_theta1=np.sqrt(1-cos_theta1**2)
                cos_phi1=qx/(q*sin_theta1)
                sin_phi1=qy/(q*sin_theta1)
                k_new_v=electron_v_update([cos_theta1,sin_theta1,cos_phi1,sin_phi1],np.arccos((kz+q)/np.sqrt(kr**2+(kz+q)**2)),2*np.pi*np.random.rand())
                se_theta_par=np.arccos(k_new_v[0])
                se_phi_par=np.arctan2(k_new_v[3],k_new_v[2])
            else:
                # plasmon case
                se_E=0.0
                while se_E<Ef:
                    k1=min(le/Ef,1.0)
                    R1=k1*(np.random.rand()-1.0)+1.0
                    R2=np.random.rand()
                    if np.sqrt(R1*(R1+le/Ef)/(1.0+le/Ef))>R2:
                        se_E=Ef*R1+le
                se_theta_par=np.arccos(2.0*np.random.rand()-1.0)
                se_phi_par=2*np.pi*np.random.rand()

        se_v=electron_v_update(electron_v_old,se_theta_par,se_phi_par)

        (self.secondary_electron_x).append(self.electron_r[0])
        (self.secondary_electron_y).append(self.electron_r[1])
        (self.secondary_electron_z).append(self.electron_r[2])
        #(self.secondary_electron_x).append(x1)
        #(self.secondary_electron_y).append(y1)
        #(self.secondary_electron_z).append(self.electron_r[2])        
        (self.secondary_electron_v1).append(se_v[0])
        (self.secondary_electron_v2).append(se_v[1])
        (self.secondary_electron_v3).append(se_v[2])
        (self.secondary_electron_v4).append(se_v[3])
        (self.secondary_electron_l).append(self.electron_l+1)
        (self.secondary_electron_state).append(0.0)
        (self.secondary_electron_E).append(se_E)
        
        
        self.electron_E=self.electron_E-le
        pass