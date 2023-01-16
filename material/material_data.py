import numpy as np

alpha=137.03599976
Hatree=0.511*np.power(10,6)/(np.power(alpha,2))
MaxE=1e9
MaxD=1e9
class Material(object):
    def __init__(self,material_name,rho,Ze,Me,Ef,Wf,Eion,penetration_depth,mfp_rate):
        self.material_name=material_name
        self.rho=rho
        self.Ze=Ze
        self.Me=Me
        self.Ef=Ef
        self.Wf=Wf
        self.Eion=Eion
        self.penetration_depth=penetration_depth
        self.kf=np.sqrt(2*self.Ef/Hatree)
        self.rm=0.511/(self.Me*938.0)
        self.mfp_rate=mfp_rate

matertial=[Material('Cu',8.96,29.0,63.5,8.7,4.5,[MaxE,MaxE,MaxE],MaxD,1.0),\
    Material('CNT',1.9,6.0,12.0,20.4,5.1,[MaxE,MaxE,MaxE],MaxD,1.0),\
    Material('Graphite',2.25,6.0,12.0,21.4,4.6,[MaxE,MaxE,MaxE],MaxD,1.0),\
    Material('Ti',4.51,22.0,48.0,13.0,4.33,[MaxE,MaxE,MaxE],MaxD,1.0),\
    Material('Ag',10.49,47.0,108.0,7.2,4.26,[MaxE,MaxE,MaxE],MaxD,1.0),\
    Material('Al',2.7,13.0,27.0,10.49,4.1,[MaxE,MaxE,MaxE],MaxD,1.0),\
    Material('Au',19.32,79.0,197.0,6.0,5.1,[MaxE,MaxE,2236.0],MaxD,1.0),\
    Material('Graphite2',2.25,6.0,12.0,8.5,4.6,[MaxE,MaxE,MaxE],MaxD,1.0)]

'''
structure constants
'''
r0=20/0.0529                                             # radius of CNT
a0=2*r0*np.sqrt(np.pi*2.25/0.062)                        # size of cell
d0=(800.0*1000.0)/0.0529                                 # length of CNT
step_f=10.0                                              # step for free motion

'''
material constants
'''
material_num=2                                            # material table number
Work_path='C:/Users/dell/Documents/CG/EB-simu/database/'  # work path