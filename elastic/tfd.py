import numpy as np

# fine structure constant
alpha=137.03599976

# Thomas-Fermi-Dirac potential
# J. Chem. Phys. 22, 1758 (1954) https://doi.org/10.1063/1.1739890
# J. Chem. Phys. 39, 2200 (1963) https://doi.org/10.1063/1.1701

class TFD(object):
    def __init__(self, Ze):
        self.Ze=Ze
        self.B,self.b=self.TFD_potential_para()
        self.Z=self.TFD_Series_para()
        self.r0=self.TFD_bon()

    def TFD_potential_para(self):
        TFD_table_B={\
            6:np.array([0.026702,0.33124,0.642058]), 
            13:np.array([0.035188,0.39031,0.574502]),
            14:np.array([0.036033,0.39557,0.568397]),
            22:np.array([0.041172,0.42545,0.533378]),
            29:np.array([0.044229,0.44170,0.514071]),
            47:np.array([0.049201,0.46606,0.484739]),
            79:np.array([0.053764,0.48643,0.459806])
            }
        TFD_table_b={\
            6:np.array([30.447,4.6653,1.2953]),
            13:np.array([32.975,5.1949,1.4108]),
            14:np.array([33.286,5.2566,1.4239]),
            22:np.array([35.518,5.6849,1.5145]),
            29:np.array([37.192,5.9955,1.5799]),
            47:np.array([40.797,6.6455,1.7176]),
            79:np.array([45.821,7.5266,1.9068])
            }
        return TFD_table_B[self.Ze],TFD_table_b[self.Ze]/alpha

    def TFD_Series_para(self):
        Z=np.array([0.0,0.0,0.0,0.0])
        Z[0]=self.Ze/alpha
        for i in range(3):
            Z[1]=Z[1]-Z[0]*self.B[i]*self.b[i]
            Z[2]=Z[2]+Z[0]*self.B[i]*(self.b[i]**2)/2
            Z[3]=Z[3]-Z[0]*self.B[i]*(self.b[i]**3)/6
        return Z

    def TFD_xi(self,r):
        y=0.0
        for j in range(3):
            y=y+self.B[j]*np.exp(-self.b[j]*r)
        return y
    
    def TFD_bon(self):
        TFD_table={\
            6:3.8332350,
            13:4.1563552,
            14:4.1854844,
            22:4.3559275,
            29:4.4539578,
            47:4.6141603,
            79:4.7709867
            }
        return TFD_table[self.Ze]*alpha
