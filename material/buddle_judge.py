import numpy as np
import numba as nb

n_box=250
d_layer=1000.0*0.062/(2.25*0.0529)
cross_rand_m_x=np.random.rand(n_box)
cross_rand_m_y=np.random.rand(n_box)

@nb.jit
def cyliner_buddle(r,r0,a0,d0):
    if r[2]<d_layer:
        return 1
    if r[2]>=d0:
        return 1
    rx=int(np.floor(r[0]/a0+0.5))
    ry=int(np.floor(r[1]/a0+0.5))

    rp=np.sqrt(np.power(r[0]-rx*a0,2)+np.power(r[1]-ry*a0,2))
    rate=0.4
    if rp>2*rate*r0 and rp<2*np.sqrt(1+rate**2)*r0:
        return 1
    else:
        return 0

@nb.jit
def G_cross(z,r0,a0,d0):
    '''
    structure function for each CNT
    '''
    beta=n_box/1e6
    b0=0.5*a0-2*r0
    z0=0.5*d0/n_box
    mu=0.3*z0
    c0=1.0/(np.exp(beta*(-z+mu+z0))+1)+1.0/(np.exp(beta*(z+mu-z0))+1)
    return b0*c0+r0

@nb.jit
def cross_buddle(r,r0,a0,d0):
    # 后两行表示表面石墨层
    if r[2]<d_layer:
        return 1    
    if r[2]>=d0:
        #return 0
        print('warning!')
        return 1
    z0=d0/n_box
    rz=int(np.floor(r[2]/z0))
    Gz=G_cross(r[2]-rz*z0,r0,a0,d0)
    #Gz=0.25*a0
    rx=int(np.floor(r[0]/a0+0.5-cross_rand_m_x[rz]))
    ry=int(np.floor(r[1]/a0+0.5-cross_rand_m_y[rz]))
    '''
    if np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0-Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0-Gz,2))<r0:
        return 1
    if np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0+Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0-Gz,2))<r0:
        return 1
    if np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0-Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0+Gz,2))<r0:
        return 1
    if np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0+Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0+Gz,2))<r0:
        return 1
    '''
    rate1=0.4
    if np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0-Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0-Gz,2))>rate1*r0 and np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0-Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0-Gz,2))<np.sqrt(1+rate1**2)*r0:
        return 1
    if np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0+Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0-Gz,2))>rate1*r0 and np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0+Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0-Gz,2))<np.sqrt(1+rate1**2)*r0:
        return 1
    if np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0-Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0+Gz,2))>rate1*r0 and np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0-Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0+Gz,2))<np.sqrt(1+rate1**2)*r0:
        return 1
    if np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0+Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0+Gz,2))>rate1*r0 and np.sqrt(np.power(r[0]-rx*a0-cross_rand_m_x[rz]*a0+Gz,2)+np.power(r[1]-ry*a0-cross_rand_m_y[rz]*a0+Gz,2))<np.sqrt(1+rate1**2)*r0:
        return 1
    return 0