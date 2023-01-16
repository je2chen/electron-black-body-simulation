import sys
import re
#sc_number = 0 : cross bundle, 1 : cylinder bundle, 2: bulk
sc_number=sys.argv[1]

f=open('scattering_ac_cross.py','r',encoding='utf-8')
temp=None
s=''
s1=f.readline()
while(temp==None):
    s=s+s1
    s1=f.readline()
    #print(s1)
    temp=re.match(r'        # buddle judge options',s1)
s=s+s1
#print(s)
if sc_number=='0':
    s=s+r'        return buddle_judge.cross_buddle(self.electron_r,r0,a0,d0)'
    s=s+'\n'
elif sc_number=='1':
    s=s+r'        return buddle_judge.cyliner_buddle(self.electron_r,r0,a0,d0)'
    s=s+'\n'
else:
    s=s+r'        return 1'
    s=s+'\n'
s1=f.readline()
#print(s)
while(temp!=''):
    temp=f.readline()
    s=s+temp
#print(s)
f.close()
f=open('scattering_ac_cross_energy.py','w',encoding='utf-8')
f.write(s)
f.close()
