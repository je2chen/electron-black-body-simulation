import sys
import re
m_number=sys.argv[1]

f=open('scattering_ac_cross.py','r',encoding='utf-8')
temp=None
s=''
s1=f.readline()
while(temp==None):
    s=s+s1
    s1=f.readline()
    temp=re.match('material_num=',s1)
#print(s)
s=s+'material_num='+m_number+'\n'
temp=s1
while(temp!=''):
    temp=f.readline()
    s=s+temp
#print(s)
f.close()
f=open('scattering_ac_cross_energy.py','w',encoding='utf-8')
f.write(s)
f.close()
