#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tbmodels as tb
import sys
sys.path.append('..')
from sym_wann import SymWann
from _utility import read_tb, write_hr, write_tb 

seedname="wannier90"
efermi=6.0

num_wann, nRvec, iRvec, XX_R, real_lattice = read_tb('Mn3Sn')

symmetrize_wann = SymWann(
            num_wann=num_wann,
            lattice=real_lattice,
            positions=np.array([
                 [0.6666667,       0.8333333,       0],
                 [0.1666667,       0.3333333,       0],
                 [0.6666667,       0.3333333,       0],
                 [0.3333333,       0.1666667,       0.5],
                 [0.8333333,       0.6666667,       0.5],
                 [0.3333333,       0.6666667,       0.5],
                 [0.8333333,       0.1666667,       0.5],
                 [0.1666667,       0.8333333,       0]]),
            atom_name=['Mn','Mn','Mn','Mn','Mn','Mn','Sn','Sn'],
            projections=['Mn:s;d','Sn:p'],
            magmom=[
                 [0, 2, 0],
                 [np.sqrt(3), -1,  0],
                 [-np.sqrt(3), -1, 0],
                 [0, 2, 0],
                 [np.sqrt(3), -1, 0],
                 [-np.sqrt(3), -1, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
            soc=True,
            DFT_code='vasp',
            iRvec=iRvec,
            XX_R=XX_R,
        )

XX_R, iRvec,wc = symmetrize_wann.symmetrize()
write_hr(num_wann, np.shape(iRvec)[0], iRvec, XX_R, real_lattice) 
write_tb(num_wann, np.shape(iRvec)[0], iRvec, XX_R, real_lattice) 



kpatha=np.array(
[[0, 0, 0.5 ],
[0, 0, 0],
[0.333333, 0.3333333, 0.5],
[0, 0, 0.5 ],
[0.5, 0.0, 0.5  ],
[0.333333, 0.3333333, 0.5],
[0.333333, 0.3333333, 0.0],
[0, 0, 0],
[0.5, 0, 0]]
)


nk=200
kpa=[]

ax=np.linspace(0,nk*(len(kpatha)-1),nk*(len(kpatha)-1))

for npath in range(len(kpatha)-1):
    dka=(kpatha[npath+1]-kpatha[npath])/float(nk)
    for n in range(nk):
        kpa.append(list(kpatha[npath]+n*dka) )
model1 = tb.Model.from_wannier_files(hr_file="Mn3Sn_hr.dat")
model2 = tb.Model.from_wannier_files(hr_file=seedname+"_sym_hr.dat")

res1=np.array(model1.eigenval(kpa))
res2=np.array(model2.eigenval(kpa))
plt.figure()
for i in range(symmetrize_wann.num_wann):
	plt.plot(ax,res1[:,i],'b')
for i in range(symmetrize_wann.num_wann):
	plt.plot(ax,res2[:,i],'r')
plt.plot(ax,res1[:,0],'b',label='hr')
plt.plot(ax,res2[:,0],'r',label='sym_hr')
plt.legend()

plt.ylim(5,7)
plt.savefig(seedname+".png") 
plt.show()
