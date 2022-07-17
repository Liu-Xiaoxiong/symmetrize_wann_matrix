#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tbmodels as tb
import sys
sys.path.append('..')
from sym_wann import SymWann

seedname="Te"
spin=True
efermi=6.0

        
symmetrize_wann = SymWann(
        
        positions = np.array([[0.274, 0.274, 0.0],
                    [0.726, 0.0, 0.33333333],
                   [0.0, 0.726, 0.66666667]]),
        atom_name = ['Te','Te','Te'],
        proj = ['Te:s','Te:p'],
        soc=True,
        magmom=None,
        DFT_code='vasp',
        seedname="Te")
XX_R, iRvec = symmetrize_wann.symmetrize()



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
model1 = tb.Model.from_wannier_files(hr_file=seedname+"_hr.dat")
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
