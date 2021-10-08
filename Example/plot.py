#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import tbmodels as tb
import sys
sys.path.append('..')
from __sym_wann import sym_wann

seedname="Te"
spin=True
TR=True
efermi=6.0

sw=sym_wann(seedname=seedname,spin=spin,TR=TR)
sw.read_tb()
sw.read_win()
sw.findsym()
sw.symmetrize()
#sw.symmetrize_vec()
sw.write_tb()
sw.write_hr()
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
for i in range(sw.num_wann):
	plt.plot(ax,res1[:,i],'b')
for i in range(sw.num_wann):
	plt.plot(ax,res2[:,i],'r')
plt.plot(ax,res1[:,0],'b',label='hr')
plt.plot(ax,res2[:,0],'r',label='sym_hr')
plt.legend()

plt.ylim(5,7)
plt.savefig(seedname+".png") 
plt.show()
