symmetrize_wann_matrix
--------------------
This code aimed to symmetrize the wannier90 hr file.

Input file:
wannier90.win and wannier90_tb.dat
We also need a wannier90_hr.dat file to compare with our new result.

NOTE: YOU CAN NOT USE MLWF WHEN YOU RUN WANNIER90. (Which means set num_iter=0 in wannier90.win)

An example is given in Example folder.
RUN: ``python3 plot.py``

Not only the Hamiltonian matrix can be symmetrized in this method, but also position element matrix, spin matrix et. al. (Will be implemented in WannierBerri soon https://github.com/stepan-tsirkin/wannier-berri)

Thanks for the help of Changming Yue (UniFR CH).
