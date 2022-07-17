symmetrize_wann_matrix
--------------------
This code aimed to symmetrize the wannier90 hr file and tb file.

Input file:
wannier90_tb.dat

NOTE: YOU CAN NOT USE MLWF WHEN YOU RUN WANNIER90. (Which means set num_iter=0 in wannier90.win)

An example is given in Example folder.
RUN: ``python3 Te_plot.py``

I only use tb file as input, so only Hamiltonian and position element matrix can be symmetrized here.
The symmetrization of other matrix already be implemented in WannierBerri https://github.com/wannier-berri/wannier-berri

Thanks for the help from Changming Yue (UniFR CH).
