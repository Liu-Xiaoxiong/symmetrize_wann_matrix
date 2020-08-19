import numpy as np
import spglib
from atoms import Atoms

class sym_wann():
	def __init__(self,seedname="wannier90",spin=False,TR=True):
		self.seedname=seedname
		self.spin=spin
		self.TR=TR

	def read_hr(self):
		f=open(self.seedname+"_hr.dat","r")
		f.readline()
		self.num_wann=int(f.readline())
		self.nRvec=int(f.readline())
		self.Ndegen=[]
		while len(self.Ndegen)<self.nRvec:
			self.Ndegen+=f.readline().split()
		self.Ndegen=np.array(self.Ndegen,dtype=int)
		self.iRvec=[]
		self.HH_R=np.zeros((self.num_wann,self.num_wann,self.nRvec),dtype=complex)
		for ir in range(self.nRvec):
			for n in range(self.num_wann):
				for m in range(self.num_wann):
					hoppingline=f.readline()
					if m==0 and n==0:
						self.iRvec.append(list(np.array(hoppingline.split()[0:3],dtype=int)))
					self.HH_R[m,n,ir]=(float(hoppingline.split()[5])+1j*float(hoppingline.split()[6]))/float(self.Ndegen[ir])


	def write_hr(self):
		name=self.seedname+"_sym_hr.dat"	
		Ndegen=list(np.ones((self.nRvec),dtype=int))
		with open(name,"w") as f:
			f.write("symmetrize wannier hr\n"+str(self.num_wann)+"\n"+str(self.nRvec)+"\n")
			nl = np.int32(np.ceil(self.nRvec/15.0))
			for l in range(nl):
				line="    "+'    '.join([str(np.int32(i)) for i in Ndegen[l*15:(l+1)*15]])
				f.write(line+"\n")
			for ir in range(self.nRvec):
				rx = self.iRvec[ir][0];ry = self.iRvec[ir][1];rz = self.iRvec[ir][2]
				for n in range(self.num_wann):
					for m in range(self.num_wann):
						rp =self.HH_R[m,n,ir].real
						ip =self.HH_R[m,n,ir].imag
						line="{:5d}{:5d}{:5d}{:5d}{:5d}{:20.14f}{:20.14f}\n".format(rx,ry,rz,m+1,n+1,rp,ip)
						f.write(line)
			f.close()

	def read_win(self):
		name=self.seedname+".win"
		win = [line for line in open(name) if line.strip()]
		for ii, line in enumerate(win):
			if "begin projections" in line:
				pro_op=ii+1
			if "end projections" in line:
				pro_ed=ii
			if "begin unit_cell" in line:
				lattice_op=ii+1
			if "end unit_cell" in line:
				lattice_ed=ii
			if "begin atoms_cart" in line:
				atom_op=ii+1
			if "end atoms_cart" in line:
				atom_ed=ii
			if "spinors" in line:
				if "t" or "T" in line:
					self.spin=True
				else:
					self.spin=False 
		self.lattice = np.array([line.split() for line in win[lattice_op:lattice_ed]],dtype=float)
		projectiondic={} 
		for npro in range(pro_op,pro_ed):
			name = win[npro].split(":")[0].split()[0]
			orb = win[npro].split(":")[1].split()
			if name in projectiondic.keys():
				projectiondic[name]=projectiondic[name]+orb			
			else:
				newdic={name:orb}
				projectiondic.update(newdic)
			
		self.atom_info = []
		self.symbols_in=[]
		self.positions_in=[]
		for natom in range(atom_op,atom_ed):
			atom_name = win[natom].split()[0]
			position = np.array(win[natom].split()[1:],dtype=float)
			projection = projectiondic[atom_name]
			self.atom_info.append((natom - atom_op +1,atom_name,position,projection))
			self.symbols_in.append(atom_name)
			self.positions_in.append(list( np.round(np.dot(position,np.linalg.inv(self.lattice)),decimals=6) ))			

		print(self.atom_info)
	
	def findsym(self):
		def show_symmetry(symmetry):
			for i in range(symmetry['rotations'].shape[0]):
				print("  --------------- %4d ---------------" % (i + 1))
				rot = symmetry['rotations'][i]
				trans = symmetry['translations'][i]
				print("  rotation:")
				for x in rot:
					print("     [%2d %2d %2d]" % (x[0], x[1], x[2]))
				print("  translation:")
				print("     (%8.5f %8.5f %8.5f)" % (trans[0], trans[1], trans[2]))

		def show_lattice(lattice):
			print("Basis vectors:")
			for vec, axis in zip(lattice, ("a", "b", "c")):
				print("%s %10.5f %10.5f %10.5f" % (tuple(axis,) + tuple(vec)))

		def show_cell(lattice, positions, numbers):
			show_lattice(lattice)
			print("Atomic points:")
			for p, s in zip(positions, numbers):
				print("%2d %10.5f %10.5f %10.5f" % ((s,) + tuple(p)))
		atom_in=Atoms(symbols=self.symbols_in,cell=list(self.lattice),scaled_positions=self.positions_in,pbc=True)
		print("[get_spacegroup]")
		print("  Spacegroup of "+self.seedname+" is %s." %spglib.get_spacegroup(atom_in))
		self.symmetry = spglib.get_symmetry(atom_in)
		show_symmetry(self.symmetry)
		print(symmetry)
		                

	def symmetrize(self):
		HH_R=self.HH_R
		#====Hermitization====
		for ir in range(self.nRvec):
			neg_ir= self.iRvec.index( list(-1*np.array(self.iRvec[ir])) )
			if self.spin:
				HH_R[:,:,ir]=(self.HH_R[:,:,ir] + np.conj(np.transpose(self.HH_R[:,:,neg_ir])) )/2.0
			else:	
				HH_R[:,:,ir]=(self.HH_R[:,:,ir] + np.conj(np.transpose(self.HH_R[:,:,neg_ir])) ).real/2.0
			
		#====Time Reversal====
		#syl: (sigma_y)^T *1j, syr: sigma_y*1j
		if self.spin and self.TR:
			base_m = np.eye(self.num_wann//2)
			syl=np.array([[0.0,-1.0],[1.0,0.0]])
			syr=np.array([[0.0,1.0],[-1.0,0.0]])
			ul=np.kron(syl,base_m)
			ur=np.kron(syr,base_m)
			for ir in range(self.nRvec):
				hh_R=(HH_R[:,:,ir] + np.dot(np.dot(ul,np.conj(HH_R[:,:,ir])),ur))/2.0
				hh_R=np.array(hh_R)
				#print('hh_R shape\n {}'.format(len(hh_R[0])))
				#print('HH_R shape\n {}'.format(len(HH_R[0,:,ir])))
				HH_R[0:self.num_wann:2,0:self.num_wann:2,ir]=hh_R[0:self.num_wann//2,0:self.num_wann//2]
				HH_R[0:self.num_wann:2,1:self.num_wann:2,ir]=hh_R[0:self.num_wann//2,self.num_wann//2:self.num_wann]
				HH_R[1:self.num_wann:2,0:self.num_wann:2,ir]=hh_R[self.num_wann//2:self.num_wann,0:self.num_wann//2]
				HH_R[1:self.num_wann:2,1:self.num_wann:2,ir]=hh_R[self.num_wann//2:self.num_wann,self.num_wann//2:self.num_wann]


		#======add new blocks generated by rotation.
		flag=[1 for i in range(self.nRvec)]
		 


		self.HH_R = HH_R

