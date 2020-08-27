import numpy as np
import spglib
from atoms import Atoms
np.set_printoptions(threshold=np.inf)

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
						line="{:5d}{:5d}{:5d}{:5d}{:5d}{:14.7f}{:14.7f}\n".format(rx,ry,rz,m+1,n+1,rp,ip)
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
		orbital_dic = {"s":1,"p":3,"d":5,"f":7,"sp3":4,"sp2":3,"l=0":1,"l=1":3,"l=2":5,"l=3":7}	
		projectiondic={}
		self.atom_info = []
		self.symbols_in=[]
		self.positions_in=[]

		for natom in range(atom_op,atom_ed):
			atom_name = win[natom].split()[0]
			position = np.array(win[natom].split()[1:],dtype=float)
			self.symbols_in.append(atom_name)
			self.positions_in.append(list( np.round(np.dot(position,np.linalg.inv(self.lattice)),decimals=6) ))			
		self.num_atom = len(self.symbols_in)
	
		orbital_index_list=[]
		for atom in range(self.num_atom):
			orbital_index_list.append([])	
		orbital_index=0
		if self.spin: orb_spin = 2
		else: orb_spin = 1
		for npro in range(pro_op,pro_ed):
			name = win[npro].split(":")[0].split()[0]
			orb = win[npro].split(":")[1].split()
			if name in projectiondic.keys():
				projectiondic[name]=projectiondic[name]+orb			
			else:
				newdic={name:orb}
				projectiondic.update(newdic)
			for atom in range(self.num_atom):
				if self.symbols_in[atom] == name:
					num_orb = orb_spin * sum([orbital_dic[orb_name] for orb_name in orb])
					orbital_index_old = orbital_index
					orbital_index+=num_orb
					orbital_index_list[atom] += [ i for i in range(orbital_index_old,orbital_index)]
		
		for atom in range(self.num_atom):
			name = self.symbols_in[atom]
			if name in projectiondic.keys():
				projection=projectiondic[name]
			else:
				projection=[]
			self.atom_info.append((atom+1,self.symbols_in[atom],self.positions_in[atom],projection,orbital_index_list[atom]))
		print(self.atom_info)
		
	def get_index(lst=None,item=''):
		return [index for (index,value) in enumerate(lst) if value == item]
	
	def findsym(self):
		def show_symmetry(symmetry):
			for i in range(symmetry['rotations'].shape[0]):
				print("  --------------- %4d ---------------" % (i + 1))
				rot = symmetry['rotations'][i]
				trans = symmetry['translations'][i]
				print("  fold: {}".format(self.rot_fold[i]))
				print("  rotation:")
				for x in rot:
					print("     [%2d %2d %2d]" % (x[0], x[1], x[2]))
				print("  translation:")
				print("     (%8.5f %8.5f %8.5f)" % (trans[0], trans[1], trans[2]))

		atom_in=Atoms(symbols=self.symbols_in,cell=list(self.lattice),scaled_positions=self.positions_in,pbc=True)
		print("[get_spacegroup]")
		print("  Spacegroup of "+self.seedname+" is %s." %spglib.get_spacegroup(atom_in))
		self.symmetry = spglib.get_symmetry(atom_in)
		self.nsymm = self.symmetry['rotations'].shape[0]
		self.rot_fold=[]
		for rot in range(self.nsymm):
			position=np.array(self.positions_in)
			trans = self.symmetry['translations'][rot]
			if abs(trans[0]*trans[1]*trans[2]) < 1E-15:
				for i in range(6):
					atom_list=[]
					for atom in range(self.num_atom):
						new_atom =np.round( np.dot(self.symmetry['rotations'][rot],position[atom]),decimals=6)
						atom_list.append(new_atom)
					position=np.array(atom_list)
					if sum(sum(abs(position-self.positions_in))) < 1E-6:
						self.rot_fold.append(i+1)
						break
					else:
						assert i < 5, 'Error: can not find fold of rotation symmetry {}'.format(rot)
					
			else:
				self.rot_fold.append('None')
		
		show_symmetry(self.symmetry)

		'''
		find a mathod can reduce operators to generators
		'''

	def symmetrize(self):
		HH_R=self.HH_R	
		
		#====Hermitization====
		for ir in range(self.nRvec):
			neg_ir= self.iRvec.index( list(-1*np.array(self.iRvec[ir])) )
			if self.spin:
				HH_R[:,:,ir]=(self.HH_R[:,:,ir] + np.conj(np.transpose(self.HH_R[:,:,neg_ir])) )/2.0
			else:	
				HH_R[:,:,ir]=(self.HH_R[:,:,ir] + np.conj(np.transpose(self.HH_R[:,:,neg_ir])) ).real/2.0
		'''				
		#====Time Reversal====
		#syl: (sigma_y)^T *1j, syr: sigma_y*1j
		if self.spin and self.TR:
			base_m = np.eye(self.num_wann//2)
			syl=np.array([[0.0,-1.0],[1.0,0.0]])
			syr=np.array([[0.0,1.0],[-1.0,0.0]])
			ul=np.kron(base_m,syl)
			ur=np.kron(base_m,syr)
			for ir in range(self.nRvec):
				hh_R=(HH_R[:,:,ir] + np.dot(np.dot(ul,np.conj(HH_R[:,:,ir])),ur))/2.0
				HH_R[:,:,ir]=hh_R
		'''
		#======add new blocks generated by rotation.
		nRvec=self.nRvec
		flag=np.ones((self.num_wann,self.num_wann,self.nRvec),dtype=int)
		print(flag)
	#	for rot in range(1,self.nsymm):
		for rot in range(0,1):
			trans = self.symmetry['translations'][rot]
			if abs(trans[0]*trans[1]*trans[2]) < 1E-15:
				for ir in range(self.nRvec):
					Rvec=np.array(self.iRvec[ir])
					position_atom=np.array(self.positions_in)
					for atom in range(self.num_atom):
						atom_position=position_atom[atom] + Rvec
						new_atom =np.round( np.dot(self.symmetry['rotations'][rot],atom_position),decimals=6)
						for atom_index in range(self.num_atom):
							old_atom= np.array(self.positions_in[atom_index])
							diff = (new_atom-old_atom) - np.array((new_atom-old_atom),dtype=int)
							if abs(diff[0])+abs(diff[0])+abs(diff[0])<10E-6:
								print('------------------match------------------------')
								if ir==89:
									print('------------------89898989------------------------')
								if ir==91:
									print('------------------91919191------------------------')
								new_Rvec=list(np.array(np.round((new_atom-old_atom),decimals=6),dtype=int))
								print(self.symmetry['rotations'][rot])
								print('atom_position = \n {}'.format(atom_position))
								print('new atom =  {}'.format(new_atom))
								print('old atom =  {}'.format(old_atom))
								print('new_Rvec  =  {}'.format(new_Rvec))
								print('match_atom_index = {}'.format(atom_index))
								break
							else:
								assert atom_index != self.num_atom,'Error!!!!: no atom can match the new one Rvec = {}, atom_index = {}'.format(self.iRvec[ir],atom_index)
						if new_Rvec in self.iRvec:
							print('Yes!, New Rvec is in self.iRvec')
							new_Rvec_index = self.iRvec.index(new_Rvec)
							print(self.iRvec[new_Rvec_index])
							print(atom_index)
							print('==================================================')
						else:
							print('Sorry!, New Rvec is not in self.iRvec')
							print('==================================================')
							self.iRvec.append(new_Rvec)
							new_Rvec_index=-1
							new_hh_R=np.zeros((self.num_wann,self.num_wann,1))
							new_flag = np.array(new_hh_R,dtype=int)
							HH_R=np.concatenate((HH_R,new_hh_R),axis=2)
							flag=np.concatenate((flag,new_flag),axis=2)
							nRvec+=1
						for hoping_index in list(self.atom_info[atom_index][4]):
							HH_R[:,hoping_index,new_Rvec_index] += HH_R[:,hoping_index,ir]
							flag[:,hoping_index,new_Rvec_index] += 1
		select=flag > 0
		HH_R_average = HH_R
		HH_R_average[select] = np.divide(HH_R[select],flag[select])
		self.nRvec=nRvec
		self.HH_R = HH_R_average
		print(self.iRvec[89])
		print(self.iRvec[91])


