import numpy as np
import spglib
import numpy.linalg as la
from atoms import Atoms
import sympy as sym
np.set_printoptions(threshold=np.inf,linewidth=500)

test1 = 1
test2 = 221
st1=0
st2=10

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
		self.nRvec_old=self.nRvec*1
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
						line="{:5d}{:5d}{:5d}{:5d}{:5d}{:12.6f}{:12.6f}\n".format(rx,ry,rz,m+1,n+1,rp,ip)
						f.write(line)
			f.close()
	def write_hr_old(self):
		name=self.seedname+"_sym_old_hr.dat"	
		Ndegen=list(np.ones((self.nRvec_old),dtype=int))
		with open(name,"w") as f:
			f.write("symmetrize wannier hr\n"+str(self.num_wann)+"\n"+str(self.nRvec_old)+"\n")
			nl = np.int32(np.ceil(self.nRvec_old/15.0))
			for l in range(nl):
				line="    "+'    '.join([str(np.int32(i)) for i in Ndegen[l*15:(l+1)*15]])
				f.write(line+"\n")
			for ir in range(self.nRvec_old):
				rx = self.iRvec[ir][0];ry = self.iRvec[ir][1];rz = self.iRvec[ir][2]
				for n in range(self.num_wann):
					for m in range(self.num_wann):
						rp =self.HH_R[m,n,ir].real
						ip =self.HH_R[m,n,ir].imag
						line="{:5d}{:5d}{:5d}{:5d}{:5d}{:12.6f}{:12.6f}\n".format(rx,ry,rz,m+1,n+1,rp,ip)
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
		self.orbital_dic = {"s":1,"p":3,"d":5,"f":7,"sp3":4,"sp2":3,"l=0":1,"l=1":3,"l=2":5,"l=3":7}	
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
		self.wannier_center=np.zeros([self.num_wann,3],dtype=float)
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
					num_orb = orb_spin * sum([self.orbital_dic[orb_name] for orb_name in orb])
					orbital_index_old = orbital_index
					orbital_index+=num_orb
					orbital_index_list[atom] += [ i for i in range(orbital_index_old,orbital_index)]
					for wc in range(orbital_index_old,orbital_index):
						self.wannier_center[wc]=np.dot(self.positions_in[atom],self.lattice)
		
		for atom in range(self.num_atom):
			name = self.symbols_in[atom]
			if name in projectiondic.keys():
				projection=projectiondic[name]
			else:
				projection=[]
			self.atom_info.append((atom+1,self.symbols_in[atom],self.positions_in[atom],projection,orbital_index_list[atom]))
		print(self.atom_info)
	#	print(self.wannier_center)
		
	
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
	def get_angle(self,sin,cos):
		if round(sin,2) == 0:
			sin = sin/abs(sin)
		angle = np.arcsin(sin)
		if round(cos,2) == 0:
			angle = np.arcsin(sin)
		elif sin>=0:
			angle = np.arctan(sin/cos)%np.pi
		elif sin < 0:
			angle = np.arctan(sin/cos)%np.pi + np.pi
		return round(angle*(2*np.pi),2)

	def rot_orb(self,orb_symbol,rot_glb):
		x = sym.Symbol('x')
		y = sym.Symbol('y')
		z = sym.Symbol('z')
		def ss(x,y,z):
			return 1+0*(x+y+z)
		def pz(x,y,z):
			return z
		def px(x,y,z):
			return x
		def py(x,y,z):
			return y
		def dz2(x,y,z):
			return (2*z*z-x*x-y*y)/(2*sym.sqrt(3.0))
		def dxz(x,y,z):
			return x*z
		def dyz(x,y,z):
			return y*z
		def dx2_y2(x,y,z):
			return (x*x-y*y)/2
		def dxy(x,y,z):
			return x*y
		def fz3(x,y,z):
			return z*(2*z*z-3*x*x-3*y*y)/(2*sym.sqrt(15.0))
		def fxz2(x,y,z):
			return x*(4*z*z-x*x-y*y)/(2*sym.sqrt(10.0))
		def fyz2(x,y,z):
			return y*(4*z*z-x*x-y*y)/(2*sym.sqrt(10.0))
		def fzx2_zy2(x,y,z):
			return z*(x*x-y*y)/2
		def fxyz(x,y,z):
			return x*y*z
		def fx3_3xy2(x,y,z):
			return x*(x*x-3*y*y)/(2*sym.sqrt(6.0))
		def f3yx2_y3(x,y,z):
			return y*(3*x*x-y*y)/(2*sym.sqrt(6.0))
		orb_s = [ss]
		orb_p = [pz,px,py]
		orb_d = [dz2,dxz,dyz,dx2_y2,dxy]
		orb_f = [fz3,fxz2,fyz2,fzx2_zy2,fxyz,fx3_3xy2,f3yx2_y3]
		orb_function_dic={'s':orb_s,'p':orb_p,'d':orb_d,'f':orb_f}
		orb_chara_dic={'s':[],'p':[z,x,y],'d':[z*z,x*z,y*z,x*x,x*y,y*y],'f':[z*z*z,x*z*z,y*z*z,z*x*x,x*y*z,x*x*x,y*y*y]}
		orb_dim = self.orbital_dic[orb_symbol]
		orb_rot_mat = np.zeros((orb_dim,orb_dim),dtype=float)
		xp = np.dot(np.linalg.inv(rot_glb)[0],np.transpose([x,y,z]))
		yp = np.dot(np.linalg.inv(rot_glb)[1],np.transpose([x,y,z]))
		zp = np.dot(np.linalg.inv(rot_glb)[2],np.transpose([x,y,z]))
		OC = orb_chara_dic[orb_symbol]
		OC_len = len(OC)
		for i in range(orb_dim):
			e = (orb_function_dic[orb_symbol][i](xp,yp,zp)).expand()
			if orb_symbol == 's':
				orb_rot_mat[0,i] = e.subs(x,0).subs(y,0).subs(z,0).evalf()
			elif orb_symbol == 'p':
				for j in range(orb_dim):
					etmp = e
					orb_rot_mat[j,i] = etmp.subs(OC[j],1).subs(OC[(j+1)%OC_len],0).subs(OC[(j+2)%OC_len],0).evalf()
			elif orb_symbol == 'd':
				subs = []
				for j in range(orb_dim):
					etmp = e
					subs.append(etmp.subs(OC[j],1).subs(OC[(j+1)%OC_len],0).subs(OC[(j+2)%OC_len],0).subs(OC[(j+3)%OC_len],0).subs(OC[(j+4)%OC_len],0).subs(OC[(j+5)%OC_len],0))
				orb_rot_mat[0,i] = (subs[0]*sym.sqrt(3.0)).evalf()
				orb_rot_mat[1,i] = subs[1].evalf()
				orb_rot_mat[2,i] = subs[2].evalf()
				orb_rot_mat[3,i] = (2*subs[3]+subs[0]/sym.sqrt(3.0)).evalf()
				orb_rot_mat[4,i] = subs[4].evalf()
			elif orb_symbol == 'f':
				subs = []
				for j in range(orb_dim):		
					etmp = e
					subs.append(etmp.subs(OC[j],1).subs(OC[(j+1)%OC_len],0).subs(OC[(j+2)%OC_len],0).subs(OC[(j+3)%OC_len],0).subs(OC[(j+4)%OC_len],0).subs(OC[(j+5)%OC_len],0).subs(OC[(j+6)%OC_len],0))
				orb_rot_mat[0,i] = (subs[0]*sym.sqrt(15.0)).evalf()
				orb_rot_mat[1,i] = (subs[1]*sym.sqrt(10.0)/2).evalf()
				orb_rot_mat[2,i] = (subs[2]*sym.sqrt(10.0)/2).evalf()
				orb_rot_mat[3,i] = (2*subs[3]+3*subs[0]/sym.sqrt(15.0)).evalf()
				orb_rot_mat[4,i] = subs[4].evalf()
				orb_rot_mat[5,i] = ((2*subs[5]+subs[1]/sym.sqrt(10.0))*sym.sqrt(6.0)).evalf()
				orb_rot_mat[6,i] = ((-2*subs[6]-subs[2]/sym.sqrt(10.0))*sym.sqrt(6.0)).evalf()

		return np.round(orb_rot_mat,decimals=8)
	
	def Part_P(self,rot_sym,orb_symbol):
		rot_sym_glb = np.dot(np.dot(np.transpose(self.lattice),rot_sym),np.linalg.inv(np.transpose(self.lattice)) )
		rot_sym_glb = np.round(rot_sym_glb,decimals=8)
		if abs(np.dot(np.transpose(rot_sym_glb),rot_sym_glb) - np.eye(3)).sum() >1.0E-4:
			print('rot_sym_glb is not orthogomal \n {}'.format(rot_sym_glb))
		rot_sym_glb[:,1] = np.cross(rot_sym_glb[:,2],rot_sym_glb[:,0]) 
		if self.spin:
			#euler_angle=np.zeros((3),dtype=float)
			beta = np.arccos(rot_sym_glb[2,2])
			if round(abs(rot_sym_glb[2,2]),2) == 1.0:
				gamma = 0.0
				sin_alpha = rot_sym_glb[1,0] / rot_sym_glb[2,2]
				cos_alpha = rot_sym_glb[1,1]
				alpha = self.get_angle(sin_alpha,cos_alpha)
			else:
				sin_gamma = rot_sym_glb[2,1]/np.sin(beta)
				cos_gamma = -rot_sym_glb[2,0]/np.sin(beta)
				gamma = self.get_angle(sin_gamma,cos_gamma)
				sin_alpha = rot_sym_glb[1,2]/np.sin(beta)
				if round(abs(rot_sym_glb[2,2]),2) == 0.0:
					if round(abs(rot_sym_glb[2,1]),2) == 0.0:
						cos_alpha = rot_sym_glb[1,1]/np.cos(gamma)
					else:
						cos_alpha = rot_sym_glb[1,0]/np.sin(gamma)
				else:
					cos_alpha = rot_sym_glb[0,2]/rot_sym_glb[2,2]
				alpha = self.get_angle(sin_alpha,cos_alpha)
			#euler_angle = np.array([alpha,beta,gamma])
			dmat = np.zeros((2,2),dtype=complex)
			dmat[0,0] =  np.exp(-(alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
			dmat[0,1] = -np.exp(-(alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
			dmat[1,0] =  np.exp( (alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
			dmat[1,1] =  np.exp( (alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
			self.dmat=dmat
		#self.rot_sym_glb = rot_sym_glb
		rot_orbital = self.rot_orb(orb_symbol,rot_sym_glb)
		if self.spin:
			rot_orbital = np.kron(rot_orbital,dmat)
		
		return rot_orbital


	def symmetrize(self):
		HH_R=self.HH_R*1.0
		#====Hermitization====
		for ir in range(self.nRvec):
			neg_ir= self.iRvec.index( list(-1*np.array(self.iRvec[ir])) )
			if self.spin:
				HH_R[:,:,ir]=(self.HH_R[:,:,ir] + np.conj(np.transpose(self.HH_R[:,:,neg_ir])) )/2.0
			else:	
				HH_R[:,:,ir]=(self.HH_R[:,:,ir] + np.conj(np.transpose(self.HH_R[:,:,neg_ir])) ).real/2.0
		self.HH_R=HH_R*1.0
		
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
				HH_R[:,:,ir]=hh_R
		self.HH_R=HH_R
		HH_R_copy = self.HH_R*1	
			
		#======add new blocks generated by rotation.
		nRvec=self.nRvec
		flag=np.ones((self.num_wann,self.num_wann,self.nRvec),dtype=int)
		work_rot = 0
		for rot in range(1,self.nsymm):
#		for rot in range(0,2):
			trans = self.symmetry['translations'][rot]
			if (abs(trans[0])+abs(trans[1])+abs(trans[2])) < 1E-6:
				work_rot += 1
				for ir in range(self.nRvec):
					Rvec=np.array(self.iRvec[ir])
					position_atom=np.array(self.positions_in)
					for atomran in range(self.num_atom):
						if len(self.atom_info[atomran][4]) > 0:
							atom_position=position_atom[atomran] + Rvec
							new_atom =np.round( np.dot(self.symmetry['rotations'][rot],atom_position),decimals=6)
							for atom_index in range(self.num_atom):
								old_atom= np.round(np.array(self.positions_in[atom_index]),decimals=6)
								diff = (new_atom-old_atom) - np.array(np.round(new_atom-old_atom,decimals=5),dtype=int)
								if abs(diff[0])+abs(diff[1])+abs(diff[2])<10E-5:
									match_index=atom_index
						#			new_Rvec=list(np.array(np.round((new_atom-old_atom),decimals=6),dtype=int))
								else:
									if atom_index==self.num_atom-1:
										assert atom_index != 0,'Error!!!!: no atom can match the new one Rvec = {}, atom_index = {}'.format(self.iRvec[ir],atom_index)
							new_Rvec=list(np.array(np.round(np.dot(self.symmetry['rotations'][rot],Rvec),decimals=6),dtype=int))
							if new_Rvec in self.iRvec:
								new_Rvec_index = self.iRvec.index(new_Rvec)
							else:
								self.iRvec.append(new_Rvec)
								new_Rvec_index=-1
								new_hh_R=np.zeros((self.num_wann,self.num_wann,1))
								new_flag = np.array(new_hh_R,dtype=int)
								HH_R=np.concatenate((HH_R,new_hh_R),axis=2)
								flag=np.concatenate((flag,new_flag),axis=2)
								nRvec+=1
							orbitals = self.atom_info[match_index][3]
							p_mat = np.zeros((self.num_wann,self.num_wann),dtype = complex)
							wann_index_list = self.atom_info[match_index][4]
							optmp = wann_index_list[0]
							judge = np.zeros((self.num_wann,self.num_wann),dtype = int)
							for orb in orbitals:
								tmp = self.Part_P(self.symmetry['rotations'][rot],orb)
								op=optmp
								sp = 1
								if self.spin: sp = 2
								ed = op+sp*self.orbital_dic[orb]
								optmp = ed
								p_mat[op:ed,op:ed] = tmp
								judge[op:ed,op:ed] += 1
							HH_new_tmp = np.dot(np.dot(np.conj(np.transpose(p_mat)),HH_R_copy[:,:,ir]),p_mat) 
							HH_new = HH_new_tmp*0.0
							select = judge > 0
							HH_new[select] = HH_new_tmp[select]
							HH_R[:,:,new_Rvec_index] += HH_new 
							flag[select,new_Rvec_index] += 1
							
							if new_Rvec == self.iRvec[test1] or new_Rvec == self.iRvec[test2]: 
								print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
								print('symmetry number = {}'.format(rot))
								print(self.iRvec[ir])
								old_cart = np.dot(atom_position,self.lattice)
								print('old_atom_index = {}'.format(atomran))
								print('old_atom_index = {}'.format(old_cart))
								print('dis = {}'.format(old_cart[0]**2+old_cart[1]**2+old_cart[2]**2))
								print(new_Rvec)
								new_cart = np.dot(new_atom,self.lattice)
								print('new_atom_index = {}'.format(match_index))
								print('new_atom_index = {}'.format(new_cart))
								print('dis = {}'.format(new_cart[0]**2+new_cart[1]**2+new_cart[2]**2))
								
								print('++++++++++++++++++++++++++++++++++++++++++++')
								print('old_HH_R')
								print(self.HH_R[st1:st2,st1:st2,ir].real)
								print('new_HH_R')
								print(self.HH_R[st1:st2,:,new_Rvec_index].real)
								print('add_HH_R')
								print(HH_new[st1:st2,st1:st2].real)
								print('New_position_HH_R')
								print(self.HH_R[st1:st2,st1:st2,new_Rvec_index].real)
								print('after add HH_R')
								print(HH_R[st1:st2,st1:st2,new_Rvec_index].real)
								print(flag[st1:st2,st1:st2,new_Rvec_index])
								print('++++++++++++++++++++++++++++++++++++++++++++')
							
		self.nRvec=nRvec
		select=flag > 0
		print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
		print(self.HH_R[st1:st2,st1:st2,test1].real)
		print(self.HH_R[st1:st2,st1:st2,test2].real)
	#	print(self.iRvec.index([-4,1,-1]))
	#	print(self.iRvec.index([4,-1,1]))
		select= flag == 0
		flag[select] = 1
		HH_R_a = np.divide(np.array(HH_R,dtype=complex),np.array(flag,dtype=float))
		self.HH_R = HH_R_a
		
		for ir in range(self.nRvec):
			neg_ir= self.iRvec.index( list(-1*np.array(self.iRvec[ir])) )
			HH_R[:,:,ir]=(self.HH_R[:,:,ir] + np.conj(np.transpose(self.HH_R[:,:,neg_ir])) )/2.0
		print('Working operators = {}'.format(work_rot))
		print(flag[st1:st2,st1:st2,test1])
		print(flag[st1:st2,st1:st2,test2])
		print(self.HH_R[st1:st2,st1:st2,test1].real)
		print(self.HH_R[st1:st2,st1:st2,test2].real)
		print(np.round(self.HH_R[st1:st2,st1:st2,test1] -np.conj( np.transpose(self.HH_R[st1:st2,st1:st2,test2])),decimals=8))
		print(np.round(self.HH_R[:,:,test1] -np.conj( np.transpose(self.HH_R[:,:,test2])),decimals=8))
	def eigHk(self,k):
		k_list=list(k)
		eig=[]
		for kk in k_list:
			Hk=np.zeros((self.num_wann,self.num_wann),dtype=complex)
			for m in range(self.num_wann):
				for n in range(self.num_wann):
					for i in range(self.nRvec):
						R=np.array(self.iRvec[i] + self.wannier_center[n] - self.wannier_center[m])
						RR=np.dot(kk,R)
						ratio=np.exp(1j*2*np.pi*RR)
						Hk[m,n] = Hk[m,n] + ratio*self.HH_R[m,n,i]
			eig.append(np.array(la.eigvalsh(Hk).real))
		return np.array(eig)








