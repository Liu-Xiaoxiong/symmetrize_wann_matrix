import numpy as np
import spglib
import numpy.linalg as la
from atoms import Atoms
import sympy as sym
np.set_printoptions(threshold=np.inf,linewidth=500)


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
				rx = int(self.iRvec[ir][0]);ry = int(self.iRvec[ir][1]);rz = int(self.iRvec[ir][2])
				for n in range(self.num_wann):
					for m in range(self.num_wann):
						rp =self.HH_R[m,n,ir].real
						ip =self.HH_R[m,n,ir].imag
						line="{:5d}{:5d}{:5d}{:5d}{:5d}{:16.8f}{:16.8f}\n".format(rx,ry,rz,m+1,n+1,rp,ip)
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
			self.positions_in.append(list( np.round(np.dot(position,np.linalg.inv(self.lattice)),decimals=8) ))			
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

		atom_in=Atoms(symbols=self.symbols_in,cell=list(self.lattice),scaled_positions=self.positions_in,pbc=True)
		print("[get_spacegroup]")
		print("  Spacegroup of "+self.seedname+" is %s." %spglib.get_spacegroup(atom_in))
		self.symmetry = spglib.get_symmetry(atom_in)
		self.nsymm = self.symmetry['rotations'].shape[0]
		show_symmetry(self.symmetry)
		'''
		find a mathod can reduce operators to generators
		'''
	def get_angle(self,sina,cosa):
		if cosa > 1.0:
			cosa =1.0
		elif cosa < -1.0:
			cosa = -1.0
		alpha = np.arccos(cosa)
		if sina < 0.0:
			alpha = 2.0 * np.pi - alpha
		return alpha

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
		rot_glb=np.array(list(rot_glb))
		#print('inv')
		#print(np.linalg.inv(rot_glb))
		#print('xp,yp,zp')
		#print(xp,yp,zp)
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
				orb_rot_mat[3,i] = (2*subs[3]+subs[0]).evalf()
				orb_rot_mat[4,i] = subs[4].evalf()
			elif orb_symbol == 'f':
				subs = []
				for j in range(orb_dim):		
					etmp = e
					subs.append(etmp.subs(OC[j],1).subs(OC[(j+1)%OC_len],0).subs(OC[(j+2)%OC_len],0).subs(OC[(j+3)%OC_len],0).subs(OC[(j+4)%OC_len],0).subs(OC[(j+5)%OC_len],0).subs(OC[(j+6)%OC_len],0))
				orb_rot_mat[0,i] = (subs[0]*sym.sqrt(15.0)).evalf()
				orb_rot_mat[1,i] = (subs[1]*sym.sqrt(10.0)/2).evalf()
				orb_rot_mat[2,i] = (subs[2]*sym.sqrt(10.0)/2).evalf()
				orb_rot_mat[3,i] = (2*subs[3]+3*subs[0]).evalf()
				orb_rot_mat[4,i] = subs[4].evalf()
				orb_rot_mat[5,i] = ((2*subs[5]+subs[1]/2)*sym.sqrt(6.0)).evalf()
				orb_rot_mat[6,i] = ((-2*subs[6]-subs[2]/2)*sym.sqrt(6.0)).evalf()
		return np.round(orb_rot_mat,decimals=8)
	
	def Part_P(self,rot_sym,orb_symbol):
		#print('rot')
		#print(rot_sym)
		rot_sym_glb = np.dot(np.dot(np.transpose(self.lattice),rot_sym),np.linalg.inv(np.transpose(self.lattice)) )
		rot_sym_glb = np.round(rot_sym_glb,decimals=8)
		print('rot_glb')
		print(rot_sym_glb)
		if abs(np.dot(np.transpose(rot_sym_glb),rot_sym_glb) - np.eye(3)).sum() >1.0E-4:
			print('rot_sym_glb is not orthogomal \n {}'.format(rot_sym_glb))
		rmat = np.linalg.det(rot_sym_glb)*rot_sym_glb 
		if self.spin:
			if np.abs(rmat[2,2]) < 1.0:
				beta = np.arccos(rmat[2,2])
				cos_gamma = -rmat[2,0] / np.sin(beta)
				sin_gamma =  rmat[2,1] / np.sin(beta)
				gamma = where_is_angle(sin_gamma, cos_gamma)
				cos_alpha = rmat[0,2] / np.sin(beta)
				sin_alpha = rmat[1,2] / np.sin(beta)
				alpha = get_angle(sin_alpha, cos_alpha)
			else:
				if rmat[2,2] > 0: # cos(beta) = 1, beta = 0, sin(beta/2)=0.0
					beta = 0.0
					gamma = 0.0
					alpha = np.arccos(rmat[1,1])
					if   -rmat[0,1] < 0.0:
						alpha = -1.0*alpha
				else:             # cos(beta) =-1, beta =pi, sin(beta/2)=1.0
					beta = np.pi
					gamma = 0.0
					alpha = np.arccos(rmat[1,1]) # 0~pi pi/2 if rmat[0,1] 
					if -rmat[0,1] < 0.0: alpha = -1.0*alpha
			euler_angle = np.array([alpha,beta,gamma])
			dmat = np.zeros((2,2),dtype=complex)
			dmat[0,0] =  np.exp(-(alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
			dmat[0,1] = -np.exp(-(alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
			dmat[1,0] =  np.exp( (alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
			dmat[1,1] =  np.exp( (alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
			self.dmat=dmat
		rot_orbital = self.rot_orb(orb_symbol,rot_sym_glb)
		if self.spin:
			print(dmat)
			rot_orbital = np.kron(rot_orbital,dmat)
			#rot_orbital = np.kron(rot_orbital,np.array([[1.,0.],[0.,-1.]]))
		select = abs(rot_orbital)<1E-5
		rot_orbital[select]=0.0
		return rot_orbital

	def rot_orb_test(self):
		for rot in range(self.nsymm):
			print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
			print('rotation matrix = ')
			print(self.symmetry['rotations'][rot])
			orb_p = self.Part_P(self.symmetry['rotations'][rot],'p')
			orb_s = self.Part_P(self.symmetry['rotations'][rot],'s')
			orb_d = self.Part_P(self.symmetry['rotations'][rot],'d')
			print('rot_orb_s')
			print(orb_s)
			print('rot_orb_p')
			print(orb_p)
			print('rot_orb_d')
			print(orb_d)

	
	def atom_rot_map(self,sym):
		wann_atom_info = []
		num_wann_atom = 0
		for atom in range(self.num_atom):
			if len(self.atom_info[atom][4]) > 0:
				num_wann_atom +=1
				wann_atom_info.append(self.atom_info[atom])
		self.num_wann_atom = num_wann_atom
		self.wann_atom_info = wann_atom_info
		wann_atom_positions = [wann_atom_info[i][2] for i in range(self.num_wann_atom)]
		rot_map=[]
		for atomran in range(self.num_wann_atom):
			atom_position=np.array(wann_atom_positions[atomran])
			new_atom =np.round( np.dot(self.symmetry['rotations'][sym],atom_position) + self.symmetry['translations'][sym],decimals=8)
			for atom_index in range(self.num_wann_atom):
				old_atom= np.round(np.array(wann_atom_positions[atom_index]),decimals=8)
				diff = np.array(np.round(new_atom-old_atom,decimals=8))
				if abs(diff[0]%1)+abs(diff[1]%1)+abs(diff[2]%1)<10E-5:
					match_index=atom_index
					vec_shift=new_atom-np.array(wann_atom_positions[match_index])
				else:
					if atom_index==self.num_atom-1:
						assert atom_index != 0,'Error!!!!: no atom can match the new one Rvec = {}, atom_index = {}'.format(self.iRvec[ir],atom_index)
			print('old',atom_position)
			print('new',new_atom)
			print('match',np.array(wann_atom_positions[match_index]))
			print(vec_shift)
			rot_map.append((match_index,vec_shift))
		return rot_map

	def test_atom_map(self):
		for i in range(self.nsymm):
			print('######################################')
			print(self.symmetry['rotations'][i])
			rot_map = self.atom_rot_map(i)
			print(rot_map)
	def full_p_mat(self,match_index,rot):
		orbitals = self.wann_atom_info[match_index][3]
		p_mat = np.zeros((self.num_wann,self.num_wann),dtype = complex)
		wann_index_list = self.wann_atom_info[match_index][4]
		optmp = wann_index_list[0]
		for orb in orbitals:
			tmp = self.Part_P(self.symmetry['rotations'][rot],orb)
			op=optmp
			sp = 1
			if self.spin: sp = 2
			ed = op+sp*self.orbital_dic[orb]
			optmp = ed
			p_mat[op:ed,op:ed] = tmp
		a = wann_index_list[0]
		b = wann_index_list[-1]+1 
		return p_mat[a:b,a:b],a,b
	
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
		'''
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
		'''
		HH_R_copy = self.HH_R*1		
		#======add new blocks generated by rotation.
		nRvec=self.nRvec
		flag=np.ones((self.num_wann,self.num_wann,self.nRvec),dtype=int)
	#	for ir in range(1,4):
		for ir in range(0,self.nRvec):
			print('>>>>>>>>>>>>>>>>>>>>>>>>>{}'.format(self.iRvec[ir]))
			for rot in range(0,self.nsymm):
				print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ sym={}'.format(rot))
				rot_map = self.atom_rot_map(rot)
				Rvec=np.array(self.iRvec[ir])
				new_Rvec_tmp=np.array(np.round(np.dot(self.symmetry['rotations'][rot],Rvec),decimals=8),dtype=int)
				for atomran in range(self.num_wann_atom):
					for atomran_0 in [rot_map[atomran][0]]:	
						print('rot')
						print(self.symmetry['rotations'][rot])
						rotglb = np.dot(np.dot(np.transpose(self.lattice),self.symmetry['rotations'][rot] ),np.linalg.inv(np.transpose(self.lattice)) )
						new_Rvec = list(np.dot(self.symmetry['rotations'][rot],self.iRvec[ir]) - rot_map[atomran_0][1]  + rot_map[atomran][1])
						new_Rvec = [int(round(new_Rvec[0])),int(round(new_Rvec[1])),int(round(new_Rvec[2]))]
						print(new_Rvec)
						if new_Rvec in self.iRvec:
							new_Rvec_index = self.iRvec.index(new_Rvec)
							p_mat_a ,ao,ae = self.full_p_mat(atomran,rot)
							p_mat_b ,bo,be= self.full_p_mat(atomran_0,rot)
							apo = self.wann_atom_info[rot_map[atomran][0]][4][0]		
							ape = self.wann_atom_info[rot_map[atomran][0]][4][-1]+1		
							bpo = self.wann_atom_info[rot_map[atomran_0][0]][4][0]		
							bpe = self.wann_atom_info[rot_map[atomran_0][0]][4][-1]+1		
							HH_new_tmp = np.round(np.dot(np.dot(np.conj(np.transpose(p_mat_b)),HH_R_copy[bpo:bpe,apo:ape,new_Rvec_index]),p_mat_a),decimals=6)
							print('??????')							
							print(p_mat_b)
							print(HH_new_tmp)
							print(HH_R_copy[bo:be,ao:ae,ir])
							new_Rvec_index= self.iRvec.index(new_Rvec)	
							print(HH_R_copy[bpo:bpe,apo:ape,new_Rvec_index])
							print('??????')
							HH_R[bo:be,ao:ae,ir] += HH_new_tmp
							flag[bo:be,ao:ae,ir] += 1	
		self.nRvec=nRvec
		select=flag > 0
		select= flag == 0
		flag[select] = 1
		HH_R_a = np.round(np.divide(np.array(HH_R,dtype=complex),np.array(flag,dtype=float)),decimals=8)
		for i in range(self.nRvec):
			neg_i= self.iRvec.index( list(-1*np.array(self.iRvec[i])) )
			HH_R_a[:,:,i] = np.round( HH_R_a[:,:,i] - (HH_R_a[:,:,i] - np.conj(np.transpose(HH_R_a[:,:,neg_i]))),decimals=6 )
		self.HH_R = HH_R_a
		def test_H(ll):
			a = self.iRvec.index(ll)
			b = self.iRvec.index(list(-1*np.array(ll)))
			print(np.round(self.HH_R[:,:,a] -np.conj( np.transpose(self.HH_R[:,:,b])),decimals=8))
		
#		test_H([0,0,0])
#		test_H([4,2,4])
#		test_H([4,3,5])








