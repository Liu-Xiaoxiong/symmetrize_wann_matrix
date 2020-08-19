import numpy as np

class sym_wann():
	def __init__(self,seedname="wannier90",spin=False,TR=True):
		self.seedname=seedname
		self.spin=spin
		self.TR=TR
		self.lattice=np.array([[3.658341000000000,0.0000000000000000,0.0000000000000000],[-1.829171000000000,3.168216000000000,0.0000000000000000],[0.0000000000000000,0.0000000000000000,4.7614159999999996]])

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

		self.HH_R=HH_R
