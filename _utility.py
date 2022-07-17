

def read_tb(self):
    f=open(self.seedname+"_tb.dat","r")
    f.readline()
    self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
    self.num_wann=int(f.readline())
    self.nRvec=int(f.readline())
    self.Ndegen=[]
    while len(self.Ndegen)<self.nRvec:
        self.Ndegen+=f.readline().split()
    self.Ndegen=np.array(self.Ndegen,dtype=int)
    self.iRvec=[]
    self.HH_R=np.zeros( (self.num_wann,self.num_wann,self.nRvec) ,dtype=complex)
    for ir in range(self.nRvec):
        f.readline()
        self.iRvec.append(list(np.array(f.readline().split()[0:3],dtype=int)))
        hh=np.array( [[f.readline().split()[2:4]
                for n in range(self.num_wann)]
                    for m in range(self.num_wann)],dtype=float).transpose( (1,0,2) )
        self.HH_R[:,:,ir]=(hh[:,:,0]+1j*hh[:,:,1])/self.Ndegen[ir]
    self.AA_R=np.zeros( (self.num_wann,self.num_wann,self.nRvec,3) ,dtype=complex)
    for ir in range(self.nRvec):
        f.readline()
        assert (np.array(f.readline().split(),dtype=int)==np.array(self.iRvec[ir],dtype=int)).all()
        aa=np.array( [[f.readline().split()[2:8]
                    for n in range(self.num_wann)]
                    for m in range(self.num_wann)],dtype=float)
        self.AA_R[:,:,ir,:]=(aa[:,:,0::2]+1j*aa[:,:,1::2]).transpose( (1,0,2) )/self.Ndegen[ir]
    print('HH',np.shape(self.HH_R),'AA',np.shape(self.AA_R))


def write_hr(self):
    name=self.seedname+"_sym_hr.dat"
    ndegen=list(np.ones((self.nRvec),dtype=int))
    with open(name,"w") as f:
        f.write("symmetrize wannier hr\n"+str(self.num_wann)+"\n"+str(self.nRvec)+"\n")
        nl = np.int32(np.ceil(self.nRvec/15.0))
        for l in range(nl):
            line="    "+'    '.join([str(np.int32(i)) for i in ndegen[l*15:(l+1)*15]])
            f.write(line+"\n")
        whh_r = np.round(self.HH_R,decimals=6)
        for ir in range(self.nRvec):
            rx = int(self.iRvec[ir][0]);ry = int(self.iRvec[ir][1]);rz = int(self.iRvec[ir][2])
            for n in range(self.num_wann):
                for m in range(self.num_wann):
                    rp =whh_r[m,n,ir].real
                    ip =whh_r[m,n,ir].imag
                    line="{:5d}{:5d}{:5d}{:5d}{:5d}{:12.6f}{:12.6f}\n".format(rx,ry,rz,m+1,n+1,rp,ip)
                    f.write(line)
        f.close()

def write_tb(self):
    name=self.seedname+"_sym_tb.dat"
    ndegen=list(np.ones((self.nRvec),dtype=int))
    with open(name,"w") as f:
        f.write("symmetrize wannier tb\n"
                +"{:12.6f}{:12.6f}{:12.6f}\n".format(self.lattice[0,0],self.lattice[0,1],self.lattice[0,2])
                +"{:12.6f}{:12.6f}{:12.6f}\n".format(self.lattice[1,0],self.lattice[1,1],self.lattice[1,2])
                +"{:12.6f}{:12.6f}{:12.6f}\n".format(self.lattice[2,0],self.lattice[2,1],self.lattice[2,2])
                +str(self.num_wann)+"\n"+str(self.nRvec)+"\n")
        nl = np.int32(np.ceil(self.nRvec/15.0))
        for l in range(nl):
            line="    "+'    '.join([str(np.int32(i)) for i in ndegen[l*15:(l+1)*15]])
            f.write(line+"\n")
        whh_r = np.round(self.HH_R,decimals=10)
        waa_r = np.round(self.AA_R,decimals=10)
        for ir in range(self.nRvec):
            rx = int(self.iRvec[ir][0]);ry = int(self.iRvec[ir][1]);rz = int(self.iRvec[ir][2])
            f.write("\n{:5d}{:5d}{:5d}\n".format(rx,ry,rz))
            for n in range(self.num_wann):
                for m in range(self.num_wann):
                    rp =whh_r[m,n,ir].real
                    ip =whh_r[m,n,ir].imag
                    line="{:5d}{:5d}{:17.8E}{:17.8E}\n".format(m+1,n+1,rp,ip)
                    f.write(line)
        for ir in range(self.nRvec):
            rx = int(self.iRvec[ir][0]);ry = int(self.iRvec[ir][1]);rz = int(self.iRvec[ir][2])
            f.write("\n{:5d}{:5d}{:5d}\n".format(rx,ry,rz))
            for n in range(self.num_wann):
                for m in range(self.num_wann):
                    rp =waa_r[m,n,ir].real
                    ip =waa_r[m,n,ir].imag
                    line="{:5d}{:5d}{:18.8E}{:18.8E}{:18.8E}{:18.8E}{:18.8E}{:18.8E}\n".format(m+1,n+1,rp[0],ip[0],rp[1],ip[1],rp[2],ip[2])
                    f.write(line)
        f.close()







def get_angle(sina, cosa):
    '''Get angle in radian from sin and cos.'''
    if abs(cosa) > 1.0:
        cosa = np.round(cosa, decimals=1)
    alpha = np.arccos(cosa)
    if sina < 0.0:
        alpha = 2.0 * np.pi - alpha
    return alpha
