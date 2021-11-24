import numpy as np
import spglib
import numpy.linalg as la
from atoms import Atoms
import sympy as sym
#np.set_printoptions(threshold=np.inf,linewidth=500)
np.set_printoptions(suppress=True,precision=4,threshold=np.inf,linewidth=500)


class sym_wann():
    def __init__(self,seedname="wannier90",spin=False,TR=False,magmom=None):
        self.seedname=seedname
        self.spin=spin
        self.TR=TR
        self.magmom=magmom
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
            #if "spinors" in line:
            #   if "t" or "T" in line:
            #        self.spin=True
            #    else:
            #        self.spin=False 
        self.lattice = np.array([line.split() for line in win[lattice_op:lattice_ed]],dtype=float)
        self.orbital_dic = {"s":1,"p":3,"d":5,"f":7,"sp3":4,"sp2":3,"sp":2,"sp3d":5,"sp3d2":6,
                            "px":1,"px":1,"pz":1}	
        projectiondic={}
        self.wann_atom_info = []
        self.symbols_in=[]
        self.positions_in=[]
        num_wann_atom = 0
        for natom in range(atom_op,atom_ed):
            atom_name = win[natom].split()[0]
            position = np.array(win[natom].split()[1:],dtype=float)
            self.symbols_in.append(atom_name)
            self.positions_in.append(list( np.round(np.dot(position,np.linalg.inv(self.lattice)),decimals=8) ))			
        self.num_atom = len(self.symbols_in)
        if self.spin: orb_spin = 2
        else: orb_spin = 1
        orbital_index_list=[]
        orb_op=[]
        orb_ed=[]
        for i in range(self.num_atom):
            orbital_index_list.append([])
            orb_op.append([])
            orb_ed.append([])
        orbital_index=0
        for npro in range(pro_op,pro_ed):
            name = win[npro].split(":")[0].split()[0]
            orb = win[npro].split(":")[1].strip('\n').strip().split(';')
            print(orb)
            if name in projectiondic.keys():
                projectiondic[name]=projectiondic[name]+orb			
            else:
                newdic={name:orb}
                projectiondic.update(newdic)
            print(projectiondic)
            for atom in range(self.num_atom):
                if self.symbols_in[atom] == name:
                    for orb_name in orb:
                        num_orb = orb_spin * self.orbital_dic[orb_name]
                        orbital_index_old = orbital_index
                        orbital_index+=num_orb
                        orb_op[atom] += [orbital_index_old]
                        orb_ed[atom] += [orbital_index]
                        orbital_index_list[atom] += [ i for i in range(orbital_index_old,orbital_index)]
        for atom in range(self.num_atom):
            name = self.symbols_in[atom]
            if name in projectiondic.keys():
                projection=projectiondic[name]	
                num_wann_atom +=1
                if self.magmom is None:
                    self.wann_atom_info.append((atom+1,self.symbols_in[atom],self.positions_in[atom],projection,orbital_index_list[atom],orb_op[atom],orb_ed[atom]))
                else:
                    self.wann_atom_info.append((atom+1,self.symbols_in[atom],self.positions_in[atom],projection,orbital_index_list[atom],orb_op[atom],orb_ed[atom],self.magmom[atom]))
        self.num_wann_atom = num_wann_atom
        self.H_select=np.zeros((self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann),dtype=bool)
        for atom_a in range(self.num_wann_atom):
            for atom_b in range(self.num_wann_atom):
                orb_name_a = self.wann_atom_info[atom_a][3]
                orb_name_b = self.wann_atom_info[atom_b][3]
                orb_op_a = self.wann_atom_info[atom_a][5]
                orb_op_b = self.wann_atom_info[atom_b][5]
                orb_ed_a = self.wann_atom_info[atom_a][6]
                orb_ed_b = self.wann_atom_info[atom_b][6]
                for oa in range(len(orb_name_a)):
                    for ob in range(len(orb_name_b)):
                        self.H_select[atom_a,atom_b,orb_op_a[oa]:orb_ed_a[oa],orb_op_b[ob]:orb_ed_b[ob]]=True
        print('Wannier atoms info')
        for i in range(self.num_wann_atom):
            print(self.wann_atom_info[i])
			
	
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
    

    def get_angle(self,sina,cosa):
        if abs(cosa) > 1.0:
            cosa = np.round(cosa,decimals=1)
        alpha = np.arccos(cosa)
        if sina < 0.0:
            alpha = 2.0 * np.pi - alpha
        return alpha

    def rot_orb(self,orb_symbol,rot_glb):
        x = sym.Symbol('x')
        y = sym.Symbol('y')
        z = sym.Symbol('z')
        def ss(x,y,z): return 1+0*x
        def pz(x,y,z): return z
        def px(x,y,z): return x
        def py(x,y,z): return y
        def dz2(x,y,z): return (2*z*z-x*x-y*y)/(2*sym.sqrt(3.0))
        def dxz(x,y,z): return x*z
        def dyz(x,y,z): return y*z
        def dx2_y2(x,y,z): return (x*x-y*y)/2
        def dxy(x,y,z): return x*y
        def fz3(x,y,z): return z*(2*z*z-3*x*x-3*y*y)/(2*sym.sqrt(15.0))
        def fxz2(x,y,z): return x*(4*z*z-x*x-y*y)/(2*sym.sqrt(10.0))
        def fyz2(x,y,z): return y*(4*z*z-x*x-y*y)/(2*sym.sqrt(10.0))
        def fzx2_zy2(x,y,z): return z*(x*x-y*y)/2
        def fxyz(x,y,z): return x*y*z
        def fx3_3xy2(x,y,z): return x*(x*x-3*y*y)/(2*sym.sqrt(6.0))
        def f3yx2_y3(x,y,z): return y*(3*x*x-y*y)/(2*sym.sqrt(6.0))
       
        def sp_1(x,y,z): return 1/sym.sqrt(2) * (1 + x)
        def sp_2(x,y,z): return 1/sym.sqrt(2) * (1 - x)

        def sp2_1(x,y,z): return 1/sym.sqrt(3) - 1/sym.sqrt(6)*x + 1/sym.sqrt(2)*y
        def sp2_2(x,y,z): return 1/sym.sqrt(3) - 1/sym.sqrt(6)*x - 1/sym.sqrt(2)*y
        def sp2_3(x,y,z): return 1/sym.sqrt(3) + 2/sym.sqrt(6)*x 

        def sp3_1(x,y,z): return 0.5*(1 + x + y + z) 
        def sp3_2(x,y,z): return 0.5*(1 + x - y - z) 
        def sp3_3(x,y,z): return 0.5*(1 - x + y - z) 
        def sp3_4(x,y,z): return 0.5*(1 - x - y + z) 

        def sp3d_1(x,y,z): return 1/sqrt(3)*ss -1/sqrt(6)*px +1/sqrt(2)*py
        def sp3d_2(x,y,z): return 1/sqrt(3)*ss -1/sqrt(6)*px -1/sqrt(2)*py
        def sp3d_3(x,y,z): return 1/sqrt(3)*ss +2/sqrt(6)*px
        def sp3d_4(x,y,z): return 1/sqrt(2)*(pz + dz2) 
        def sp3d_5(x,y,z): return 1/sqrt(2)*(pz + dz2) 
        
        def sp3d2_1(x,y,z): return 1/sqrt(6)*ss -1/sqrt(2)*px - 1/sqrt(12)*dz2 + 0.5*dx2_y2
        def sp3d2_2(x,y,z): return 1/sqrt(6)*ss +1/sqrt(2)*px - 1/sqrt(12)*dz2 + 0.5*dx2_y2
        def sp3d2_3(x,y,z): return 1/sqrt(6)*ss -1/sqrt(2)*py - 1/sqrt(12)*dz2 - 0.5*dx2_y2
        def sp3d2_4(x,y,z): return 1/sqrt(6)*ss +1/sqrt(2)*py - 1/sqrt(12)*dz2 - 0.5*dx2_y2
        def sp3d2_5(x,y,z): return 1/sqrt(6)*ss -1/sqrt(2)*pz + 1/sqrt(3)*dz2
        def sp3d2_6(x,y,z): return 1/sqrt(6)*ss +1/sqrt(2)*pz + 1/sqrt(3)*dz2

        orb_function_dic={'s':[ss],
                          'p': [pz,px,py],
                          'd':[dz2,dxz,dyz,dx2_y2,dxy],
                          'f': [fz3,fxz2,fyz2,fzx2_zy2,fxyz,fx3_3xy2,f3yx2_y3],
                         'sp':[sp_1,sp_2],
                        'sp2':[sp2_1,sp2_2,sp2_3],
                        'sp3':[sp3_1,sp3_2,sp3_3,sp3_4],
                       'sp3d':[sp3d_1,sp3d_2,sp3d_3,sp3d_4,sp3d_5],
                      'sp3d2':[sp3d2_1,sp3d2_2,sp3d2_3,sp3d2_4,sp3d2_5,sp3d2_6],
                         'px':[x],
                         'py':[y],
                         'pz':[z],
                      }
        orb_chara_dic={'s':[x],'p':[z,x,y],'d':[z*z,x*z,y*z,x*x,x*y,y*y],
                'f':[z*z*z,x*z*z,y*z*z,z*x*x,x*y*z,x*x*x,y*y*y  ,z*y*y,x*y*y,y*x*x],
                'sp':[x],'sp2':[x,y],'sp3':[x,y,z],'sp3d2':[x,y,z,z*z,x*x,y*y],
                'px':[x],'py':[y],'pz':[z]
                }
        orb_dim = self.orbital_dic[orb_symbol]
        orb_rot_mat = np.zeros((orb_dim,orb_dim),dtype=float)
        xp = np.dot(np.linalg.inv(rot_glb)[0],np.transpose([x,y,z]))
        yp = np.dot(np.linalg.inv(rot_glb)[1],np.transpose([x,y,z]))
        zp = np.dot(np.linalg.inv(rot_glb)[2],np.transpose([x,y,z]))
        rot_glb=np.array(list(rot_glb))
        OC = orb_chara_dic[orb_symbol]
        OC_len = len(OC)
        for i in range(orb_dim):
            subs = []
            equation = (orb_function_dic[orb_symbol][i](xp,yp,zp)).expand()
            #if orb_symbol != 's':
            for j in range(OC_len):
                    for j_add in range(OC_len):
                        if j_add == 0:
                            eq_tmp = equation.subs(OC[j],1)
                        else:
                            eq_tmp = eq_tmp.subs(OC[(j+j_add)%OC_len],0)
                    subs.append(eq_tmp)
            if orb_symbol == 's':
                orb_rot_mat[0,0] = subs[0].evalf()
            elif orb_symbol == 'p':
                orb_rot_mat[0,i] = subs[0].evalf()
                orb_rot_mat[1,i] = subs[1].evalf()
                orb_rot_mat[2,i] = subs[2].evalf()
            elif orb_symbol == 'd':
                orb_rot_mat[0,i] = (subs[0]*sym.sqrt(3)).evalf()
                orb_rot_mat[1,i] = subs[1].evalf()
                orb_rot_mat[2,i] = subs[2].evalf()
                orb_rot_mat[3,i] = (2*subs[3]+subs[0]).evalf()
                orb_rot_mat[4,i] = subs[4].evalf()
            elif orb_symbol == 'f':
                orb_rot_mat[0,i] = (subs[0]*sym.sqrt(15.0)).evalf()
                orb_rot_mat[1,i] = (subs[1]*sym.sqrt(10.0)/2).evalf()
                orb_rot_mat[2,i] = (subs[2]*sym.sqrt(10.0)/2).evalf()
                orb_rot_mat[3,i] = (2*subs[3]+3*subs[0]).evalf()
                orb_rot_mat[4,i] = subs[4].evalf()
                orb_rot_mat[5,i] = ((2*subs[5]+subs[1]/2)*sym.sqrt(6.0)).evalf()
                orb_rot_mat[6,i] = ((-2*subs[6]-subs[2]/2)*sym.sqrt(6.0)).evalf()
            elif orb_symbol == 'sp':
                orb_rot_mat[0,i] = 1/sym.sqrt(2)*(1+subs[0]) 
                orb_rot_mat[1,i] = 1/sym.sqrt(2)*(1-subs[0])
            elif orb_symbol == 'sp2':
                orb_rot_mat[0,i] = 1/sym.sqrt(3)-1/sym.sqrt(6)*sub[0]+1/sym.sqrt(2)*sub[1]
                orb_rot_mat[1,i] = 1/sym.sqrt(3)-1/sym.sqrt(6)*sub[0]-1/sym.sqrt(2)*sub[1]
                orb_rot_mat[2,i] = 1/sym.sqrt(3)+2/sym.sqrt(6)*sub[0]
            elif orb_symbol == 'sp3':
                orb_rot_mat[0,i] = 0.5*(subs[0]+subs[1]+subs[2] - 1)
                orb_rot_mat[1,i] = 0.5*(subs[0]-subs[1]-subs[2] + 1)
                orb_rot_mat[2,i] = 0.5*(subs[1]-subs[0]-subs[2] + 1)
                orb_rot_mat[3,i] = 0.5*(subs[2]-subs[1]-subs[0] + 1)
            #elif orb_symbol == 'sp3d':
            #elif orb_symbol == 'sp3d2':

       # print(orb_rot_mat)
        return orb_rot_mat
	
    def Part_P(self,rot_sym_glb,orb_symbol):
        if abs(np.dot(np.transpose(rot_sym_glb),rot_sym_glb) - np.eye(3)).sum() >1.0E-4:
            print('rot_sym is not orthogomal \n {}'.format(rot_sym_glb))
        rmat = np.linalg.det(rot_sym_glb)*rot_sym_glb
        select = np.abs(rmat) < 0.01
        rmat[select] = 0.0 
        select = rmat > 0.99 
        rmat[select] = 1.0 
        select = rmat < -0.99 
        rmat[select] = -1.0 
        if self.spin:
            if np.abs(rmat[2,2]) < 1.0:
                beta = np.arccos(rmat[2,2])
                cos_gamma = -rmat[2,0] / np.sin(beta)
                sin_gamma =  rmat[2,1] / np.sin(beta)
                gamma = self.get_angle(sin_gamma, cos_gamma)
                cos_alpha = rmat[0,2] / np.sin(beta)
                sin_alpha = rmat[1,2] / np.sin(beta)
                alpha = self.get_angle(sin_alpha, cos_alpha)
            else:
                beta = 0.0
                if rmat[2,2] == -1. :beta = np.pi
                gamma = 0.0
                alpha = np.arccos(rmat[1,1])
                if rmat[0,1] > 0.0:alpha = -1.0*alpha
            euler_angle = np.array([alpha,beta,gamma])
            dmat = np.zeros((2,2),dtype=complex)
            dmat[0,0] =  np.exp(-(alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
            dmat[0,1] = -np.exp(-(alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
            dmat[1,0] =  np.exp( (alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
            dmat[1,1] =  np.exp( (alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
        rot_orbital = self.rot_orb(orb_symbol,rot_sym_glb)
        if self.spin:
            rot_orbital = np.kron(rot_orbital,dmat)
            rot_imag = rot_orbital.imag
            rot_real = rot_orbital.real
            rot_imag[abs(rot_imag) < 10e-6] = 0
            rot_real[abs(rot_real) < 10e-6] = 0
            rot_orbital = np.array(rot_real + 1j*rot_imag,dtype=complex)
        return rot_orbital

    def atom_rot_map(self,sym):
        wann_atom_positions = [self.wann_atom_info[i][2] for i in range(self.num_wann_atom)]
        rot_map=[]
        vec_shift_map=[]
        for atomran in range(self.num_wann_atom):
            atom_position=np.array(wann_atom_positions[atomran])
            new_atom =np.round( np.dot(self.symmetry['rotations'][sym],atom_position) + self.symmetry['translations'][sym],decimals=5)
            for atom_index in range(self.num_wann_atom):
                old_atom= np.round(np.array(wann_atom_positions[atom_index]),decimals=5)
                diff = np.array(np.round(new_atom-old_atom,decimals=8))
                if abs(diff[0]%1)+abs(diff[1]%1)+abs(diff[2]%1)<10E-5:
                    match_index=atom_index
                    vec_shift= np.array(np.round(new_atom-np.array(wann_atom_positions[match_index]),decimals=2),dtype=int)
                else:
                    if atom_index==self.num_wann_atom-1:
                        assert atom_index != 0,'Error!!!!: no atom can match the new one Rvec = {}, atom_index = {}'.format(self.iRvec[ir],atom_index)
            rot_map.append(match_index)
            vec_shift_map.append(vec_shift)
        rot_sym = self.symmetry['rotations'][sym]
        rot_sym_glb = np.dot(np.dot(np.transpose(self.lattice),rot_sym),np.linalg.inv(np.transpose(self.lattice)) )
        if self.magmom is not None:
            for i in range(self.num_wann_atom):
                magmom = np.round(self.wann_atom_info[i][-1],decimals=4)
                new_magmom =np.round( np.dot(rot_sym_glb,magmom),decimals=4)
                if abs(np.linalg.norm(magmom-new_magmom)) > 0.0005:
                    magsym_break = True
                    print('Symmetry operator {} is not respect to magnetic moment'.format(sym) )
                    break
                else:
                    magsym_break = False
        else:
            magsym_break=False
        return np.array(rot_map,dtype=int),np.array(vec_shift_map,dtype=int),magsym_break

    def full_p_mat(self,atom_index,rot):
        orbitals = self.wann_atom_info[atom_index][3]
        op = self.wann_atom_info[atom_index][5]
        ed = self.wann_atom_info[atom_index][6]
        p_mat = np.zeros((self.num_wann,self.num_wann),dtype = complex)
        p_mat_dagger = np.zeros((self.num_wann,self.num_wann),dtype = complex)
        rot_sym = self.symmetry['rotations'][rot]
        rot_sym_glb = np.dot(np.dot(np.transpose(self.lattice),rot_sym),np.linalg.inv(np.transpose(self.lattice)) )
        for orb in range(len(orbitals)):
            orb_name = orbitals[orb]
            tmp = self.Part_P(rot_sym_glb,orb_name)
            p_mat[op[orb]:ed[orb],op[orb]:ed[orb]] = tmp 	
            p_mat_dagger[op[orb]:ed[orb],op[orb]:ed[orb]] = np.conj(np.transpose(tmp)) 	
        return p_mat,p_mat_dagger

    def spin_range(self,Mat_in,inv=False):        
        Mat_out = np.zeros(np.shape(Mat_in),dtype=complex)
        if inv:
            Mat_out[0:self.num_wann//2,0:self.num_wann//2,:]                         = Mat_in[0:self.num_wann:2,0:self.num_wann:2,:]
            Mat_out[self.num_wann//2:self.num_wann,0:self.num_wann//2,:]             = Mat_in[1:self.num_wann:2,0:self.num_wann:2,:]
            Mat_out[0:self.num_wann//2,self.num_wann//2:self.num_wann,:]             = Mat_in[0:self.num_wann:2,1:self.num_wann:2,:]
            Mat_out[self.num_wann//2:self.num_wann,self.num_wann//2:self.num_wann,:] = Mat_in[1:self.num_wann:2,1:self.num_wann:2,:]
            return Mat_out
        else:
            Mat_out[0:self.num_wann:2,0:self.num_wann:2,:] = Mat_in[0:self.num_wann//2,0:self.num_wann//2,:]
            Mat_out[1:self.num_wann:2,0:self.num_wann:2,:] = Mat_in[self.num_wann//2:self.num_wann,0:self.num_wann//2,:]
            Mat_out[0:self.num_wann:2,1:self.num_wann:2,:] = Mat_in[0:self.num_wann//2,self.num_wann//2:self.num_wann,:]
            Mat_out[1:self.num_wann:2,1:self.num_wann:2,:] = Mat_in[self.num_wann//2:self.num_wann,self.num_wann//2:self.num_wann,:]
            return Mat_out		


    def symmetrize(self):
        #====Time Reversal====
        #syl: (sigma_y)^T *1j, syr: sigma_y*1j
        if self.spin and self.TR:
            print('TR == True')
            base_m = np.eye(self.num_wann//2)
            syl=np.array([[0.0,-1.0],[1.0,0.0]])
            syr=np.array([[0.0,1.0],[-1.0,0.0]])
            ul=np.kron(syl,base_m)
            ur=np.kron(syr,base_m)
            for ir in range(self.nRvec):
                self.HH_R[:,:,ir]=(self.HH_R[:,:,ir] + np.dot(np.dot(ul,np.conj(self.HH_R[:,:,ir])),ur))/2.0
        
        if self.spin:
            self.HH_R = self.spin_range(self.HH_R)
            self.AA_R = self.spin_range(self.AA_R)
        
        def average_H(self,H_res,A_res,iRvec,keep_New_R=True):
            nrot = self.nsymm
            R_list = np.array(iRvec,dtype=int)
            nRvec=len(R_list)
            tmp_R_list = []
            for rot in range(self.nsymm):
                print('rot = ',rot)
                p_map = np.zeros((self.num_wann_atom,self.num_wann,self.num_wann),dtype=complex)
                p_map_dagger = np.zeros((self.num_wann_atom,self.num_wann,self.num_wann),dtype=complex)
                for atom in range(self.num_wann_atom):
                    p_map[atom],p_map_dagger[atom] = self.full_p_mat(atom,rot)
                rot_map,vec_shift,magsym_break = self.atom_rot_map(rot)
                if magsym_break:
                    nrot -= 1
                else:
                    R_map = np.dot(R_list,np.transpose(self.symmetry['rotations'][rot]))
                    atom_R_map = R_map[:,None,None,:] - vec_shift[None,:,None,:] + vec_shift[None,None,:,:]
                    HH_all = np.zeros((nRvec,self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann),dtype=complex)
                    AA_all = np.zeros((nRvec,self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann,3),dtype=complex)
                    for iR in range(nRvec):
                        for atom_a in range(self.num_wann_atom):
                            for atom_b in range(self.num_wann_atom):
                                new_Rvec=list(atom_R_map[iR,atom_a,atom_b])
                                if new_Rvec in self.iRvec:
                                    new_Rvec_index = self.iRvec.index(new_Rvec)
                                    HH_all[iR,atom_a,atom_b,self.H_select[atom_a,atom_b]] = self.HH_R[self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index]
                                    AA_all[iR,atom_a,atom_b,self.H_select[atom_a,atom_b],:] = self.AA_R[self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index,:].dot(np.transpose(self.symmetry['rotations'][rot]) )
                                else:
                                    if new_Rvec in tmp_R_list:
                                        pass
                                    else:
                                        tmp_R_list.append(new_Rvec)

                    for atom_a in range(self.num_wann_atom):
                        for atom_b in range(self.num_wann_atom):
                            tmp = np.dot(np.dot(p_map_dagger[atom_a],HH_all[:,atom_a,atom_b]),p_map[atom_b])
                            H_res += tmp.transpose(0,2,1)
                            tmp0 = np.dot(np.dot(p_map_dagger[atom_a],AA_all[:,atom_a,atom_b,:,:,0]),p_map[atom_b])
                            tmp1 = np.dot(np.dot(p_map_dagger[atom_a],AA_all[:,atom_a,atom_b,:,:,1]),p_map[atom_b])
                            tmp2 = np.dot(np.dot(p_map_dagger[atom_a],AA_all[:,atom_a,atom_b,:,:,2]),p_map[atom_b])
                            A_res[:,:,:,0] += tmp0.transpose(0,2,1)
                            A_res[:,:,:,1] += tmp1.transpose(0,2,1)
                            A_res[:,:,:,2] += tmp2.transpose(0,2,1)
                            #if atom_a ==0 and atom_b == 0:
                            #    test_i = self.iRvec.index([1,1,0])	
                            #    print(self.HH_R[self.H_select[atom_a,atom_b],test_i].reshape(8,8).real)
                                #print(self.AA_R[self.H_select[atom_a,atom_b],test_i,2].reshape(8,8).real)
                            #    print('======================')
                            #    print(tmp.transpose(0,2,1)[self.H_select[atom_a,atom_b],test_i].reshape(8,8).real)
                                #print(tmp2.transpose(0,2,1)[self.H_select[atom_a,atom_b],test_i].reshape(8,8).real)
                            #    print('======================')


            if keep_New_R:
                    return  H_res/nrot,A_res/nrot , tmp_R_list
            else:
                    return  H_res/nrot,A_res/nrot			
        
        H_res_exist = np.zeros((self.num_wann,self.num_wann,self.nRvec),dtype=complex)
        A_res_exist = np.zeros((self.num_wann,self.num_wann,self.nRvec,3),dtype=complex)
        print('##########################')
        print('Existing Block')
        HH_R_re1,AA_R_re1, iRvec_add =  average_H(self,H_res_exist,A_res_exist,self.iRvec,keep_New_R=True)
        nRvec_add = len(iRvec_add)
        print('nRvec_add =',nRvec_add)
        H_res_add=np.zeros((self.num_wann,self.num_wann,nRvec_add),dtype=complex)
        A_res_add=np.zeros((self.num_wann,self.num_wann,nRvec_add,3),dtype=complex)
        if nRvec_add > 0:
            print('##########################')
            print('Additional Block')
            HH_R_re2,AA_R_re2  =  average_H(self,H_res_add,A_res_add,iRvec_add,keep_New_R=False)
            HH_R_final = np.zeros((self.num_wann,self.num_wann,nRvec_add+self.nRvec),dtype=complex)
            AA_R_final = np.zeros((self.num_wann,self.num_wann,nRvec_add+self.nRvec,3),dtype=complex)
            HH_R_final[:,:,:self.nRvec]=HH_R_re1
            HH_R_final[:,:,self.nRvec:]=HH_R_re2
            AA_R_final[:,:,:self.nRvec]=AA_R_re1
            AA_R_final[:,:,self.nRvec:]=AA_R_re2
            self.nRvec += nRvec_add
            self.iRvec += iRvec_add
        else:
            HH_R_final = HH_R_re1
            AA_R_final = AA_R_re1

        if self.spin:
            self.HH_R = self.spin_range(HH_R_final,inv=True)
            self.AA_R = self.spin_range(AA_R_final,inv=True)
        else:
            self.HH_R = HH_R_final
            self.AA_R = AA_R_final

