import numpy as np

def num_cart_dim(key):
    """
    returns the number of cartesian dimensions of a matrix by key
    """
    if key in ["Ham"]:
        return 0
    elif key in ["AA", "BB", "CC", "SS", "SH", "OO"]:
        return 1
    elif key in ["SHA", "SA", "SR", "SHR", "GG", "FF"]:
        return 2
    else:
        raise ValueError(f"unknown matrix {key}")

def read_tb(seedname):
    f=open(seedname+"_tb.dat","r")
    f.readline()
    real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
    num_wann=int(f.readline())
    nRvec=int(f.readline())
    Ndegen=[]
    while len(Ndegen)<nRvec:
        Ndegen+=f.readline().split()
    Ndegen=np.array(Ndegen,dtype=int)
    iRvec=[]
    HH_R=np.zeros( (num_wann,num_wann,nRvec) ,dtype=complex)
    for ir in range(nRvec):
        f.readline()
        iRvec.append(list(np.array(f.readline().split()[0:3],dtype=int)))
        hh=np.array( [[f.readline().split()[2:4]
                for n in range(num_wann)]
                    for m in range(num_wann)],dtype=float).transpose( (1,0,2) )
        HH_R[:,:,ir]=(hh[:,:,0]+1j*hh[:,:,1])/Ndegen[ir]
    AA_R=np.zeros( (num_wann,num_wann,nRvec,3) ,dtype=complex)
    for ir in range(nRvec):
        f.readline()
        assert (np.array(f.readline().split(),dtype=int)==np.array(iRvec[ir],dtype=int)).all()
        aa=np.array( [[f.readline().split()[2:8]
                    for n in range(num_wann)]
                    for m in range(num_wann)],dtype=float)
        AA_R[:,:,ir,:]=(aa[:,:,0::2]+1j*aa[:,:,1::2]).transpose( (1,0,2) )/Ndegen[ir]
    XX_R = {'Ham':HH_R, 'AA':AA_R} 

    return num_wann, nRvec, iRvec, XX_R, real_lattice

def write_hr(num_wann, nRvec, iRvec, XX_R, lattice):
    name="wannier90_sym_hr.dat"
    ndegen=list(np.ones((nRvec),dtype=int))
    with open(name,"w") as f:
        f.write("symmetrize wannier hr\n"+str(num_wann)+"\n"+str(nRvec)+"\n")
        nl = np.int32(np.ceil( nRvec/15.0))
        for l in range(nl):
            line="    "+'    '.join([str(np.int32(i)) for i in ndegen[l*15:(l+1)*15]])
            f.write(line+"\n")
        whh_r = np.round(XX_R['Ham'],decimals=6)
        for ir in range( nRvec):
            rx = int( iRvec[ir][0]);ry = int( iRvec[ir][1]);rz = int( iRvec[ir][2])
            for n in range( num_wann):
                for m in range( num_wann):
                    rp =whh_r[m,n,ir].real
                    ip =whh_r[m,n,ir].imag
                    line="{:5d}{:5d}{:5d}{:5d}{:5d}{:12.6f}{:12.6f}\n".format(rx,ry,rz,m+1,n+1,rp,ip)
                    f.write(line)
        f.close()

def write_tb(num_wann, nRvec, iRvec, XX_R, lattice):
    name="wannier90_sym_tb.dat"
    ndegen=list(np.ones(( nRvec),dtype=int))
    with open(name,"w") as f:
        f.write("symmetrize wannier tb\n"
                +"{:12.6f}{:12.6f}{:12.6f}\n".format( lattice[0,0], lattice[0,1], lattice[0,2])
                +"{:12.6f}{:12.6f}{:12.6f}\n".format( lattice[1,0], lattice[1,1], lattice[1,2])
                +"{:12.6f}{:12.6f}{:12.6f}\n".format( lattice[2,0], lattice[2,1], lattice[2,2])
                +str( num_wann)+"\n"+str( nRvec)+"\n")
        nl = np.int32(np.ceil( nRvec/15.0))
        for l in range(nl):
            line="    "+'    '.join([str(np.int32(i)) for i in ndegen[l*15:(l+1)*15]])
            f.write(line+"\n")
        whh_r = np.round(XX_R['Ham'],decimals=10)
        waa_r = np.round(XX_R['AA'],decimals=10)
        for ir in range( nRvec):
            rx = int( iRvec[ir][0]);ry = int( iRvec[ir][1]);rz = int( iRvec[ir][2])
            f.write("\n{:5d}{:5d}{:5d}\n".format(rx,ry,rz))
            for n in range( num_wann):
                for m in range( num_wann):
                    rp =whh_r[m,n,ir].real
                    ip =whh_r[m,n,ir].imag
                    line="{:5d}{:5d}{:17.8E}{:17.8E}\n".format(m+1,n+1,rp,ip)
                    f.write(line)
        for ir in range( nRvec):
            rx = int( iRvec[ir][0]);ry = int( iRvec[ir][1]);rz = int( iRvec[ir][2])
            f.write("\n{:5d}{:5d}{:5d}\n".format(rx,ry,rz))
            for n in range( num_wann):
                for m in range( num_wann):
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
