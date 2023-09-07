import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import pickle
from scipy.optimize import curve_fit as cf
import sys
#%matplotlib notebook
import time
import matplotlib.colors as mcolors

cmap_colors = [(0.0, 'black'), (0.25, 'red'), (0.5, 'green'), (0.75, 'blue'), (1.0, 'yellow')]
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)


L=int(sys.argv[1])
T=float(sys.argv[2])
#e=int(sys.argv[3])


Lx=L
Ly=3*L
N_sites=Lx*Ly

vmin, vmax = -1.0, 3.0

'''
----------
convention
----------
-1 : No particle
0 : Particle along right
1 : Particle along down
2 : Particle along left
3 : Particle along up
'''

#start = time.time()


def nbr2D(L):
    nbrarr=np.zeros((N_sites,4),dtype=int)
    
    for i in range(N_sites):
        k1=i//Ly
        k2=i%Ly
        if 1<=k1<=Lx-2:
            nbrarr[i][3]=(k1-1)*Ly+k2 #up
            nbrarr[i][1]=(k1+1)*Ly+k2 #down
        else:
            a=(k1==0)
            nbrarr[i][3]=Ly*(Lx-2+a)+k2 #up
            nbrarr[i][1]=Ly*a+k2 #down
        if 1<=k2<=Ly-2:
            nbrarr[i][2]=i-1 #left
            nbrarr[i][0]=i+1 #right
        else:
            b=(k2==0)
            nbrarr[i][2]=(i-1+Ly*b)
            nbrarr[i][0]=(i+1-Ly*(1-b))
    return nbrarr
nbrarr=nbr2D(L)

@jit(nopython=True)
def MC_update(state_arr):
    pos=np.random.randint(N_sites)
    d=0
    n=int(np.random.normal(0,T))
    if state_arr[pos]==-1:
        return None
    else:
        d=(state_arr[pos]+n)%4
        state_arr[pos]=d
    r=np.random.random()
    if r<1/20:
        d=(d+1)%4
    elif r<2/20:
        d=(d+2)%4
    elif r<3/20:
        d=(d+3)%4
    #print(d)
    if state_arr[nbrarr[pos][d]]!=-1:
        return None
    state_arr[nbrarr[pos][d]]=state_arr[pos]
    state_arr[pos]=-1
    return None


def random_initialize(state_arr):
    pos_arr=np.arange(0,N_sites)
    np.random.shuffle(pos_arr)
    for i in range(N_sites//2):
        state_arr[pos_arr[i]]=np.random.choice([0,1,2,3])

@jit(nopython=True)
def get_data(state_arr):
    #start=time.time()
    Trlax=10**8
    for i in range(Trlax):
        for j in range(N_sites):
            MC_update(state_arr)
    #end=time.time()

if __name__=='__main__':
    state_arr=np.ones(N_sites,dtype='int')*-1
    random_initialize(state_arr)
    plt.imshow(state_arr.reshape(Lx,Ly),cmap=custom_cmap, vmin=vmin, vmax=vmax)
    plt.title(f'T={T}')
    plt.savefig(f'init_T={T}.png')
    
    start=time.time()
    get_data(state_arr)
    end=time.time()
    tm=end-start

    plt.imshow(state_arr.reshape(Lx,Ly),cmap=custom_cmap, vmin=vmin, vmax=vmax)
    plt.title(f'T={T}')
    plt.savefig(f'final_T={T}.png')
    print(f"runtime = {tm/60} mins")





