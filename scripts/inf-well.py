import numpy as np
from methods import SSF
from methods import CN
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import erf

def analytic(x,t):
    psi = x*0j
#add eigenvectors as many as number of spatial point
    for n in range(len(x)):
#calculate allowed momentum and corresponding energies
        kn = n*np.pi/(2*xw)
        en = kn**2/2
#calculate projections to gaussian with initial position x_0 and momentum k_0
#projection to e^{ikx}
        A_n = np.exp(-s**2 * (kn-k0)**2 - 1j*(kn-k0)*x0) * np.sqrt(np.pi)*2*s
#projection to e^{-ikx}
        B_n = np.exp(-s**2 * (-kn-k0)**2 - 1j*(-kn-k0)*x0) * np.sqrt(np.pi)*2*s
#if the integer n is even calculate the constant in front of the sine
        if(n % 2 == 0):
            psi += 1j*(A_n-B_n) * np.sin(x*kn) *np.exp(-1j*en*t)
#if the integer n is odd calculate the constant in front of the cosine
        else:
            psi += (A_n+B_n) * np.cos(x*kn) *np.exp(-1j*en*t)
#scale the solution with width of the well (normalization factor of FT)
    return psi / (2*xw)

#define position grid and potential term
xw = 20 #half of the width of well centered at origin
xf = 2*xw #half of the total position space
Nx = xf*10*2 # number of positional point
x = np.linspace(-xf,xf,Nx) #position
dx = x[1]-x[0] 
v = np.asarray( [1e12 if abs(i) >= xw else 0 for i in x] ) # infinite potential well

#prepare the initial wave
#take a gaussian centered at some defined location and s.dev.
x0 = 8
k0 = 6
s = 2
psi = np.exp( -(x-x0)**2/4/s**2)
#add initial momentum
psi = np.exp(1j*x*k0)*psi
psi1 = psi*1
psi2 = psi*1

#arrange figures
N_fig_x = 5
N_fig_y = 2
fig,ax = plt.subplots(N_fig_y , N_fig_x , figsize=((N_fig_x+1)*5/2.54,(N_fig_y+1)*5/2.54) , tight_layout=True)

#evolve w.f.
Dt=1.2  #time difference between subfigures
dt = dx**2/4 #time difference between steps in time evolution
for sf in range(N_fig_x*N_fig_y):
#locate the target subfigure
    i = sf // N_fig_x
    j = sf % N_fig_x
#calculate the actual time in the plot
    t = (sf+1)*Dt
#plot the analytical solution
    ax[i][j].plot(x, analytic(x,t).real , lw=2,label="analytic")
#evolve the w.f.
#w/split-step fourier method
    psi1 = SSF(psi1 , x , Dt , v , dt)
#w/crank-nicolson method
    psi2 = CN(psi2 , x , Dt , v , dt)
#plot numerical solutions
    ax[i][j].plot(x,psi1.real , lw=1, label="SSF")
    ax[i][j].plot(x,psi2.real , lw=1, label="CN")
#plot inf. potential well
    ax[i][j].plot(x,v , label="v(x)")

    ax[i][j].set_title("t=%.3f"%(t))
    ax[i][j].grid(alpha=.3)
    ax[i][j].set_ylim(-1.1,1.1)
    ax[i][j].set_xlim(-xw,xw)
ax[0][0].legend()
plt.show()
