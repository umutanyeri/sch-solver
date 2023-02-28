import numpy as np
from methods import SSF
from methods import CN
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import erf

"""
This codes changes dt and observe how the numerical results changes accordingly by checking;
    1) the difference between numerical results using "different dt values" in order to detect any convergence if there is any
    2) the difference between analytical solution and numerical solution. This is feasible since we have an analytical solution in this case.
"""

#returns analytic solution at given time t
def analytic(x,t):
#define psi with same length of space "x" and zero indices
    psi = np.zeros( len(x) )+0j
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
    return psi / (2*xw) * 1/(s**2*2*np.pi)**0.25

#returns initial wave function
def initwave(x):
#prepare the initial wave
    psi = np.exp( -(x-x0)**2/4/s**2)
#add initial momentum
    psi = np.exp(1j*x*k0)*psi
    return psi * 1/(s**2*2*np.pi)**0.25

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
psi_0 = np.exp( -(x-x0)**2/4/s**2) * 1/(s**2*2*np.pi)**.25
#add initial momentum
psi_0 = np.exp(1j*x*k0)*psi_0



#determine different step sizes
dt_list = dx**2*np.linspace(.1,1,10)

#write figure
fig,ax = plt.subplots(1,2 ,figsize=(40/2.54,20/2.54))
Dt = 0.12 #time differnces between data points in plot
N_data = 100
xaxis = np.linspace(0,N_data*Dt,N_data+1)

#define lists that will contain projections w.r.t time
proj_analytic = np.zeros(N_data+1) #projection between each numerical state with analytical solution
proj_first = np.zeros(N_data+1) #projection between each numerical state with analytical solution
proj_successive = np.zeros(N_data+1) #projection between successive numerical states ( e.g. proj between state[1]&state[0]  or state[2]&state[1] etc.)

#loop over dt_list
for i in range(1,len(dt_list)):
#take initial waves    
    psi_old = psi_0+0j  #wave using dt = dt_list[i-1]
    psi_next = psi_0+0j #wave using dt = dt_list[i]

#calculate initial projections
    proj = abs(sum(np.conj(psi_next) * psi_old * dx)) - 1
    proj_successive[0] = abs(proj)
    proj = abs(sum(np.conj(analytic(x,0)) * psi_next * dx)) - 1
    proj_analytic[0] = abs(proj)
#take projection of the first parameter seperately since in every loop, we need to calculate the psi_next only unless it is not the first step of the loop
    if ( i == 1 ):
        proj = abs(sum(np.conj(analytic(x,0)) * psi_next * dx)) - 1
        proj_first[0] = abs(proj)
#loop over time    
    for j in  range(1,N_data+1):
#use the time saved in the xaxis list previously        
        t = xaxis[j]
#calculate the numerical solutions        
        psi_next = SSF(psi_next , x , Dt , v , dt_list[i])
        psi_old = SSF(psi_old , x , Dt , v , dt_list[i-1])        
#calculate projections
        if ( i == 1 ):
            proj = abs(sum(np.conj(analytic(x,t)) * psi_old * dx)) - 1
            proj_first[j] = abs(proj)
        proj = abs(sum(np.conj(psi_next) * psi_old * dx)) - 1
        proj_successive[j] = abs(proj)

        proj = abs(sum(np.conj(analytic(x,t)) * psi_next * dx)) - 1
        proj_analytic[j] = abs(proj)
#plot the time evolution of projections
    if(i==1):
        ax[1].plot( xaxis , abs(proj_first) , label="$dt/dx^2$=%.4f"%(dt_list[i-1]/dx**2))
    ax[1].plot( xaxis , abs(proj_analytic) , label="$dt/dx^2$=%.4f"%(dt_list[i]/dx**2))
    ax[0].plot( xaxis , abs(proj_successive) , label="%.4f vs. %.4f"%(dt_list[i-1]/dx**2,dt_list[i]/dx**2))

#make our plot more beautiful
ax[0].set_ylabel("absolute projection")
ax[0].set_title("Projections between states \n obtained by using different parameters")
ax[1].set_title("Projection between analytical function")
for i in range(2):
    ax[i].set_xlabel("time")
    ax[i].legend()
    ax[i].grid(alpha=.3)
    ax[i].set_yscale('log')
    ax[i].set_ylim(10**-7,10**0)
    ax[i].set_xlim(0,N_data*Dt)
#see the results    
plt.show()
