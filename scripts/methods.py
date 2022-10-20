import numpy as np

def SSF( f , x , t , v , dt ):
    # f(x,y)    : initial wavefunction
    # x         : position grid in 1D
    # t         : time to be evolved
    # v         : potential term in Sch. eqn.
    # dt        : magnitude of time steps used in the evolution

#calculate max # time steps
    N_t = 1.0*abs(t)/dt
#extract number of space points and minimum distance
    N_x = len(x)
    dx = x[1]-x[0]
#calculate momentum from space array
    px = np.fft.fftfreq(N_x,dx)*2*np.pi
#make sure that wave and potential is numpy.array
    f = np.asarray(f)
    v = np.asarray(v)
#initialize counter for time-steps
    n_t = 0    
    while(n_t < N_t):
#increase number of time step : n_t
        n_t = n_t + 1
#evolve the state by dt/2 time w/ position part of the Hamiltonian
        f = np.exp(-1j*dt/2*v) * f
#evolve the state by  dt  time w/ momentum part of the Hamiltonian
        f = np.fft.fft(f)
        f = np.exp(-1j*dt*(px**2)/2) * f
        f = np.fft.ifft(f , axis=0)        
# evolve the state by dt/2 time w/ position part of the Hamiltonian  
        f = np.exp(-1j*dt/2*v) * f
#return the state
    return f

def CN( f , x , t , v , dt ):
    # f(x,y)    : initial wavefunction
    # x         : position grid in 1D
    # t         : time to be evolved
    # v         : potential term in Sch. eqn.
    # dt        : magnitude of time steps used in the evolution

#calculate max # time steps
    N_t = 1.0*abs(t)/dt
#extract number of space points and minimum distance
    N_x = len(x)
    dx = x[1]-x[0]
#make sure that wave and potential is numpy.array
    f = np.asarray(f)
    v = np.asarray(v)
#construct Hamiltonian matrix
    H = np.identity(N_x)/dx**2 +  np.diag(v)
    H += np.asarray( [ [ -1/dx**2/2 if i+1 == j else 0 for j in range(N_x) ] for i in range(N_x) ] )
    H += np.asarray( [ [ -1/dx**2/2 if i-1 == j else 0 for j in range(N_x) ] for i in range(N_x) ] )
#construct matrix used in CN method
    H = np.matmul( np.linalg.inv( np.identity(N_x) + 1j*H*dt/2 ) , (np.identity(N_x) - 1j*H*dt/2) )
#initialize counter for time-steps
    n_t = 0
    while(n_t < N_t):
#increase number of time step : n_t
        n_t = n_t + 1
#evolve the state by using matrix above
        f = np.matmul(H , f)
#return the state
    return f
