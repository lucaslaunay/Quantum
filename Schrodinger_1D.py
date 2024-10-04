import numpy as np 
import matplotlib.pyplot as p
from matplotlib import animation
from matplotlib.animation import PillowWriter
import numba 
from numba import jit

#define params 

Nx = 301
Nt = 100000
dx = 1/(Nx-1)
dt = 1e-7
x = np.linspace(0, 1, Nx)

psi0_x = np.sqrt(2)*np.sin(np.pi*x)
mu, sigma = 1/4, 1/20
V = 1e4*np.exp(-(x-mu)**2/(2*sigma**2))

#print(dt/dx**2)

psi = np.zeros([Nt, Nx])

psi[0] = psi0_x


#print(V)


@numba.jit("c16[:,:](c16[:,:])", nopython=True, nogil=True)
def compute_psi(psi):
    for t in range(0, Nt-1):
        for i in range(1, Nx-1):
            psi[t+1][i] = psi[t][i] + 1j/2 * dt/dx**2 * (psi[t][i+1] - 2*psi[t][i] + psi[t][i-1]) - 1j*dt*V[i]*psi[t][i]
        
        normal = np.sum(np.absolute(psi[t+1])**2)*dx
        for i in range(1, Nx-1):
            psi[t+1][i] = psi[t+1][i]/normal
        
    return psi
        
psi = compute_psi(psi.astype(complex))

#plt.plot(x, np.absolute(psi[5000])**2)
#plt.show()

#animation - takes forever to run 
def animate(i):
    ln1.set_data(x, np.absolute(psi[100*i])**2)
    ln2.set_data(x, V)
    time_text.set_text('$(10^4 mL^2)^{-1}t=$'+'{:.1f}'.format(100*i*dt*1e4))
    
fig, ax = plt.subplots(1,1, figsize=(8,4))
ax.grid()
ln1, = plt.plot([], [], 'r-', lw=2, markersize=8)
ln1.set_label('$|\Psi(x)|^2$')
ln2, = plt.plot([], [], 'b--', lw=2, markersize=8)
ln2.set_label('V(x)')
time_text = ax.text(0.65, 16, '', fontsize=15,
           bbox=dict(facecolor='white', edgecolor='black'))
ax.set_ylim(-1, 20)
ax.set_xlim(0,1)
ax.set_ylabel('$|\psi(x)|^2$', fontsize=20)
ax.set_xlabel('$x/L$', fontsize=20)
ax.legend(loc='upper left')
ax.set_title('$(mL^2)V(x) = -10^4 \cdot n(x, \mu=L/4, \sigma=L/20)$')
plt.tight_layout()
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
ani.save('off_center_gaussian.gif',writer='pillow',fps=50,dpi=100)



