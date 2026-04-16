import numpy as np
# helper: cross-product matrix
def cross(a): return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
m=5.0
c=np.array([0.0,-0.1,-0.05])
Icm=np.diag([0.06,0.04,0.05])
I_o=Icm + m*cross(c)@cross(c)            # inertia about origin
I_sp=np.block([[I_o, m*cross(c)],[-m*cross(c), m*np.eye(3)]])  # spatial inertia
omega=np.array([0.0,0.0,2.0]); vel=np.array([0.1,0.0,0.0])
v=np.concatenate([omega,vel])
# motion cross and dual
vx = np.block([[cross(omega), np.zeros((3,3))],[cross(vel), cross(omega)]])
vx_star = -vx.T
h = I_sp@v                               # spatial momentum
gyroscopic = vx_star@(I_sp@v)            # v\times* I v (for dot v = 0)
# results (h and gyroscopic) available for controller or observer use