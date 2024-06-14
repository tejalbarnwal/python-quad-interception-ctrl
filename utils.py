import numpy as np

def Rot_v_to_v1(psi):
    R = np.array([
        [ np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [     0,           0    ,   1]
    ])
    return R

def Rot_v1_to_v2(theta):
    R = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [      0      , 1,        0      ],
        [np.sin(theta), 0,  np.cos(theta)]
    ])
    return R

def Rot_v2_to_b(phi):
    R = np.array([
        [1,       0,           0     ],
        [0,  np.cos(phi), np.sin(phi)],
        [0, -np.sin(phi), np.cos(phi)]
    ])
    return R

def Rot_v_to_b(phi, theta, psi):
    return Rot_v2_to_b(phi).dot(Rot_v1_to_v2(theta).dot(Rot_v_to_v1(psi)))

def Rot_i_to_b(phi, theta, psi):
    return Rot_v_to_b(phi, theta, psi)

def rk4(f, y, dt):
    """Runge-Kutta 4th Order
    
    Solves an autonomous (time-invariant) differential equation of the form dy/dt = f(y).
    """
    k1 = f(y)
    k2 = f(y + dt/2*k1)
    k3 = f(y + dt/2*k2)
    k4 = f(y + dt  *k3)
    a = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return a


def hat_map(x):
    x = x.flatten()
    # print(x)
    return np.array([   [0.0, -1.0 * x[2], x[1]],
                        [x[2], 0.0, -1.0 * x[0]],
                        [-1.0 * x[1], x[0], 0.0]])

def inverse_hat_map(X):
    return np.array([X[2, 1], X[0, 2], X[1, 0]])
