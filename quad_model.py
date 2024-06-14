import numpy as np
import matplotlib.pyplot as plt
from utils import rk4, hat_map

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

class Quadrotor(object):
    """Quadrotor
    This class models the physical quadrotor vehicle evolving in SE(3).
    """
    def __init__(self, r=None, v=None, R=None):
        # internal quadrotor state
        self.r = r if r is not None else np.zeros((3))
        self.v = v if v is not None else np.zeros((3))
        self.R = R if R is not None else np.identity(3)
        
        # phyiscal true parameters
        self.g = np.array([0.0, 0.0, -9.81])
        self.mass = 1.0
        self.e3 = np.array([0.0, 0.0, 1.0])
        
        # max control actuation
        self.wmax = 0.2
        self.fmax = 30.0        
        # convenience
        self.Niters = 0
        self.state = {}
        self.state["r"] = self.r
        self.state["v"] = self.v
        self.state["R"] = self.R
        
    def __str__(self):
        s  = "Quadrotor state after {} iters:\n".format(self.Niters)
        s += "r:     {}.T\n".format(self.r.T)
        s += "v:     {}.T\n".format(self.v.T)
        s += "R:     {}\n".format(self.R)
        return s
    
    def clamp(self, v, limit):
        if np.linalg.norm(v) > limit:
            a = (limit / np.linalg.norm(v)) * v
        else:
            a = v
        return a
    
    def update(self, u, dt):
        # We've been through another iteration
        self.Niters += 1
        
        # thrust and body rates input
        u = u.flatten()
        f = np.array([u[0]])
        w_b = np.array([[u[1], u[2], u[3]]])
        
        print("u:", u)
        print("f: ", f)
        print("w_b: ", w_b)
        
        # Saturate control effort
        f = min(max(f, 0.0), self.fmax)
        print("clipped f: ", f)
        w_b = self.clamp(w_b, self.wmax)
        print("clipped w_b: ", w_b)
        
        print("RUN KINEMATICS AND DYNAMICS")
        
        # Translational Kinematics
        fun1 = lambda r: self.v # inertial velocities
        self.r = rk4(fun1, self.r, dt)
        print("r: ", self.r)
        
        # Rotational kinematics
        y = np.matmul(self.R, hat_map(w_b))
        # print("y: ", y)
        fun2 = lambda R: (y.flatten())
        flattenedR = rk4(fun2, (self.R).flatten(), dt)
        self.R = np.reshape(flattenedR, (3, 3))
        print("R: ", self.R)
        
        # # # Translational dynamics
        # print(self.g.shape)
        # print(np.matmul(self.R, self.e3).shape)
        h = (self.g + (1/ self.mass)* (f * np.matmul(self.R, self.e3)))
        # print(h)
        # print(h.shape)
        fun3 = lambda v: h
        self.v = rk4(fun3, self.v, dt)
        print("v: ", self.v)
        
        self.state["r"] = self.r
        self.state["v"] = self.v
        self.state["R"] = self.R
        
        # f = np.reshape(f, (1, 1))
        # # update control input
        # u = np.hstack((f, w_b.T))

        return f, w_b