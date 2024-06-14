from derivative_lpf import DirtyDerivative
from scipy.spatial.transform import Rotation
import numpy as np
from utils import inverse_hat_map

class Controller():
    def __init__(self):
        self.kb = 3.0
        self.k1 = 0.005
        self.k2 = 0.005
        self.G = np.array([0.0, 0.0, -9.81])
    
    def inner_loop(self, vr, pr, n_t, n_td, R):
        mod_pr = np.linalg.norm(pr)
        t1 = -1.0 * self.k1 * vr
        
        z2 = vr + self.k1 * pr
        t2 = -1.0 * self.k2 * z2
        
        t3 = -1.0 * pr
        
        z1 = 1 - np.matmul(n_td.T, n_t)
        scale = z1 / (self.kb**2 - z1**2)
        t4 = scale * (1.0 / mod_pr) \
            * np.matmul((-1.0 * np.identity(3)) + np.matmul(n_t, n_t.T), n_td)
            
        ad = t1 + t2 + t3 + t4
        net_acc = ad- self.G
        n_fd = net_acc / np.linalg.norm(net_acc)
        n_f = np.matmul(R, np.array([0.0, 0.0, 1.0]))
        
        axis = np.cross(n_f, n_fd)
        angle = np.arccos(np.matmul(n_t.T, n_fd))
        axis = axis / np.linalg.norm(axis)
        Rtilt_ = Rotation.from_rotvec(angle * axis)
        Rtilt = Rtilt_.as_matrix()
        
        Rd = np.matmul(Rtilt, R)
        
        fd = np.dot(n_f, net_acc)
        tr1 = np.matmul(Rd.T, R)
        tr2 = np.matmul(R.T, Rd)
        w2 = -1.0 * inverse_hat_map(tr1 - tr2)
        
        
        print("------ innner loop ------")
        print("ad: ", ad)
        print("fd: ", fd)
        print("w2: ", w2)
        
        return fd, w2
    
    def outer_loop(self, n_td, n_t, R):
        z1 = 1 - np.matmul(n_td.T, n_t)
        scale = z1 / (self.kb**2 - z1**2)
        w1 = scale * np.matmul(R.T, np.cross(n_td, n_t))
        
        print("------ outer loop control ------")
        print("z1: ", z1)
        print("scale: ", scale)
        print("w1: ", w1)
        return w1
    
    def update(self, state):
        # read states and commanded
        n_td = state["n_td"]
        n_t = state["n_t"]
        pr = state["pr"]
        vr = state["vr"]
        R = state["R"]
        # run outer loop
        wb1 = self.outer_loop(n_td, n_t, R)
        # run inner loop
        fd, wb2 = self.inner_loop(vr, pr, n_t, n_td, R)
        # return thrust and body rates
        return fd, (wb1+wb2)
