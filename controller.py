from derivative_lpf import DirtyDerivative
from scipy.spatial.transform import Rotation
import numpy as np
from utils import inverse_hat_map, Rot_i_to_b

class Controller():
    def __init__(self):
        self.k1 = 1.0
        self.tau = 1.0
        self.G = np.array([0.0, 0.0, -9.81])
    
    def compute_thrust(self,tilt_los, start_tilt_los, R):
        print("------ cntrl: thrust scalar ------")
        
        f1 = self.k1 * (180.0 / 3.1415926) * (tilt_los - start_tilt_los) + 9.81
        print("f1: ", f1)
        e3 = e3 = np.array([[0], [0], [1]])
        print("e3: ", e3)
        r = np.matmul(R, e3)
        print("R: ", R)
        print("r: ", r)
        f2 = np.dot(r.flatten(), e3.flatten())
        print("f2: ", f2)
        
        f = f1/ f2
        print("f: ", f)
        return f
    
    def compute_w(self, desired_pitch, yaw_los, R):
        print("---- cntrl: compute angular rates")
        
        Rd_T = Rot_i_to_b(0.0, desired_pitch, yaw_los)
        Rd = Rd_T.T
        tr1 = np.matmul(Rd.T, R)
        tr2 = np.matmul(R.T, Rd)
        w = self.tau * -1.0 * inverse_hat_map(tr1 - tr2)
        print("w: ", w)
        return w
    
    def update(self, state):
        # read states and commanded
        start_tilt_los = state["start_tilt_los"]
        desired_pitch = state["desired_pitch"]
        n_t = state["n_t"]
        pr = state["pr"]
        vr = state["vr"]
        R = state["R"]
        
        deg_to_rad = 3.1415926/180.0
        tilt_los = np.arctan2(n_t[2], np.sqrt(n_t[0]**2 + n_t[1]**2)) #+ np.random.normal(loc=0, scale=1.0*deg_to_rad)
        yaw_los = np.arctan2(n_t[1] , n_t[0]) #+ np.random.normal(loc=0, scale=0.5*deg_to_rad)

        fd = self.compute_thrust(tilt_los, start_tilt_los, R)
        wb = self.compute_w(desired_pitch, yaw_los, R)

        return fd, wb
