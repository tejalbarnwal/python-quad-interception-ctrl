import numpy as np
from quad_model import Quadrotor
from controller import Controller
import matplotlib.pyplot as plt

class Simulator():
    def __init__(self, quad, ctrl):
        self.quad = quad
        self.ctrl = ctrl
        
        self.Tstep = 0.01
        self.Tf = 20.0
        self.N = int(self.Tf/ self.Tstep)
        self.target_position = None
        self.n_td = None
        self.state = {}
        self.history = {}
        
    
    def run(self, target_pos, n_td = None, Tf = 5.0,  Ts=0.01):
        self.Tf = Tf
        self.Tstep = Ts
        self.N = int(self.Tf/ self.Tstep)
        print("time step: ", self.Tstep)
        print("final time: ", self.Tf)
        print("no of iteration: ", self.N)
        
        self.history["time_step"] = np.zeros((1,self.N))
        self.history["n_t"] = np.zeros((3, self.N))
        self.history["n_td"] = np.zeros((3, self.N))
        self.history["pr"] = np.zeros((3,self.N))
        self.history["vr"] = np.zeros((3,self.N))
        self.history["R"] = np.zeros((3, 3,self.N))
        self.history["drone_angles"] = np.zeros((3, self.N))
        self.history["p"] = np.zeros((3, self.N))
        self.history["v"] = np.zeros((3, self.N))
        self.history["fd"] = np.zeros((1, self.N))
        self.history["wd"] = np.zeros((3, self.N))
        self.history["clip_fd"] = np.zeros((1, self.N))
        self.history["clip_wd"] = np.zeros((3, self.N))
        
        self.target_position = target_pos
        
        self.state["pr"] = self.quad.state["r"] - self.target_position
        self.state["vr"] = self.quad.state["v"]
        self.state["R"] = self.quad.state["R"]
        self.state["n_t"] = -1.0 * self.state["pr"] / np.linalg.norm(self.state["pr"])
        
        self.n_td_bool = True if n_td is not None else False
        self.state["n_td"] = n_td if self.n_td_bool else self.state["n_t"]
        
        for i in range(self.N):
            print("-----------------------------------------------------------------")
            print("iteration: ", i)
            
            fd, wd = self.ctrl.update(self.state)
            ud = np.array([fd, wd[0], wd[1], wd[2]])
            updated_fd, updated_wd = self.quad.update(ud, self.Tstep)
            
            self.state["pr"] = self.quad.state["r"] - self.target_position
            self.state["vr"] = self.quad.state["v"]
            self.state["R"] = self.quad.state["R"]
            self.state["n_t"] = -1.0 * self.state["pr"] / np.linalg.norm(self.state["pr"])
            self.state["n_td"] = n_td if self.n_td_bool else self.state["n_t"]
            
            
            self.history["time_step"][:, i] = (i)*self.Tstep
            self.history["n_t"][:, i] = self.state["n_t"]
            self.history["n_td"][:, i] = self.state["n_td"]
            self.history["pr"][:, i] = self.state["pr"]
            self.history["vr"][:, i] = self.state["vr"]
            self.history["R"][:, :, i] = self.state["R"]
            self.history["p"][:, i] = self.quad.state["r"]
            self.history["v"][:, i] = self.quad.state["v"]
            self.history["fd"][:, i] = fd
            self.history["wd"][:, i] = wd
            self.history["clip_fd"][:, i] = updated_fd
            self.history["clip_wd"][:, i] = updated_wd
            
            R = self.state["R"]
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arcsin(-1.0 * R[2, 0])
            yaw = np.arctan2(R[1, 0], R[0, 0])
            self.history["drone_angles"][:, i] = np.array([roll, pitch, yaw])
            
            print("roll: ", roll)
            print("pitch: ", pitch)
            print("yaw: ", yaw)
            

    def plot(self):
        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(wspace=0.25)
        
        tvec = self.history["time_step"][0, :]
        pos = self.history["p"]
        vel = self.history["v"]
        
        ### position
        ax = fig.add_subplot(3, 2, 1)
        ax.plot(tvec, pos[0, :], label="drone pos")
        ax.set_ylabel("x")
        ax.grid()
        ax.legend()
        
        ax = fig.add_subplot(3, 2, 3)
        ax.plot(tvec, pos[1, :], label="drone pose")
        ax.set_ylabel("y")
        ax.grid()
        ax.legend()
        
        ax = fig.add_subplot(3, 2, 5)
        ax.plot(tvec, pos[2, :], label="drone pos")
        ax.set_ylabel("z")
        ax.grid()
        ax.legend()
        
        # velocity
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(tvec, vel[0, :], label="drone vel")
        ax.set_ylabel("x")
        ax.grid()
        ax.legend()
        
        ax = fig.add_subplot(3, 2, 4)
        ax.plot(tvec, vel[1, :], label="drone vel")
        ax.set_ylabel("y")
        ax.grid()
        ax.legend()
        
        ax = fig.add_subplot(3, 2, 6)
        ax.plot(tvec, vel[2, :], label="drone vel")
        ax.set_ylabel("z")
        ax.grid()
        ax.legend()
        
        # angles
        fig2 = plt.figure(figsize=(12, 3))
        fig2.subplots_adjust(wspace=0.25)
        fig2.suptitle("control inputs")
        
        th = self.history["fd"]
        th_clip = self.history["clip_fd"]
        wd = self.history["wd"]
        wd_clip = self.history["clip_wd"]
        
        ax = fig2.add_subplot(2, 2, 1)
        ax.plot(tvec, th[0, :], "r-", label="control thrust")
        ax.plot(tvec, th_clip[0, :], "b-", label="clipped control thrust")
        ax.set_ylabel("thrust")
        ax.grid()
        ax.legend()
        
        ax = fig2.add_subplot(2, 2, 2)
        ax.plot(tvec, wd[0, :], "r-", label="w_x")
        ax.plot(tvec, wd_clip[0, :], "b-", label="clipped w_x")
        ax.set_ylabel("w_x")
        ax.grid()
        ax.legend()
        
        ax = fig2.add_subplot(2, 2, 3)
        ax.plot(tvec, wd[1, :], "r-", label="w_y")
        ax.plot(tvec, wd_clip[1, :], "b-", label="clipped w_y")
        ax.set_ylabel("w_y")
        ax.grid()
        ax.legend()
        
        ax = fig2.add_subplot(2, 2, 4)
        ax.plot(tvec, wd[2, :], "r-", label="w_z")
        ax.plot(tvec, wd_clip[2, :], "b-", label="clipped w_z")
        ax.set_ylabel("w_z")
        ax.grid()
        ax.legend()
        
        
        
        print("drone position: ")
        x = self.history["p"][0, :]
        print(x.shape)
        
        plt.show()