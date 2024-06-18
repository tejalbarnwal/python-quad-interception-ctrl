import numpy as np
from quad_model import Quadrotor
from controller import Controller
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.last_strike_index = 0
        
    
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
        self.history["LOS_tilt"] = np.zeros((1, self.N))
        self.history["LOS_yaw"] = np.zeros((1, self.N))
        self.history["z1"] = np.zeros((1, self.N))
        
        self.target_position = target_pos
        
        self.state["pr"] = self.quad.state["r"] - self.target_position
        self.state["vr"] = self.quad.state["v"]
        self.state["R"] = self.quad.state["R"]
        self.state["n_t"] = -1.0 * self.state["pr"] / np.linalg.norm(self.state["pr"])
        
        self.n_td_bool = True if n_td is not None else False
        print("ntd bool: ", self.n_td_bool)
        self.state["n_td"] = n_td if self.n_td_bool else self.state["n_t"]
        
        mod_pr_value = 1000000.0
        
        
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
            print("n_t: ", self.state["n_t"])
            print("n_td: ", self.state["n_td"])
            print("mod pr: ", np.linalg.norm(self.state["pr"]))
            
            tilt_los = np.arctan2(self.state["n_t"][2], np.sqrt(self.state["n_t"][0]**2 + self.state["n_t"][1]**2))
            yaw_los = np.arctan2(self.state["n_t"][1] , self.state["n_t"][0])
            z1_ = 1 - np.dot(self.state["n_td"], self.state["n_t"])
            
            
            print("LOS tilt angles: ", tilt_los)
            print("LOS yaw angles: ", yaw_los)
            self.history["LOS_tilt"][:, i] = tilt_los
            self.history["LOS_yaw"][:, i] = yaw_los
            self.history["z1"][:, i] = z1_
            
            if (np.linalg.norm(self.state["pr"]) <= mod_pr_value):
                mod_pr_value = np.linalg.norm(self.state["pr"])
            else:
                print("strikkeee doneee")
                self.last_strike_index = i+1
                break
            

    def plot(self):
        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(wspace=0.25)
        
        tvec = self.history["time_step"][0, :self.last_strike_index]
        pos = self.history["p"][:, :self.last_strike_index]
        vel = self.history["v"][:, :self.last_strike_index]
        
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
        
        # control inputs
        fig2 = plt.figure(figsize=(12, 3))
        fig2.subplots_adjust(wspace=0.25)
        fig2.suptitle("control inputs")
        
        th = self.history["fd"][:, :self.last_strike_index]
        th_clip = self.history["clip_fd"][:, :self.last_strike_index]
        wd = self.history["wd"][:, :self.last_strike_index]
        wd_clip = self.history["clip_wd"][:, :self.last_strike_index]
        
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
        
        
        # pr
        fig3 = plt.figure(figsize=(12, 3))
        fig3.subplots_adjust(wspace=0.25)
        fig3.suptitle("relative position")
        
        pr_ = self.history["pr"][:, :self.last_strike_index]
        
        ax = fig3.add_subplot(2, 2, 1)
        ax.plot(tvec, pr_[0, :], label="pr_x")
        ax.set_ylabel("pr_x")
        ax.grid()
        ax.legend()
        
        ax = fig3.add_subplot(2, 2, 2)
        ax.plot(tvec, pr_[1, :], label="pr_y")
        ax.set_ylabel("pr_y")
        ax.grid()
        ax.legend()
        
        ax = fig3.add_subplot(2, 2, 3)
        ax.plot(tvec, pr_[2, :], label="pr_z")
        ax.set_ylabel("pr_z")
        ax.grid()
        ax.legend()
        
        ax = fig3.add_subplot(2, 2, 4)
        ax.plot(tvec, np.linalg.norm(pr_, axis=0), label="mod_pr")
        ax.set_ylabel("mod_pr")
        ax.grid()
        ax.legend()
        
        # drone angles
        fig4 = plt.figure(figsize=(12, 3))
        fig4.subplots_adjust(wspace=0.25)
        fig4.suptitle("drone angles")
        
        angle = self.history["drone_angles"][:, :self.last_strike_index]
        
        ax = fig4.add_subplot(2, 2, 1)
        ax.plot(tvec, angle[0, :]*180/np.pi, label="pitch")
        ax.set_ylabel("pitch")
        ax.grid()
        ax.legend()
        
        ax = fig4.add_subplot(2, 2, 2)
        ax.plot(tvec, angle[1, :]*180.0/np.pi, label="roll")
        ax.set_ylabel("roll")
        ax.grid()
        ax.legend()
        
        ax = fig4.add_subplot(2, 2, 3)
        ax.plot(tvec, angle[2, :]*180.0/np.pi, label="yaw")
        ax.set_ylabel("yaw")
        ax.grid()
        ax.legend()
        
        # los angles
        fig4 = plt.figure(figsize=(12, 3))
        fig4.subplots_adjust(wspace=0.25)
        fig4.suptitle("LOS angles")
        
        los_tilt_angles = self.history["LOS_tilt"][:, :self.last_strike_index]
        los_yaw_angles = self.history["LOS_yaw"][:, :self.last_strike_index]
        z1__ = self.history["z1"][:, :self.last_strike_index]
        
        ax = fig4.add_subplot(2, 2, 1)
        ax.plot(tvec, los_tilt_angles[0, :]*180/np.pi, label="los tilt")
        ax.set_ylabel("los tilt")
        ax.grid()
        ax.legend()
        
        ax = fig4.add_subplot(2, 2, 2)
        ax.plot(tvec, los_yaw_angles[0, :]*180.0/np.pi, label="los yaw")
        ax.set_ylabel("los yaw")
        ax.grid()
        ax.legend()
        
        ax = fig4.add_subplot(2, 2, 3)
        ax.plot(tvec, z1__[0, :], label="z1")
        ax.set_ylabel("z1")
        ax.grid()
        ax.legend()
        
        ax1 = plt.figure().gca(projection='3d')
        ax1.plot(pos[0, :], pos[1, :], pos[2, :], zdir='z', label='path of the drone')
        
        
        
        print("drone position: ")
        x = self.history["p"][0, :]
        print(x.shape)
        
        plt.show()