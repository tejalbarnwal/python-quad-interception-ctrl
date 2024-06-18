import numpy as np
from quad_model import Quadrotor
from controller import Controller
from quad_sim import Simulator

# quad = Quadrotor(r= np.array([18.0, 0.0, 35.0]))
# thrust = np.linalg.norm(quad.mass * quad.g[2]) + 0.01
# print("thrust: ", thrust)
# Tstep = 0.01
# Tf = 20.0
# N = int(Tf/Tstep)
# print(quad)
# for i in range(N):
#     print("iteration: ", i)
#     u = np.array([thrust, 0.02, 0.0, 0.0])
#     quad.update(u, Tstep)
#     print("------------------------------------------")
# print(quad)

#######################################################

quad = Quadrotor(r= np.array([200.0, 25.0, 100.0]))
# quad = Quadrotor(r= np.array([99.0, 0.0, 35.0]))

print("Initial state of quadrotor: ")
print(quad)

ctrl = Controller()

sim = Simulator(quad, ctrl)

target_position = np.array([1.0, 2.0, 20.0])
# ntd = np.array([-0.93969262078, -0.0, -0.34202014332]) # 20 deg LOS
# ntd = np.array([-0.8660, -0.0, -0.5]) # 30 deg LOS
ntd = np.array([-0.7071, -0.0, -0.7071]) # 30 deg LOS



sim.run(target_pos=target_position, n_td=ntd, Tf=100.0, Ts=0.01)
sim.plot()


######
# x = 0.93969262078, z = 0.34202014332
####