import numpy as np

def rk4(f, y, dt):
    """Runge-Kutta 4th Order
    
    Solves an autonomous (time-invariant) differential equation of the form dy/dt = f(y).
    """
    k1 = f(y)
    k2 = f(y + dt/2 * k1)
    k3 = f(y + dt/2 * k2)
    k4 = f(y + dt * k3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

class Simulation:
    def __init__(self):
        self.v = np.array([1.0, 2.0, 3.0])
        self.r = np.array([0.0, 0.0, 0.0])

    def update_position(self, dt):
        # Define the lambda function that returns the velocity
        fun1 = lambda r: self.v
        # Use rk4 method to update position based on constant velocity
        self.r = rk4(fun1, self.r, dt)
        print("r: ", self.r)

# Example usage
sim = Simulation()
sim.update_position(0.1)
sim.update_position(0.1)
sim.update_position(0.1)