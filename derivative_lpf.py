import numpy as np

class DirtyDerivative:
    """Dirty Derivative
    
    Provides a first-order derivative of a signal.
    
    This class creates a filtered derivative based on a
    band-limited low-pass filter with transfer function:
    
        G(s) = s/(tau*s + 1)
        
    This is done because a pure differentiator (D(s) = s)
    is not realizable.    
    """
    def __init__(self, order=1, tau=0.05):
        # time constant of dirty-derivative filter.
        # Higher leads to increased smoothing.
        self.tau = tau
        
        # Although this class only provides a first-order
        # derivative, we use this parameter to know how
        # many measurements to ignore so that the incoming
        # data is smooth and stable. Otherwise, the filter
        # would be hit with a step function, causing
        # downstream dirty derivatives to be hit with very
        # large step functions.
        self.order = order
        
        # internal memory for lagged signal value
        self.x_d1 = None
        
        # Current value of derivative
        self.dxdt = None
        
    def update(self, x, Ts):
        # Make sure to store the first `order` measurements,
        # but don't use them until we have seen enough
        # measurements to produce a stable output
        if self.order > 0:
            self.order -= 1
            self.x_d1 = x
            return np.zeros(x.shape)        
        
        # Calculate digital derivative constants
        a1 = (2*self.tau - Ts)/(2*self.tau + Ts)
        a2 = 2/(2*self.tau + Ts)
        
        if self.dxdt is None:
            self.dxdt = np.zeros(x.shape)
        
        # calculate derivative
        self.dxdt = a1*self.dxdt + a2*(x - self.x_d1)
        
        # store value for next time
        self.x_d1 = x
                
        return self.dxdt


        