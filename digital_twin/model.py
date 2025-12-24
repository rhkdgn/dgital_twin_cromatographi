import numpy as np
from scipy.integrate import odeint

class VirtualColumn:
    def __init__(self, N=50, V_total=10.0, Q=1.0):
        self.N = N 
        self.V_tank = V_total / N 
        self.Q = Q
        
        self.k_target = 5.0    
        self.k_impurity = 3.0  
        self.aging_factor = 1.0 

    def model_derivative(self, C, t):
        C_target = C[:self.N]
        C_impurity = C[self.N:]
        

        if 0.5 <= t <= 1.0:
            Cin_target = 100.0   
            Cin_impurity = 50.0  
        else:
            Cin_target = 0.0
            Cin_impurity = 0.0
            
        dCdt_target = np.zeros(self.N)
        dCdt_impurity = np.zeros(self.N)
        
        eff_factor_target = 1 + (self.k_target * self.aging_factor)
        eff_factor_impurity = 1 + (self.k_impurity * self.aging_factor)
        
        V_eff_target = self.V_tank * eff_factor_target
        V_eff_impurity = self.V_tank * eff_factor_impurity

        dCdt_target[0] = (self.Q / V_eff_target) * (Cin_target - C_target[0])
        dCdt_impurity[0] = (self.Q / V_eff_impurity) * (Cin_impurity - C_impurity[0])
        
        for i in range(1, self.N):
            dCdt_target[i] = (self.Q / V_eff_target) * (C_target[i-1] - C_target[i])
            dCdt_impurity[i] = (self.Q / V_eff_impurity) * (C_impurity[i-1] - C_impurity[i])
            
        return np.concatenate([dCdt_target, dCdt_impurity])

    def run_simulation(self, t_max=100, dt=0.1):
        t = np.arange(0, t_max, dt)
        C0 = np.zeros(2 * self.N)
        
        result = odeint(self.model_derivative, C0, t, hmax=0.05)
        
        out_target = result[:, self.N - 1]
        out_impurity = result[:, 2*self.N - 1]
        
        return t, out_target, out_impurity