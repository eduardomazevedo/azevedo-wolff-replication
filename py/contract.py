# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:36:01 2024

@author: Ilan
"""

class contract:
    def __init__(self, lmda, mu, eta, a_hat, v_star_func, u_inverse_func, w0):
        self.lmda = lmda
        self.mu = mu
        self.eta = eta
        self.a_hat = a_hat
        self.v_star_func = v_star_func
        self.u_inverse_func = u_inverse_func
        self.w0 = w0

    def value_function(self, yy):
        # Calls the provided v_star function using the contract's parameters
        return self.v_star_func(yy, self.lmda, self.mu, self.eta, self.a_hat)

    def wage_function(self, yy): 
        return self.u_inverse_func(self.value_function(yy)) - self.w0
    
    def __str__(self):
       return (f"Contract Class:\n"
               f"  Ubar Multiplier (lmda): {self.lmda}\n"
               f"  Local IC Multiplier (mu): {self.mu}\n"
               f"  Global IC Multipliers (eta): {self.eta}\n"
               f"  Alternative Actions (a_hat): {self.a_hat}\n"
               f"  Initial Wealth (w0): {self.w0}\n")

