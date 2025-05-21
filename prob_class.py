# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:52:30 2024

@author: Ilan
"""
from scipy.integrate import simpson
import numpy as np 
from plotter import plotter
import inspect 
from err_class import err

# define the problem class 
class prob:
    def __init__(self,cost='quad',
                 error='norm',u='log',
                 w0=1,ubar=0,theta=.05,alpha=2,gamma=.5,n=None) :
        
        """ explanation of inputs 
        cost: type of cost function 
              options: quad, cost function of the form theta*a^{alpha}
        theta: see cost 
        alpha: see cost 
        
        error: type of distribution y is drawn from. 
               options: normal, log normal, exponential, poisson, binomial, gamma, geometric 
        n: spare input to error functions with extra parameters (see binomial)
        
        u: type of utility function. 
           options: log, crra, cara 
           gamma: parameter in utility function 
           
        # other params 
        w0: starting wealth 
        ubar: reservation utility 
        """ 
        
        """ 
        Cost functions 
        c: agent's cost function
        mc: agent's marginal cost function 
        latex_cost: Print out of the cost function 
        """ 
        
        if cost == 'quad': 
            self.c = lambda aa: theta * (aa ** alpha)  
            self.mc = lambda aa: alpha * theta * (aa ** (alpha-1))  
            self.latex_cost = f'{theta} a^{{{alpha}}}'
        
        # instantiate the error, see err_class for more information 
        if isinstance(error, str):
            self.error = err(default_option=error,n=n)
        elif isinstance(error, dict):
            self.error = err(args=error,n=n)
        elif isinstance(error, err):  # Check if 'error' is an instance of the 'err' class
            self.error = error
        else:
            raise TypeError("Invalid type for 'error'. Expected a string, dictionary, or an 'err' instance.")
            
        """ 
        Utility Functions
        u: the utility function 
        mu: marginal utility w.r.t consumption 
        u_inverse: x(u), or the amount of consumption needed to achieve u 
        k_prime_inverse: *hard to explain in short note*, called g in current notation 
        latex_u: Clean print out of utility function. 
        """ 
        
        if u=='log': 
            self.u = lambda xx: np.log(xx) 
            self.mu = lambda xx: 1/xx 
            self.u_inverse = lambda vv: np.exp(vv) 
            self.k_prime_inverse = lambda ll: np.log(ll) 
            self.latex_u = 'log(x)' 
            
        if u=='crra': 
            self.u = lambda xx: xx**(1 - gamma) / (1 - gamma) if gamma != 1 else np.log(xx)
            self.mu = lambda xx: xx**(-gamma)
            self.u_inverse = lambda vv: (vv * (1 - gamma))**(1 / (1 - gamma)) if gamma != 1 else np.exp(vv)
            self.k_prime_inverse = lambda ll: (1 / (1 - gamma))*(ll)**((1 - gamma) / gamma) if gamma != 1 else np.log(ll)
            self.latex_u = f'\\frac{{x^{{{1 - gamma}}}}}{{{1 - gamma}}}' 
            
        if u=='cara': 
            self.u = lambda xx: -np.exp(-gamma*xx)*(1/gamma)
            self.mu = lambda xx: np.exp(-gamma*xx) 
            self.u_inverse = lambda vv: np.log(-vv*gamma)/-gamma 
            self.k_prime_inverse = lambda ll: -1/(gamma*ll) 
            self.latex_u = f'\\frac{{- e^{{-{gamma} x}}}}{{{gamma}}}'

        # assign parameters 
        self.w0 = w0 
        self.ubar = ubar 
        
        # Initialize an instance of plotter. This class analyzes and plot
        # result for the problem 
        self.plotter = self.plotter(self)
  
    # Can possibly retire this at some point and replace with exp_U_vectorized
    def exp_U(self, vv, aa, num_points=100000):
        """
        Compute the expected utility using the value function vv and effort aa.
        
        Parameters:
        vv (function): Value function to integrate.
        aa (float): Effort level.
        num_points (int): Number of points to use for integration.
    
        Returns:
        float: Expected utility.
        """ 
                
        # Create an array of yy values around the mean aa, within a certain range
        yy_values = self.error.y_grid_func(aa,num_points).flatten() 
        
        integrand_values = vv(yy_values) * self.error.phi(yy_values, aa)  
                    
        if self.error.error_type=='continuous': 

            # Use Simpson's rule for the integration
            return simpson(integrand_values, yy_values) - self.c(aa) 
        
        elif self.error.error_type=='discrete': 
            
            # return expected value minus cost 
            return sum(integrand_values) - self.c(aa)
    
    def exp_U_vectorized(self, vv, aa_array, num_points=10000):
        """
        Vectorized version of exp_U to handle array input for aa without explicit loops.
    
        Parameters:
        vv (function): Value function to integrate.
        aa_array (array): Array of effort levels.
        num_points (int): Number of points to use for integration.
    
        Returns:
        array: Expected utility for each aa.
        """
        # Ensure aa_array is a NumPy array
        aa_array = np.atleast_1d(aa_array)
        
        # Generate yy_values for all aa in aa_array (num_points for each aa)
        yy_values = self.error.y_grid_func(aa_array, num_points)  # Shape: (len(aa_array), num_points)
        
        # Compute the integrand values for all aa and yy simultaneously using broadcasting
        integrand_values = vv(yy_values) * self.error.phi(yy_values, aa_array)  # Shape: (len(aa_array), num_points)
        
        if self.error.error_type=='continuous': 
        # Perform Simpson's integration across the yy axis (axis=1) for each aa in the array
            return simpson(integrand_values.T, yy_values.T, axis=1) - self.c(aa_array) # Shape: (len(aa_array),)
        
        elif self.error.error_type=='discrete': 
            
            # return expected value minus cost 
            return sum(integrand_values) - self.c(aa_array)
        
    def wage_util(self, vv): 
        """ 
        vv: utility achieved at outcome y 
        returns the wage the agent is paid at outcome y 
        """ 
        return self.u_inverse(vv) - self.w0   
    
    def wage_util_func(self, vv): 
        """
        vv: value function 
        returns a wage function 
        """  
        return lambda yy: self.u_inverse(vv(yy)) - self.w0 
    
    def util_grad(self, vv, aa,num_points=10000): 
        """
        Compute the gradient of utility with respect to effort aa.

        Parameters:
        vv (function): Value function to integrate.
        aa (float): Effort level.

        Returns:
        float: Gradient of the utility function.
        """
        # Create an array of yy values around the mean aa, within a certain range
        yy_values = self.error.y_grid_func(aa,num_points).flatten() 
        
        # Compute the integrand for all yy values simultaneously (vectorized operation)
        integrand_values = vv(yy_values) * self.error.dphi(yy_values, aa) 
                
        # Use a vectorized integration method like Simpson's rule for the integration
        if self.error.error_type=='continuous': 

            return simpson(integrand_values, yy_values) - self.mc(aa)
    
        elif self.error.error_type=='discrete': 
            
            return sum(integrand_values) - self.mc(aa)
    
    def exp_wage(self, vv, aa, num_points=10000):
        
        """"
        vv (function): Value function to integrate.
        """ 
        
        # Create an array of yy values around the mean aa, within a certain range
        yy_values = self.error.y_grid_func(aa,num_points).flatten()  
                
        integrand_values = self.wage_util(vv(yy_values)) * self.error.phi(yy_values, aa) 

        # Use a vectorized integration method like Simpson's rule for the integration
        if self.error.error_type=='continuous': 

            return simpson(integrand_values, yy_values)
        
        elif self.error.error_type=='discrete': 
            
            return sum(integrand_values)
    
    def profit(self, vv, aa): 
        """ 
        Computes principle's expected profit for a
        vv: value function 
        aa: agent's induced action 
        """  
        return self.error.expected_revenue(aa) - self.exp_wage(vv, aa) 
    
    # Define a helper function to get the source of lambda functions
    def function_source(self, func):
        try:
            return inspect.getsource(func).strip()
        except OSError:
            return "<built-in function>"

    # Define the __str__ method for printing the class attributes and functions
    def __str__(self):
        
        return (f"Problem class with parameters:\n"
                f"  Utility function (u): {self.latex_u}\n"
                f"  Cost function (c): {self.latex_cost}\n"
                f"  Error: {self.error.print_error}\n"
                f"  Initial Wealth (w_0): {self.w0}\n"
                f"  Reservation Utility (ubar): {self.ubar}\n") 
    
    class plotter(plotter):
        def __init__(self, prob):
            super().__init__(prob) 
