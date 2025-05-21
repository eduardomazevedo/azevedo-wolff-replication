# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:02:54 2024

@author: Ilan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:53:12 2024

@author: Ilan
"""
import numpy as np
from scipy.optimize import minimize, Bounds
import copy, warnings 

# import the prob_class 
from prob_class import prob  # Now you can import prob from the parent directory
from contract import contract 

# define the cost_min_prob class 
class cost_min_prob(prob):
    def __init__(self,cost='quad',
                 error='norm',u='log',
                 w0=1,ubar=0,theta=.05,alpha=2,a0=4,gamma=.5,n=None): 
        
        """ 
        This section catches problems with invalid parameters, and 
        prompts the use to choose a valid parameter. 
        """ 
        
        message = None 
        
        if a0 <= 0: 
            message = "The target action must be > 0"
        
        if error in ['binomial','bernoulli'] and a0 > 1:
            message = (f"Invalid target action. In the {error} distribution, "
                       "'a0' is the probability of success and must be between 0 and 1. "
                       f"Your a_0={a0}.") 
            
        if error in ['geometric'] and a0 < 1: 
            message = (f"Invalid target action. In the {error} distribution, "
                       "'1/a0' is the probability of failure and must be between 0 and 1. "
                       f"Your 1/a_0={1/a0}.") 
            
        if message: 
            raise ValueError(message)   
        """ *********************************************************** """ 
        
        # Call the parent class (prob) constructor
        super().__init__(cost,error,u,w0,ubar,theta,alpha,gamma,n) 
        
        # Additional initialization specific to cost_min_prob
        self.a0 = a0  # Minimizing cost to achieve a0
        
        self.min_v = self.u(self.w0)  # Minimum value for v self.u(self.w0)
        self.min_k_prime = 1 / self.mu(self.w0)  # Minimum value for (1/self.du(self.w0))
        
    def h_local(self, vv): 
        
        """"
        vv (function): Agent's value function. 
        Computes the gradient of the value function at effort level a0  
        """
        
        return  -self.util_grad(vv, self.a0) 
    
    def h_global(self, vv, a_hat): 
        if a_hat == self.a0: 
            return self.h_local(vv) # why not 0? This adds the local constraint twice? 
        else: 
            num = self.exp_U(vv,a_hat) - self.exp_U(vv,self.a0) # the constraint  
            denom = abs(a_hat-self.a0) # I don't understand this scaling term 
            return (num/denom)

    def v_star(self, yy, lmda, mu, eta, a_hat): 
        """
        ADD FUNCTION NOTE HERE 
        """ 
        phi0 = self.error.phi(yy, self.a0) # probability of outcome y from normal with mean a0  
        dphi0 = self.error.dphi(yy, self.a0) # derivative of outcome prob w.r.t effort 
        
        # IC and IR constraints
        kp = lmda + mu*(dphi0/phi0) # if invalid divide warning, sets nans to 0 
    
        # fillna with zero in kp 
        kp = np.nan_to_num(kp, nan=0.0) 
        
        # Add global constraints to lagrangian 
        for i in range(len(eta)): 
             phi_a_hat = self.error.phi(yy, a_hat[i])
             num = eta[i] * (1 - (phi_a_hat / phi0)) 
             #print(f'phi_hat: {phi_a_hat}, phi_0: {phi0}, eta: {eta[i]}')
             denom = abs(a_hat[i]-self.a0) 
             kp = kp + (num/denom)  
                
        # Set to min kprime when limited liability constraint binds 
        kp = np.maximum(kp, self.min_k_prime) 
        
        # or what this does 
        return self.k_prime_inverse(kp)   
    
    def lagrangean(self, vv, lmda, mu, eta, a_hat, flex_lambda=True): 
        
        # lagrangean with IR and local IC constraints 
        if flex_lambda: 
            ll = (self.exp_wage(vv,self.a0) - lmda*(self.exp_U(vv, self.a0)-self.ubar) +
                                               mu*self.h_local(vv))        
        else: 
            ll = self.exp_wage(vv,self.a0) - lmda*self.exp_U(vv, self.a0) + mu*self.h_local(vv)
        
        # Add in global IC constraints 
        for i in range(len(eta)): 
            ll = ll + eta[i] * self.h_global(vv, a_hat[i]) 
        
        return ll 

    def lagrange_dual(self, lmda, mu, eta, a_hat, flex_lambda=True):
        
        # Create an anonymous function v_star as a function of yy
        v_star_func = lambda yy: self.v_star(yy, lmda, mu, eta, a_hat)
        
        # Return a function that computes the Lagrangian given yy
        return self.lagrangean(v_star_func, lmda, mu, eta, a_hat, flex_lambda) 
    
    def maximize_dual(self, a_hat_initial=[], flexible_a=False, lambda_in=[True]):
        """
        Maximizes the dual Lagrangian function with the option for flexible or fixed a_hat.

        Parameters:
        lmda (float): Lagrange multiplier for the IR constraint.
        a_hat_initial (array-like): Initial guess for the parameters of global IC constraints.
        flexible_a (bool): If True, a_hat is optimized; if False, a_hat is fixed.
        lambda: If true, optimizes lambda. 
        
        Returns:
        tuple: optimal_mu, optimal_eta, optimal_a_hat, optimal_cost, optimal_U
        """
        n_a_hat = len(a_hat_initial)
        flex_lambda = lambda_in[0]
        
        # compute the number of variables. booleans are 1 if true, false otherwise
        num_mults = 1 + flex_lambda
        n_vars = num_mults + (1+flexible_a) * n_a_hat  # Number of variables (mu, eta, a_hat)

        # Indices for eta and a_hat in the optimization vector
        if n_a_hat == 0:
            index_eta = []
            index_a_hat = []
        else:
            index_eta = list(range(num_mults, num_mults + n_a_hat))
            index_a_hat = list(range(num_mults + n_a_hat, n_vars)) if flexible_a else []

        # Lower bounds: 0 for lmda, -inf for mu, 0 for each eta, and greater than 0 for a_hat if flexible
        lb = [0]*flex_lambda + [-np.inf] + [0] * n_a_hat + [1e-3] * flexible_a*n_a_hat  # a_hat must be > 0 
        ub = [np.inf] * len(lb)  # No upper bounds
        bounds = Bounds(lb, ub)  # initialize bounds 
        
        # Initial guess for the optimization (x0)
        x0 = np.zeros(n_vars)

        # Objective function
        if flex_lambda:
            def objective(x): 
                lmda = x[0]
                mu = x[1]
                eta = x[index_eta]
                a_hat = x[index_a_hat] if flexible_a else a_hat_initial 
                return -self.lagrange_dual(lmda, mu, eta, a_hat, flex_lambda) 
        else: 
            def objective(x): 
                lmda = lambda_in[1] # second input to lambda_in specifies lmda value
                mu = x[0]
                eta = x[index_eta]
                a_hat = x[index_a_hat] if flexible_a else a_hat_initial 
                return -self.lagrange_dual(lmda, mu, eta, a_hat, flex_lambda) 

        # Minimize the negative of the dual Lagrangian
        result = minimize(objective, x0, bounds=bounds, method='SLSQP')

        # Extract the results
        x_opt = result.x
        if flex_lambda: 
            optimal_lmda = x_opt[0]
            optimal_mu = x_opt[1]
        else: 
            optimal_lmda = lambda_in[1] # return the input lambda value  
            optimal_mu = x_opt[0]
        optimal_eta = x_opt[index_eta] if n_a_hat > 0 else []
        optimal_a_hat = x_opt[index_a_hat] if flexible_a else a_hat_initial

        # create the contract 
        con = contract(optimal_lmda, optimal_mu, optimal_eta, optimal_a_hat, 
               self.v_star, self.u_inverse, self.w0)

        # Compute the optimal cost and utility using the optimal strategy
        optimal_cost = self.exp_wage(con.value_function, self.a0)
        optimal_U = self.exp_U(con.value_function, self.a0) 
        
        # return contract, cost and utility 
        return con, optimal_cost, optimal_U
        
    def maximize_profit(self, a0_grid, a_hat_initial=[], flexible_a=False):
        """
        Maximizes the profit by optimizing over a0, given initial guess for a_hat.
    
        Parameters:
        - a0_grid (array-like): Grid of possible values for a0.
        - a_hat_initial (array-like): Initial guess for the parameters of global IC constraints.
        - flexible_a (bool): If True, a_hat is optimized; if False, a_hat is fixed.
    
        Returns:
        - optimal_a: The optimal effort level a0.
        - optimal_profit: The corresponding maximum profit.
        - optimal_cost: The cost associated with the optimal effort.
        """
        # Copy the problem instance to avoid modifying the original
        prob_copy = copy.deepcopy(self)
    
        def find_optimal_in_grid(prob_copy, grid):
            """Finds the best action in the given grid."""
            max_profit = -float('inf')
            optimal_a = None
    
            for a in grid:
                prob_copy.a0 = a  
                con, optimal_cost, optimal_U = prob_copy.maximize_dual(a_hat_initial, flexible_a)
                temp_profit = prob_copy.error.expected_revenue(a) - optimal_cost
    
                if temp_profit > max_profit:
                    max_profit = temp_profit
                    optimal_a = a
    
            return optimal_a, max_profit
    
        # Step 1: Coarse search in a0_grid
        optimal_a, optimal_profit = find_optimal_in_grid(prob_copy, a0_grid)
    
        # Step 2: Identify neighboring points in a0_grid
        index_optimal = np.where(a0_grid == optimal_a)[0] 
    
        # Handle boundary cases
        if index_optimal == 0:
            warnings.warn(f"Optimal action {optimal_a} is at the lower bound of a0_grid. Refinement may be limited.")
            lower_bound, upper_bound = a0_grid[0], a0_grid[1]
        elif index_optimal == len(a0_grid) - 1:
            warnings.warn(f"Optimal action {optimal_a} is at the upper bound of a0_grid. Refinement may be limited.")
            lower_bound, upper_bound = a0_grid[-2], a0_grid[-1]
        else:
            lower_bound, upper_bound = a0_grid[index_optimal - 1], a0_grid[index_optimal + 1]
    
        # Step 3: Create a finer grid within [lower_bound, upper_bound]
        fine_grid = np.linspace(lower_bound, upper_bound, num=20)
    
        # Step 4: Refined search in the fine grid
        optimal_a, optimal_profit = find_optimal_in_grid(prob_copy, fine_grid)
    
        # Compute the optimal cost for the final optimal_a
        prob_copy.a0 = optimal_a
        _, optimal_cost, _ = prob_copy.maximize_dual(a_hat_initial, flexible_a)
    
        return optimal_a, optimal_profit, optimal_cost

    # print function 
    def __str__(self):
        # Call the parent class __str__ method to get its string representation
        base_str = super().__str__()
        # Add a0 to the string
        return base_str + f"  Target Effort (a0): {self.a0}\n"
      
    def __eq__(self, other):
        if not isinstance(other, prob):
            return False
        return (self.latex_u, self.w0, self.ubar, self.latex_cost, self.error.print_error) == \
               (other.latex_u, other.w0, other.ubar, other.latex_cost, other.error.print_error)

    def __hash__(self):
        return hash((self.latex_u, self.w0, self.ubar, self.latex_cost, self.error.print_error))
    

""" 
old profit function 

# need to update profit function for error distributions that don't have mean a
# Define the profit function to maximize
def profit_func(a0):
    prob_copy.a0 = a0  # Update a0 in the problem copy
    (con, optimal_cost, optimal_U) = prob_copy.maximize_dual(a_hat_initial, flexible_a)

    # minimize negative profit 
    return -(prob_copy.error.expected_revenue(a0) - optimal_cost)  

# equation for full monitor effort to use as a right bound on optimal effort 
def mc_equation(e):
    return self.mc(e) - 1 # this assumes expected revenue is a_0 
 
# Solve using Newton's method
max_e = root_scalar(mc_equation, x0=self.a0, method='newton').root
 
# Use a scalar minimization over a0 to find the optimal profit
prob_copy.a0 = minimize_scalar(profit_func, bounds=(0, max_e), method='bounded').x # mutate a0 in the copy 

(con, optimal_cost, optimal_U) = prob_copy.maximize_dual(a_hat_initial, flexible_a) 
optimal_profit = prob_copy.error.expected_revenue(prob_copy.a0) - optimal_cost  """ 


