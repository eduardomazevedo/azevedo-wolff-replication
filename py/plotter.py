# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:19:55 2024

@author: Ilan
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

class plotter:
    def __init__(self, prob):
        self.prob = prob
        
    def plot_optimal_results(self, ubar_grid, title, a_hat_init=[0], optimize_a0=False):
        """
        Plots the results of the optimization.
    
        Parameters:
        - ubar_grid: Array of ubar values.
        - title: Title of the plot.
        - a_hat_init: Initial guess for the parameters of global IC constraints.
        - optimize_a0: If True, solve for the optimal a0 first and then create plots.
        """
        lambda_values = []
        mu_values = []
        eta_values = []
        cost_values = []
        U_values = [] 
        a_hat_values = [] 
        a0_values = []
        ubar_values = [] 
    
        # Perform the optimization for each ubar in the grid
        for ubar in ubar_grid:  
            
            self.prob.ubar = ubar 
            
            # Optionally solve for the optimal a0
            if optimize_a0:
                self.prob.a0 = self.prob.maximize_profit(a_hat_init, flexible_a=True)
                
            (con, optimal_cost, optimal_U) = self.prob.maximize_dual(a_hat_init, True) 
            
            ubar_values.append(ubar)
            lambda_values.append(con.lmda)
            mu_values.append(con.mu)
            eta_values.append(con.eta)
            cost_values.append(optimal_cost)
            U_values.append(optimal_U)
            a0_values.append(self.prob.a0)  
            a_hat_values.append(con.a_hat)
    
        # create plot 
        plt.figure(figsize=(14, 18))
        
        # Set the main title for the entire figure
        plt.suptitle(title, fontsize=16)
        
        # Plot Optimal Lambda
        plt.subplot(3, 2, 1)
        plt.plot(ubar_values, lambda_values, label='Lambda', marker='*', color='purple')
        plt.xlabel('Ubar')
        plt.ylabel('Lambda')
        plt.legend()
        plt.grid(True)
        
        # Plot Optimal a
        plt.subplot(3, 2, 2)
        plt.plot(ubar_values, a0_values, label='Optimal a', marker='o', color='blue')
        plt.xlabel('Ubar')
        plt.ylabel('Optimal a')
        plt.legend()
        plt.grid(True)
        
        # Plot Optimal Mu
        plt.subplot(3, 2, 3)
        plt.plot(ubar_values, mu_values, label='Optimal Mu', marker='s')
        plt.xlabel('Ubar')
        plt.ylabel('Optimal Mu')
        plt.legend()
        plt.grid(True)
        
        # Plot Optimal Eta
        plt.subplot(3, 2, 4)
        plt.plot(ubar_values, eta_values, label='Optimal Eta', marker='d', color='orange')
        plt.xlabel('Ubar')
        plt.ylabel('Optimal Eta')
        plt.legend()
        plt.grid(True)
        
        # Plot Optimal Cost
        plt.subplot(3, 2, 5)
        plt.plot(ubar_values, cost_values, label='Optimal Cost', marker='x', color='green')
        plt.xlabel('Ubar')
        plt.ylabel('Optimal Cost')
        plt.legend()
        plt.grid(True)
        
        # Plot Optimal U
        plt.subplot(3, 2, 6)
        plt.plot(ubar_values, U_values, label='Optimal U', marker='^', color='red')
        plt.xlabel('Ubar')
        plt.ylabel('Optimal U')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
        ()
            
        results_dict = {
        'lambda': lambda_values,
        'mu': mu_values,
        'eta': eta_values,
        'cost': cost_values,
        'U': U_values,
        'a0': a0_values, 
        'a_hat': a_hat_values 
        }

        return results_dict

    def plot_func_effort(self, vv, func, yaxis="", title="", aa_range=(0, 10)):
        aa_values = np.linspace(aa_range[0], aa_range[1], 100)
        func_values = [func(vv, aa) for aa in aa_values]
        
        plt.figure()
        plt.plot(aa_values, func_values)
        plt.xlabel('Effort (aa)')
        plt.ylabel(yaxis)
        plt.title(title)
        plt.grid(True)
        () 
        
    def plot_func_outcome(self, func, yaxis="", title="", yy_range=(0, 10), density=False):
        
        if self.prob.error.error_type=='continuous': 
            yy_values = np.linspace(yy_range[0], yy_range[1], 100)
            
            # Compute function values for the yy range
            func_values = [func(yy) for yy in yy_values]
            
            fig, ax1 = plt.subplots()  # Create a figure and an axis (ax1)
        
            # Plot the function output on the first y-axis
            color = 'tab:blue'
            ax1.set_xlabel('Outcome (yy)')
            ax1.set_ylabel(yaxis, color=color)
            ax1.plot(yy_values, func_values, label=yaxis, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
        
            # If density is enabled, plot phi_values on a second y-axis
            if density:
                ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
                color = 'tab:red'
                phi_values = [self.prob.error.phi(yy, self.prob.a0) for yy in yy_values]
                ax2.set_ylabel('Density', color=color)  # Set label for the second y-axis
                ax2.plot(yy_values, phi_values, label="Density", color=color)
                ax2.tick_params(axis='y', labelcolor=color)
        
            fig.tight_layout()  # Adjust the layout to make room for both y-axis labels
            plt.title(title)
            plt.grid(True)
            ()
            
        # Discrete case
        elif self.prob.error.error_type == 'discrete':
            
            # Generate discrete y-values using arange
            yy_values = np.arange(yy_range[0], yy_range[1] + 1)
            
            # Compute function values for each discrete yy value
            func_values = [func(yy) for yy in yy_values]
            
            # Create the bar plot
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Outcome (yy)')
            ax1.set_ylabel(yaxis, color=color)
            ax1.bar(yy_values, func_values, label=yaxis, color=color, alpha=0.5)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xticks(yy_values) 
            
            # If density is enabled, plot density as well
            if density:
                ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
                color = 'tab:red'
                phi_values = [self.prob.error.phi(yy, self.prob.a0) for yy in yy_values]
                ax2.set_ylabel('Density', color=color) 
                ax2.bar(yy_values, phi_values, label="Density", color=color, alpha=.5)
                ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()
            plt.title(title)
            plt.grid(True)
            () 
                    
    def plot_func_effort_grid(self, vv, funcs, yaxes, titles, main_title="", aa_range=(0, 10)):
        """
        Plots multiple functions as a function of effort in a 2x2 grid.
    
        Parameters:
        - vv: The value function (e.g., contract).
        - funcs: List of functions to plot (e.g., [prob.exp_U, prob.util_grad, ...]).
        - yaxes: List of y-axis labels corresponding to each function.
        - titles: List of titles corresponding to each plot.
        - aa_range: Tuple specifying the range of effort (aa) values.
        - main_title: Main title for the entire figure.
        """
        aa_values = np.linspace(aa_range[0], aa_range[1], 100)
        
        # Create a 2x2 grid plot
        plt.figure(figsize=(14, 12))
        
        # Set the main title for the entire figure
        if main_title:
            plt.suptitle(main_title, fontsize=16)
        
        for i, func in enumerate(funcs):
            func_values = [func(vv, aa) for aa in aa_values]
            plt.subplot(2, 2, i+1)
            plt.plot(aa_values, func_values)
            plt.xlabel('Effort (aa)')
            plt.ylabel(yaxes[i])
            plt.title(titles[i])
            plt.grid(True)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
        ()
    
    def plot_pareto_frontier(self, ubar_grid, title="", path=""):
        # Initialize storage for two cases (relaxed and partially relaxed)
        optimal_costs = [[], []]
        deviations = [[], []]
        optimal_actions = [[], []]  # Store the optimal actions for plotting
        
        for ubar in ubar_grid: 
            # set reservation utility 
            self.prob.ubar = ubar 
            
            # Perform cost minimization for relaxed and partially relaxed problems
            for i, (a_hat, param) in enumerate([([], False)]):  # , ([0,.5], True)]):
                (con, optimal_cost, optimal_U) = self.prob.maximize_dual(a_hat, param)  
                                    
                # Append optimal costs 
                optimal_costs[i].append(optimal_cost)
                
                # Check for global deviations 
                a_grid = np.linspace(0.01, self.prob.a0*2, 201)  # Grid of deviations  
                global_utils = self.prob.exp_U_vectorized(con.value_function, a_grid)  # Grid of utilities 
                deviations[i].append(optimal_U - max(global_utils))  # Store the deviation
                
                # Record corresponding action (action from a_grid that gives max utility)
                max_index = np.argmax(global_utils)
                optimal_actions[i].append(a_grid[max_index])  # Store the optimal action
    
        # Normalize the deviations between -0.1 and 0 for color mapping (finer scale)
        norm = Normalize(vmin=-0.1, vmax=0)
    
        # Create a continuous colormap from red to green
        cmap = cm.get_cmap('Spectral')
    
        # Map the deviations to the colormap
        colors_0 = [cmap(norm(dev)) for dev in deviations[0]]
        
        # Create subplots: one for the costs and another for the optimal actions
        fig, ax = plt.subplots(2, 1, figsize=(8, 10)) 
        
        # Scatter plot with color-coded deviations for both cases using continuous color map
        scatter = ax[0].scatter(ubar_grid, optimal_costs[0], c=deviations[0], cmap=cmap, norm=norm, marker='o', zorder=2)
        
        # Add a color bar to show the deviation scale
        cbar = plt.colorbar(scatter, ax=ax[0])
        
        # Adjust the color bar ticks to represent the desired range [-0.1, 0]
        cbar.set_ticks([-0.1, -0.05, 0])
        cbar.set_label('Deviation from Global Utility')
    
        # Top subplot: plot for optimal costs (for both scenarios)
        ax[0].plot(ubar_grid, optimal_costs[0], color='gray', linestyle='--', label='Relaxed Cost Trend', zorder=1)
    
        ax[0].set_xlabel('Reservation Utility')
        ax[0].set_ylabel('Optimal Cost') 
        ax[0].set_title(title)
        
        # Bottom subplot: plot for optimal actions (for both scenarios)
        ax[1].plot(ubar_grid, optimal_actions[0], marker='o', linestyle='--', color='blue', label='Relaxed Optimal Actions')
        
        # Add a horizontal line at prob.a0 for reference
        ax[1].axhline(self.prob.a0, color='purple', linestyle='--', label=f'Target Action')
        ax[1].set_ylim(0, 2 * self.prob.a0)
        ax[1].legend()
        ax[1].set_xlabel('Reservation Utility')
        ax[1].set_ylabel('Optimal Actions')
        ax[1].set_title('Optimal Action from Global Deviation Grid')
    
        # Save and show plots
        plt.tight_layout() 
        if path: 
            plt.savefig(f'{path}.png', bbox_inches='tight') 
        ()

        
        
