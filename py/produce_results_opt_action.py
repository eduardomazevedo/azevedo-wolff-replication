# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:01:26 2025

@author: Ilan
"""
from .cost_min import cost_min_prob 
import numpy as np  
import matplotlib.pyplot as plt
from scipy.integrate import simpson  
from matplotlib.ticker import FuncFormatter 
import dill 
import os 
import copy  

# nice color scheme 
ghibli_line_colors = [
    "#4C413F",  # Muted charcoal
    "#278B9A",  # Teal
    "#D8AF39"   # Golden mustard
]

# Default tick formatting function
def dol_20k_formatter(value, _):
    """Default: Format axis ticks as dollar amounts in units of $20,000."""
    if value > 0:
        return f"${int(value * 20)}K"
    else:
        return f"${value}"  

def create_combined_plot(prob_in, xx_range, yy_range, ubars, grid_a0, answer_key={},  
                         plot_type="util", xlab="", ylab="", title="", legend_size=10, 
                         grid=None, output_filename=None, 
                         tick_formatter_x=None, tick_formatter_y=None, dist=False, opt_a=True):
    """
    Creates a single plot with all lines, emphasizing the first, middle, and last values of ubar. 
    arguments: prob_in: the problem to solve 
    xx_range: the range of values on x axis to plot (can be outcome for wage plots or action for util plots) 
    yy_range: the range of values on y axis to plot (wage for wage plots, util for util plots) 
    ubars: list of reservation utilities to solve the problem for 
    answer_key: dictionary of solutions to already solved problems, only cooperates if opt_a=True
    plot_type: option for plot_type (util or wage) 
    opt_a: optimize action or treat as cost minimization problem 
    """
    
    xx_values = np.linspace(xx_range[0], xx_range[1], 100) 

    # Identify first, middle, and last ubars
    first_ubar = ubars[0]
    middle_ubar = ubars[len(ubars) // 2]
    last_ubar = ubars[-1]
    special_ubars = {first_ubar, middle_ubar, last_ubar}

    if grid is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    else:
        ax = grid

    efforts = [] 
    opt_us = [] 
    for ubar in ubars: 
                
        # update utility 
        prob = copy.deepcopy(prob_in)
        prob.ubar = ubar  
        
        #print(prob)
        
        if opt_a : 
        
            # check if we have already recorded the solution to this problem 
            if (prob.latex_u, prob.w0, prob.ubar, 
                prob.latex_cost, prob.error.print_error) in answer_key.keys():  
                            
                effort, opt_profit, con, opt_u = answer_key[
                    (prob.latex_u, prob.w0, prob.ubar, prob.latex_cost, prob.error.print_error)] 
                
            else: 
                
                # compute optimal actions 
                effort, opt_profit, cost = prob.maximize_profit(grid_a0, a_hat_initial=[1e-2],flexible_a=False) 
            
                # solve unconstrained minimization problem 
                prob.a0 = effort # use optimal effort 
                con, opt_cost, opt_u = prob.maximize_dual(a_hat_initial=[1e-2],flexible_a=False)
                
                # add solutions to answer key 
                answer_key[(prob.latex_u, prob.w0, prob.ubar, prob.latex_cost, 
                     prob.error.print_error)] = (effort, opt_profit, con, opt_u) 
            
            #print(con) 
            prob.a0 = effort # use optimal effort
            
        else: 
            
            con, optimal_cost, opt_u = prob.maximize_dual([], False)
        
        # check whether the foa is valid 
        # We say the foa is valid if there is no global deviation with approximately the same utility 
        dev_vec = prob.exp_U_vectorized(con.value_function, np.linspace(.005,.1,100)) 
        max_util_dev = max(dev_vec) 
        util_dif = prob.exp_U(con.value_function, prob.a0) - max_util_dev
        if opt_a: 
            foa_valid = util_dif > 1e-2 and con.eta < 1e-5
        else: 
            foa_valid = util_dif > 1e-2 
        print(f"ubar is {ubar} and foa is {foa_valid}")

        if plot_type == "util":
            if opt_a: 
                efforts.append(effort) # add efforts to list 
            opt_us.append(opt_u)
            yy_values = prob.exp_U_vectorized(con.value_function, xx_values)
        elif plot_type == "contract":
            yy_values = con.wage_function(xx_values) 
        elif plot_type == "exp_wage": 
            yy_values = [prob.exp_wage(con.value_function, xx_value) for xx_value in xx_values]
        elif plot_type == "prob": 
            y_lower = prob.a0+((prob.w0-con.lmda)/con.mu) # only use for gaussian example! 
            yy_values = [] 
            for xx in xx_values: 
                outcomes = prob.error.y_grid_func(xx, 10000).flatten() 
                outcomes = outcomes[outcomes >= y_lower]  # Discard outcomes < y_lower
                
                if len(outcomes) > 1:
                    probabilities = prob.error.phi(outcomes, xx)
                    yy_value = simpson(probabilities, outcomes)  # Integrate using Simpson's rule
                else:
                    yy_value = 0  # If there are no valid outcomes, the integral is 0
                yy_values.append(yy_value)
                
        # Format the legend label dynamically using CE formatter
        ce_value = prob.u_inverse(ubar) - prob.w0  # Calculate certainty equivalent
        ce_label = f"${round(ce_value * 20, 0)}K"  # Format as dollars 

        # Map each special ubar to a softer color
        special_ubar_list = [first_ubar, middle_ubar, last_ubar]
        color_map = dict(zip(special_ubar_list, ghibli_line_colors[:len(special_ubar_list)]))
        
        # Determine line color and width
        if ubar in special_ubars:
            color = color_map[ubar]
            line_width = 2.0
        else:
            color = "#AAAAAA"  # Subtle gray for background lines
            line_width = 0.8
        
        # Dashed style if FOA invalid
        line_style = "-" if foa_valid else "--"
        
        # Draw the line
        ax.plot(
            xx_values,
            yy_values,
            label=fr"CE = {ce_label}" if ubar in special_ubars else None,
            linewidth=line_width,
            linestyle=line_style,
            color=color
        )


    # Apply custom tick formatting dynamically
    if tick_formatter_x:
        ax.xaxis.set_major_formatter(FuncFormatter(tick_formatter_x))
    if tick_formatter_y:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter_y))
        
    if plot_type == "util":
        if opt_a: 
            ax.plot(efforts, opt_us, color="#E75B64FF", linestyle='-', label=r'$a^{*}$ vs $E[u(a^*)]$')
        ax.legend(fontsize=legend_size, loc="upper left")

    # Finalize the subplot
    ax.set_title(title, fontsize=16)  
    ax.set_xlabel(xlab, fontsize=16)  
    ax.set_ylabel(ylab, fontsize=16)
    if plot_type == "util":
        ax.legend(fontsize=legend_size, loc="lower left")
    elif plot_type == "contract":
        ax.legend(fontsize=legend_size, loc="upper left")
    elif plot_type == "exp_wage": 
        ax.legend(fontsize=legend_size, loc="upper left")
    elif plot_type == "prob": 
        ax.legend(fontsize=legend_size, loc="lower right")
    ax.set_xlim(xx_range[0], xx_range[1])
    ax.set_ylim(yy_range[0], yy_range[1]) 
    
    if dist:
        ax2 = ax.twinx()  # Create a second y-axis sharing the same x-axis
        color = 'tab:red'
        phi_values = [prob.error.phi(xx, prob.a0) for xx in xx_values]
        ax2.set_ylabel('Density', color=color, fontsize=16)  # Set label for the second y-axis
        ax2.plot(xx_values, phi_values, label="Density", color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)  # Start the y-axis at 0

    # Save only if not part of a grid
    if output_filename and grid is None:
        plt.tight_layout()
        plt.savefig(output_filename)
        ()
        print(f"Combined plot saved as {output_filename}") 
    
    # the answer key is only for problems where we optimize a 
    if opt_a: 
        return answer_key

def create_combined_plots_grid(inputs, grid_shape, answer_key={}, 
                               row_titles=None, col_titles=None, 
                               output_filename="combined_plots_grid.png",plot_type="util",title=""):
    """
    Creates a grid of combined plots with row titles as y-axis labels and column titles as top graph titles.

    Parameters:
        inputs (list): List of dictionaries with keys 'prob', 'aa_range', 'ubars', and 'title' for each plot.
        answer_key: dictionary of solutions to already solved problems, only cooperates if opt_a=True
        grid_shape (tuple): The grid shape as (rows, cols).
        row_titles (list, optional): List of titles for each row (set as y-axis labels).
        col_titles (list, optional): List of titles for each column (set as graph titles).
        output_filename (str): Name of the output PNG file.
    """
    rows, cols = grid_shape

    # Create a figure with a tight layout
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(title, fontsize=18, weight='bold')  # Overall title for the figure
    axes = np.array(axes)  # Ensure axes is a NumPy array for easy indexing

    # Create the grid of plots
    input_idx = 0
    for row in range(rows):
        for col in range(cols):
            ax = axes[row, col]  # Select the current axis 
            if input_idx < len(inputs):
                input_data = inputs[input_idx]
                temp_answer_key = create_combined_plot(
                    input_data['prob'],
                    input_data['aa_range'],
                    input_data['yy_range'],
                    input_data['ubars'], 
                    input_data['grid_a0'], 
                    answer_key = answer_key,
                    tick_formatter_x=input_data['tick_x'],
                    tick_formatter_y=input_data['tick_y'],
                    grid=ax, 
                    plot_type=plot_type
                ) 
                answer_key.update(
                    {k: v for k, v in temp_answer_key.items() if k not in answer_key})
                input_idx += 1 
            else:
                ax.set_axis_off()  # Turn off unused axes

            # Add column titles to the top row
            if col_titles and row == 0:
                ax.set_title(col_titles[col], fontsize=16, weight='bold')

            # Add row titles to the left column
            if row_titles and col == 0:
                ax.set_ylabel(row_titles[row], fontsize=16, weight='bold', labelpad=10)
                ax.yaxis.set_label_coords(-0.2, 0.5)  # Ensure consistent label position

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0.02)
    ()
    print(f"Combined grid plot saved as {output_filename}") 
    
    return answer_key 

# define the gaussian-log utility problem 
prob = cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,n=1)  
aa_range = (0, 15) 
ubars = [prob.u(i + prob.w0) for i in np.linspace(0, 2, 11)] 

# Try to load answer_key, otherwise initialize an empty dictionary 
answer_key = {} 

if os.path.exists("answer_key.pkl"):
    try:
        with open("answer_key.pkl", "rb") as file:
            answer_key = dill.load(file)
        print("Loaded answer_key successfully!")
    except Exception as e:
        print(f"Error loading answer_key: {e}")
        answer_key = {}  # Reset to empty dictionary if loading fails
else:
    print("No existing answer_key file found. Initializing a new dictionary.")
    answer_key = {} 

a0_grid = np.linspace(1e-3,10,20) 

# solve gaussian log utility problem 
create_combined_plot(prob, aa_range, (0,15), ubars, a0_grid, answer_key=answer_key,
                    xlab="Outcome", ylab="Wage", plot_type = "contract", 
                    output_filename="output/norm_log_con_opt_a.pdf",
                    title="Wage Function for Different Reservation Utilities",
                    tick_formatter_x=dol_20k_formatter,
                    tick_formatter_y=dol_20k_formatter,
                    opt_a=True) 

create_combined_plot(prob, aa_range, (.35,1.6), ubars, a0_grid, answer_key=answer_key, 
                     xlab="Action", 
                     ylab= "Expected Utility", 
                     output_filename="output/norm_log_util_opt_a.pdf",
                     title="Expected Utility vs Action for Different Reservation Utilities",
                     tick_formatter_x=dol_20k_formatter,
                     tick_formatter_y=None,
                     opt_a=True)    

# look for lowest ubar such that foa is valid 
ubars_small = [prob.u(i + prob.w0) for i in np.linspace(0, .1, 21)] 
answer_key = create_combined_plot(prob, aa_range, (0,14), ubars_small, a0_grid, answer_key=answer_key,
                    xlab="Outcome", ylab="Wage", plot_type = "contract", 
                    output_filename="output/norm_log_con_checkfoa.pdf",
                    title="Wage Function for Different Reservation Utilities",
                    tick_formatter_x=dol_20k_formatter,
                    tick_formatter_y=dol_20k_formatter) 
answer_key = create_combined_plot(prob, aa_range, (.35,1.6), ubars_small, a0_grid, answer_key=answer_key, 
                     xlab="Action", 
                     ylab= "Expected Utility", 
                     output_filename="output/norm_log_util_checkfoa.pdf",
                     title="Expected Utility vs Action for Different Reservation Utilities",
                     tick_formatter_x=dol_20k_formatter,
                     tick_formatter_y=None) 

# create inputs for big grid of problems and solve 
prob_crra = cost_min_prob(a0=5,ubar=1,theta=1e-3,w0=2.5,u='crra',gamma=2)
prob_cara = cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,u='cara',gamma=.4)
inputs_util = [{'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,n=1),  
          'aa_range': (1e-2,15),
          'yy_range': (0,1.6),
          'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
          'grid_a0' : a0_grid,
          'tick_x': dol_20k_formatter,
          'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='crra',gamma=2),  
         'aa_range': (1e-2,10),
         'yy_range': (-.6,-.2),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
          'grid_a0' : a0_grid,
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=8e-3,w0=2.5,n=1,u='cara',gamma=.4),  
                  'aa_range': (1e-2,10),
                  'yy_range': (-1.8,-.3),
                  'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='exponential'),  
                  'aa_range': (1e-2,15),
                  'yy_range': (0,1.6),
                  'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': None},
       {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='crra',gamma=2,error='exponential'),  
                 'aa_range': (1e-2,15),
                 'yy_range': (-.8,-.1),
                 'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                 'grid_a0' : a0_grid,
                 'tick_x': dol_20k_formatter,
                 'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=8e-3,w0=2.5,u='cara',gamma=.4,error='exponential'),  
                  'aa_range': (1e-2,15),
                  'yy_range': (-1.8,-.3),
                  'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='gamma'),  
                  'aa_range': (1e-2,15),
                  'yy_range': (.2,1.6),
                  'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='crra',gamma=2,error='gamma'),  
                  'aa_range': (1e-2,15),
                  'yy_range': (-.8,-.1),
                  'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': None}, 
        {'prob': cost_min_prob(a0=5,ubar=1,theta=8e-3,w0=2.5,u='cara',gamma=.4,error='gamma'),  
                  'aa_range': (1e-2,15),
                  'yy_range': (-1.4,-.2),
                  'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': None}] 


answer_key = create_combined_plots_grid(inputs_util, (3,3),answer_key=answer_key,
                           row_titles=["Normal Distribution",
                                       "Exponential Distribution",
                                       "Gamma Distribution", 
                                       "T Distribution"],
                           col_titles = [r"Log: CRRA ($\gamma = 1$)",r"CRRA ($\gamma = 2$)",r"CARA ($\alpha = 0.4$)"],                           
                           output_filename="output/grid_util_opt_a.pdf",
                           title="Utilities")  

inputs_con = [{'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,n=1),  
          'aa_range': (1e-2,10),
          'yy_range': (0,10),
          'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
          'grid_a0' : a0_grid,
          'tick_x': dol_20k_formatter,
          'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='crra',gamma=2),  
         'aa_range': (1e-2,10),
         'yy_range': (0,10),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
          'grid_a0' : a0_grid,
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=8e-3,w0=2.5,n=1,u='cara',gamma=.4),  
                  'aa_range': (1e-2,10),
                  'yy_range': (0,10),
                  'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='exponential'),  
                  'aa_range': (1e-2,15),
                  'yy_range': (0,15),
                  'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': dol_20k_formatter},
       {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='crra',gamma=2,error='exponential'),  
                 'aa_range': (1e-2,15),
                 'yy_range': (0,10),
                 'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                 'grid_a0' : a0_grid,
                 'tick_x': dol_20k_formatter,
                 'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=8e-3,w0=2.5,u='cara',gamma=.4,error='exponential'),  
                  'aa_range': (1e-2,15),
                  'yy_range': (0,10),
                  'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='gamma'),  
                  'aa_range': (1e-2,20),
                  'yy_range': (0,10),
                  'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='crra',gamma=2,error='gamma'),  
                  'aa_range': (1e-2,15),
                  'yy_range': (0,10),
                  'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': dol_20k_formatter}, 
        {'prob': cost_min_prob(a0=5,ubar=1,theta=8e-3,w0=2.5,u='cara',gamma=.4,error='gamma'),  
                  'aa_range': (1e-2,15),
                  'yy_range': (0,10),
                  'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 3)],
                  'grid_a0' : a0_grid,
                  'tick_x': dol_20k_formatter,
                  'tick_y': dol_20k_formatter}] 

answer_key = create_combined_plots_grid(inputs_con,(3,3),answer_key=answer_key,
                           row_titles=["Normal Distribution",
                                       "Exponential Distribution",
                                       "Gamma Distribution", 
                                       "T Distribution"],
                           col_titles = [r"Log: CRRA ($\gamma = 1$)",r"CRRA ($\gamma = 2$)",r"CARA ($\alpha = 0.4$)"],                           
                           output_filename="output/grid_con_opt_a.pdf",
                           plot_type="contract",
                           title="Contracts") 

with open("answer_key.pkl", "wb") as file:
    dill.dump(answer_key, file) 