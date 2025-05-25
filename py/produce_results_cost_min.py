# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:50:29 2024

@author: Ilan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:21:56 2024
This is an old produce results file but we use it for the figures in section 5
@author: Ilan
"""

from .cost_min import cost_min_prob
from scipy.integrate import simpson  
import matplotlib.pyplot as plt
import imageio
import numpy as np   
from io import BytesIO
from matplotlib.ticker import FuncFormatter

# nice color scheme 
ghibli_line_colors = [
    "#4C413F",  # Muted charcoal
    "#278B9A",  # Teal
    "#D8AF39"   # Golden mustard
]

def create_pareto_frontier(prob, ubar_grid,output_filename=None, 
                           tick_formatter_x=None, tick_formatter_y=None):
    """
    Generates a Pareto frontier plot of optimal costs vs. optimal utilities
    for different reservation utilities (ubar values), and shades the area 
    beneath the curve.

    Parameters:
        prob (object): An instance of the cost_min_prob class.
        ubar_grid (list): A list of reservation utilities (ubar values).
    """
    # Initialize storage
    optimal_costs = []  # To store the optimal costs
    Us = []     # To store the corresponding optimal utilities

    # Loop over the grid of reservation utilities
    for ubar in ubar_grid:
        # Set reservation utility
        prob.ubar = ubar

        # Solve the cost minimization problem
        con, optimal_cost, optimal_U = prob.maximize_dual([], False)
        
        # Append results
        optimal_costs.append(optimal_cost)
        Us.append(ubar)

    # Plot optimal costs against optimal utilities
    plt.figure(figsize=(8, 6))
    plt.plot(Us, optimal_costs, linestyle='-', linewidth=4, color='k', label="Pareto Frontier")
    plt.fill_between(Us, optimal_costs, y2=max(optimal_costs), color='gray', alpha=0.5)

    # Remove blank space at the ends of the curve
    plt.xlim(min(Us), max(Us))
    plt.ylim(0, max(optimal_costs))  
    
    # Retrieve axis 
    ax = plt.gca()
    
    # Apply custom tick formatting
    if tick_formatter_x:
        ax.xaxis.set_major_formatter(FuncFormatter(tick_formatter_x))
    if tick_formatter_y:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter_y))
    
    plt.xlabel(r"Reservation Utility: $\bar{U}$", fontsize=16)
    plt.ylabel(r"Expected Wage: $\omega(\bar U)$", fontsize=16)
    plt.title("Pareto Frontier of the Relaxed Problem", fontsize=16)
    plt.tight_layout()
    
    # Save
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Pareto Frontier saved as {output_filename}")

    ()
   
def create_gif(prob, aa_range, ubars, gif_filename="output.gif", duration=500):
    """
    Creates a GIF of plots for varying values of ubar.

    Parameters:
        prob (object): An instance of the cost_min_prob class.
        aa_range (tuple): The range for aa values (min, max).
        ubars (list): List of ubar values to iterate over.
        gif_filename (str): Name of the output GIF file.
        duration (int): Duration of each frame in milliseconds.
    """
    images = []
    aa_values = np.linspace(aa_range[0], aa_range[1], 100)

    for ubar in ubars:
        prob.ubar = ubar
        con, opt_cost, opt_u = prob.maximize_dual()
        vv_values = prob.exp_U_vectorized(con.value_function, aa_values)
        optimal_aa = aa_values[np.argmax(vv_values)] 
        optimal_vv = np.max(vv_values) 

        # Create individual plot for GIF
        plt.figure(figsize=(6, 6))
        plt.plot(aa_values, vv_values, color="blue")
        plt.axhline(y=ubar, color='green', label=fr"$\bar{{u}} = {ubar}$")
        plt.plot(optimal_aa, optimal_vv, 'ro', label="Optimal Action")
        plt.title("Expected Utility vs Chosen Action at Different Reservation Utilities")
        plt.xlabel("Action")
        plt.ylabel("Expected Utility")
        plt.legend()
        plt.grid(True)
        plt.xlim(aa_range[0], aa_range[1])  # Keep axis limits consistent
        plt.ylim(-2, 2)  # Scale y-axis based on max u

        # Save the plot to a BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))  # Append the image from the buffer
        buf.close()
        plt.close()

    # Save the GIF
    imageio.mimsave(gif_filename, images, duration=duration, loop=0)
    print(f"GIF saved as {gif_filename}") 
    
# Default tick formatting function
def dol_10k_formatter(value, _):
    """Default: Format axis ticks as dollar amounts in units of $10,000."""
    if value > 0:
        return f"${int(value * 10)}K"
    else:
        return f"${value}"
    
# Default tick formatting function
def dol_20k_formatter(value, _):
    """Default: Format axis ticks as dollar amounts in units of $10,000."""
    if value > 0:
        return f"${int(value * 20)}K"
    else:
        return f"${value}"

# Updated function
def create_combined_plot(prob, xx_range, yy_range, ubars, 
                         plot_type="util", xlab="", ylab="", title="", legend_size=10, 
                         grid=None, output_filename=None, 
                         tick_formatter_x=None, tick_formatter_y=None, dist=False):
    """
    Creates a single plot with all lines, emphasizing the first, middle, and last values of ubar.
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

    for ubar in ubars:
        prob.ubar = ubar
        con, opt_cost, opt_u = prob.maximize_dual()
        if plot_type == "util":
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

        # Draw the line
        ax.plot(
            xx_values,
            yy_values,
            label=fr"CE = {ce_label}" if ubar in special_ubars else None,
            linewidth=line_width,
            color=color
        )

    # Apply custom tick formatting dynamically
    if tick_formatter_x:
        ax.xaxis.set_major_formatter(FuncFormatter(tick_formatter_x))
    if tick_formatter_y:
        ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter_y))

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
        
# Basic plots 
prob = cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,n=1)  
aa_range = (0, 10) 
ubars = [prob.u(i + prob.w0) for i in np.linspace(0, 2, 21)] 
ubars_pf = np.linspace(0,2,21)
# Pass a lambda function to compute CE dynamically
u_ce_formatter = FuncFormatter(lambda value, pos: 
    f"{value:.2f}\n(CE = ${20 * (prob.u_inverse(value) - prob.w0):.0f}K)" 
    if pos % 4 == 0 else f"{value:.2f}"
)



create_pareto_frontier(prob,ubars_pf,output_filename="output/norm_log_pf.pdf",
                       tick_formatter_x = u_ce_formatter, 
                       tick_formatter_y=dol_20k_formatter)  
create_combined_plot(prob, aa_range, (.5,1.8), ubars, xlab="Action", 
                     ylab= "Expected Utility", 
                     output_filename="output/norm_log_util.pdf",
                     title="Expected Utility vs Action for Different Reservation Utilities",
                     tick_formatter_x=dol_20k_formatter,
                     tick_formatter_y=None)
create_combined_plot(prob, aa_range, (0,7), ubars, xlab="Outcome", 
                     ylab="Wage", plot_type = "contract", 
                     output_filename="output/norm_log_con.pdf",
                     title="Wage Function for Different Reservation Utilities",
                     tick_formatter_x=dol_20k_formatter,
                     tick_formatter_y=dol_20k_formatter)

create_combined_plot(prob, (0,5), (0,1), ubars, xlab="Action", 
                     ylab="Probability $w(y) > 0$", plot_type = "prob", 
                     output_filename="output/norm_log_probs.pdf",
                     title="",
                     tick_formatter_x=dol_20k_formatter,
                     tick_formatter_y=None) 

create_combined_plot(prob, (0,5), (0,4), ubars, xlab="Action", 
                     ylab="Expected Wage", plot_type = "exp_wage", 
                     output_filename="output/norm_log_exp_wage.pdf",
                     title="",
                     tick_formatter_x=dol_20k_formatter,
                     tick_formatter_y=dol_20k_formatter) 

def create_combined_plots_grid(inputs, grid_shape, row_titles=None, col_titles=None, 
                               output_filename="combined_plots_grid.png",plot_type="util",title=""):
    """
    Creates a grid of combined plots with row titles as y-axis labels and column titles as top graph titles.

    Parameters:
        inputs (list): List of dictionaries with keys 'prob', 'aa_range', 'ubars', and 'title' for each plot.
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
                create_combined_plot(
                    input_data['prob'],
                    input_data['aa_range'],
                    input_data['yy_range'],
                    input_data['ubars'], 
                    tick_formatter_x=input_data['tick_x'],
                    tick_formatter_y=input_data['tick_y'],
                    grid=ax, 
                    plot_type=plot_type
                )
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

prob = cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5)
prob_crra = cost_min_prob(a0=5,ubar=1,theta=1e-3,w0=2.5,u='crra',gamma=2)
prob_cara = cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,u='cara',gamma=.4)

inputs_util = [{'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5),  
          'aa_range': (0,10),
          'yy_range': (.5,1.8),
          'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
          'tick_x': dol_20k_formatter,
          'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=2e-3,w0=2.5,u='crra',gamma=2),  
         'aa_range': (0,10),
         'yy_range': (-.6,-.1),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='cara',gamma=.4),  
         'aa_range': (0,10),
         'yy_range': (-1.2,-.2),
         'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='exponential'),  
         'aa_range': (0,10),
         'yy_range': (.5,1.8),
         'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=2e-3,w0=2.5,u='crra',gamma=2,error='exponential'),  
         'aa_range': (0,10),
         'yy_range': (-.6,-.1),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='cara',gamma=.4,error='exponential'),  
         'aa_range': (0,10),
         'yy_range': (-1.2,-.2),
         'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='gamma'),  
         'aa_range': (0,10),
         'yy_range': (.5,1.8),
         'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=2e-3,w0=2.5,u='crra',gamma=2,error='gamma'),  
         'aa_range': (0,10),
         'yy_range': (-.6,-.1),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None}, 
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='cara',gamma=.4,error='gamma'),  
         'aa_range': (0,10),
         'yy_range': (-1.2,-.2),
         'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='t',n={'df': 1.5, 'sigma': 1}),  
         'aa_range': (0,10),
         'yy_range': (.5,1.8),
         'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=2e-3,w0=2.5,u='crra',gamma=2,error='t',n={'df': 1.5, 'sigma': 1}),  
         'aa_range': (0,10),
         'yy_range': (-.6,-.1),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='cara',gamma=.4,error='t',n={'df': 1.5, 'sigma': 1}),  
         'aa_range': (0,10),
         'yy_range': (-1.2,-.2),
         'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': None}]

inputs_con = [{'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5),  
          'aa_range': (0,10),
          'yy_range': (0,8),
          'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
          'tick_x': dol_20k_formatter,
          'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=2e-3,w0=2.5,u='crra',gamma=2),  
         'aa_range': (0,10),
         'yy_range': (0,8),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='cara',gamma=.4),  
         'aa_range': (0,10),
         'yy_range': (0,8),
         'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter}, 
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='exponential'),  
         'aa_range': (0,10),
         'yy_range': (0,10),
         'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=2e-3,w0=2.5,u='crra',gamma=2,error='exponential'),  
         'aa_range': (0,10),
         'yy_range': (0,20),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='cara',gamma=.4,error='exponential'),  
        'aa_range': (0,10),
        'yy_range': (0,8),
        'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
        'tick_x': dol_20k_formatter,
        'tick_y': dol_20k_formatter}, 
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='gamma'),  
         'aa_range': (0,10),
         'yy_range': (0,8),
         'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=2e-3,w0=2.5,u='crra',gamma=2,error='gamma'),  
         'aa_range': (0,10),
         'yy_range': (0,8),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter}, 
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='cara',gamma=.4,error='gamma'),  
        'aa_range': (0,10),
        'yy_range': (0,8),
        'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
        'tick_x': dol_20k_formatter,
        'tick_y': dol_20k_formatter}, 
        {'prob': cost_min_prob(a0=5,ubar=1,theta=1e-2,w0=2.5,error='t',n={'df': 1.5, 'sigma': 1}),  
         'aa_range': (0,10),
         'yy_range': (0,8),
         'ubars': [prob.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=2e-3,w0=2.5,u='crra',gamma=2,error='t',n={'df': 1.5, 'sigma': 1}),  
         'aa_range': (0,10),
         'yy_range': (0,8),
         'ubars': [prob_crra.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter},
        {'prob': cost_min_prob(a0=5,ubar=1,theta=4e-3,w0=2.5,u='cara',gamma=.4,error='t',n={'df': 1.5, 'sigma': 1}),  
         'aa_range': (0,10),
         'yy_range': (0,8),
         'ubars': [prob_cara.u(i + prob.w0) for i in np.linspace(0, 2, 21)],
         'tick_x': dol_20k_formatter,
         'tick_y': dol_20k_formatter}] 


create_combined_plots_grid(inputs_util, (4,3), 
                           row_titles=["Normal Distribution",
                                       "Exponential Distribution",
                                       "Gamma Distribution", 
                                       "T Distribution"],
                           col_titles = [r"Log: CRRA ($\gamma = 1$)",r"CRRA ($\gamma = 2$)",r"CARA ($\alpha = 0.4$)"],                           
                           output_filename="output/grid_util_temp.pdf",
                           title="Utilities")  

create_combined_plots_grid(inputs_con, (4,3), 
                           row_titles=["Normal Distribution",
                                       "Exponential Distribution",
                                       "Gamma Distribution", 
                                       "T Distribution"],
                           col_titles = [r"Log: CRRA ($\gamma = 1$)",r"CRRA ($\gamma = 2$)",r"CARA ($\alpha = 0.4$)"],                           
                           output_filename="output/grid_contract_temp.pdf",
                           plot_type="contract",
                           title="Contracts")  


"""
prob =  
aa_range = (0, 20) 
ubars = [prob.u(i + prob.w0) for i in np.linspace(0, 3, 15)] 

create_combined_plot(prob, aa_range, (-2,2), ubars, xlab="Action", 
                     ylab= "Expected Utility", 
                     output_filename="output/norm_log_util.pdf",
                     title="Expected Utility vs Action for Different Reservation Utilities",
                     tick_formatter_x=dol_10k_formatter,
                     tick_formatter_y=None) 
"""             

"""
def ce_formatter(value, _, prob):
    Format ticks based on the prob object.
    Uses the u_inverse method to adjust the values dynamically and ensures clean rounding.
    try:
        ce_value = prob.u_inverse(value) - prob.w0  # Calculate the certainty equivalent
        ce_value_scaled = round(ce_value * 10, 1)  # Scale and round to 1 decimal place
        if ce_value_scaled > 0:
            return f"${ce_value_scaled}K"  # Positive values
        elif ce_value_scaled < 0:
            return f"$-{abs(ce_value_scaled)}K"  # Negative values, formatted cleanly
        else:
            return "$0"  # Zero values explicitly
    except:
        return ""  # Return blank for invalid values during formatting 
""" 
