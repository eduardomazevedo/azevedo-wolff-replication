"""
Gaussian log utility example — refactored and cleaned version.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# -------------------------------------------------------------
# Color options for graphs
# -------------------------------------------------------------
# COLORMAP = sns.dark_palette("#8FB1E9", reverse=False, as_cmap=True)  # Ghibli blue: RGB(143, 177, 233)
# Sequential: Deep Teal to Red
teal_norm = (59/255.0, 105/255.0, 120/255.0)  # Deep teal: RGB(59, 105, 120)
red_norm = (255/255.0, 92/255.0, 92/255.0)    # Red: RGB(255, 92, 92)
COLORMAP = LinearSegmentedColormap.from_list('teal_red', [teal_norm, red_norm], N=256)
SCATTER_COLOR = "red"
HORIZONTAL_LINE_COLOR = "gray"
ARROW_COLOR = "black"

# -------------------------------------------------------------
# Line width options for graphs (scaled for 6" width, 16:9 aspect ratio)
# -------------------------------------------------------------
PLOT_LINEWIDTH = 1.2  # scaled from 2 (60% of original for 6" vs 10" width)
HORIZONTAL_LINE_WIDTH = 0.3  # scaled from 0.5
ARROW_LINEWIDTH = 0.48  # scaled from 0.8
FONT_SIZE = 6  # scaled from 10 for annotations
LABEL_FONT_SIZE = 7.2  # scaled from 12 (default) for axis labels
TITLE_FONT_SIZE = 8.4  # scaled from 14 (default) for titles
MARKER_SIZE = 3  # scaled from 5 for scatter points
DISCRETE_WAGE_MARKER_SIZE = 4  # marker size for discrete wage function plots

# -------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------
def arrow_positions_for_labels(ax, text1, text2):
    """Compute anchor positions for text arrows in data coordinates."""
    fig = ax.figure
    fig.canvas.draw()

    bbox1 = text1.get_window_extent(fig.canvas.get_renderer())
    bbox2 = text2.get_window_extent(fig.canvas.get_renderer())

    trans = ax.transData.inverted()
    x1, y1 = trans.transform((bbox1.x1, bbox1.y0 + bbox1.height/2))
    x2, y2 = trans.transform((bbox2.x1, bbox2.y0 + bbox2.height/2))

    xlim = ax.get_xlim()
    # Reduced offset to move arrow origins about 2 characters to the left
    offset = (xlim[1] - xlim[0]) * 0.005  # reduced from 0.015

    return (x1 + offset, y1), (x2 + offset, y2)


def _plot_wage_basic(ax, y_grid, wage_functions, reservation_wage_grid, foa_flags, title, 
                     xlim=None, distribution_type="continuous"):
    """
    Basic function to plot wage functions on an axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    y_grid : array
        Output values for x-axis
    wage_functions : list of arrays
        Wage functions to plot
    reservation_wage_grid : array
        Reservation wages for coloring
    foa_flags : list of bool
        First order approach flags for line styles
    title : str
        Plot title
    xlim : tuple or None
        If provided, restricts x-axis to (xmin, xmax) and masks data accordingly
    distribution_type : str
        "discrete" or "continuous" - determines whether to use scatter or line plots
    """
    cmap = COLORMAP
    norm = plt.Normalize(reservation_wage_grid.min(), reservation_wage_grid.max())

    for i, wf in enumerate(wage_functions):
        if xlim is not None:
            # Only plot the portion of wage functions within the xlim range
            mask = (y_grid >= xlim[0]) & (y_grid <= xlim[1])
            y_vals = y_grid[mask]
            w_vals = wf[mask]
        else:
            y_vals = y_grid
            w_vals = wf
        
        color = cmap(norm(reservation_wage_grid[i]))
        
        if distribution_type == "discrete":
            # Use scatter plot with lines connecting points for discrete distributions
            ax.plot(
                y_vals, w_vals,
                color=color,
                linestyle="-" if foa_flags[i] else "--",
                marker='o',  # same marker for all
                markersize=DISCRETE_WAGE_MARKER_SIZE,
                alpha=0.6,
                linewidth=PLOT_LINEWIDTH
            )
        else:
            # Use line plot for continuous distributions
            ax.plot(
                y_vals, w_vals,
                color=color,
                linestyle="-" if foa_flags[i] else "--",
                alpha=0.6, linewidth=PLOT_LINEWIDTH
            )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    
    ax.set_xlabel("Output (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Wage (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)


def plot_wage_functions(
    filename, y_grid, wage_functions, reservation_wage_grid, foa_flags, title,
    distribution_type="continuous"
):
    """Plot wage functions and save to file."""
    cmap = COLORMAP
    norm = plt.Normalize(reservation_wage_grid.min(), reservation_wage_grid.max())

    fig, ax = plt.subplots(figsize=(6, 3.375))  # 6 inches wide, 16:9 aspect ratio
    
    _plot_wage_basic(ax, y_grid, wage_functions, reservation_wage_grid, foa_flags, title,
                     distribution_type=distribution_type)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Reservation Wage (USD 1,000s)")

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def _plot_utility_basic(ax, action_grid_plot, agent_utilities, targets,
                        reservation_wage_grid, foa_flags, title, mhp, utility_cfg):
    """
    Basic function to plot agent utility curves on an axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    action_grid_plot : array
        Action values for x-axis
    agent_utilities : list of arrays
        Utility curves to plot
    targets : list of tuples
        (action, utility) pairs for scatter points
    reservation_wage_grid : array
        Reservation wages for coloring
    foa_flags : list of bool
        First order approach flags for line styles
    title : str
        Plot title
    mhp : MoralHazardProblem
        The moral hazard problem instance
    utility_cfg : dict
        Utility configuration dictionary
    """
    cmap = COLORMAP
    norm = plt.Normalize(reservation_wage_grid.min(), reservation_wage_grid.max())

    # curves and red dots
    for i, U in enumerate(agent_utilities):
        ax.plot(
            action_grid_plot, U,
            color=cmap(norm(reservation_wage_grid[i])),
            linestyle="-" if foa_flags[i] else "--",
            alpha=0.6, linewidth=PLOT_LINEWIDTH
        )
        a_star, u_star = targets[i]
        ax.scatter(a_star, u_star, color=SCATTER_COLOR, s=MARKER_SIZE, zorder=5)

    # label positions
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    # Add horizontal lines through each red dot (only when first order approach fails)
    for i, (a_star, u_star) in enumerate(targets):
        if not foa_flags[i]:
            ax.axhline(y=u_star, color=HORIZONTAL_LINE_COLOR, linestyle='-', linewidth=HORIZONTAL_LINE_WIDTH, alpha=0.7, zorder=1)
    
    # Check which conditions exist
    has_holds = any(foa_flags)
    has_fails = any(not f for f in foa_flags)
    
    x_text = xlim[0] + 0.08*(xlim[1]-xlim[0])
    y_text1 = ylim[1] - 0.05*(ylim[1]-ylim[0])
    y_text2 = ylim[1] - 0.12*(ylim[1]-ylim[0])

    text_holds = None
    text_fails = None
    
    if has_holds:
        text_holds = ax.text(
            x_text, y_text1, "first order approach holds",
            fontsize=FONT_SIZE, va="top"
        )
    
    if has_fails:
        y_pos = y_text1 if not has_holds else y_text2
        text_fails = ax.text(
            x_text, y_pos, "first order approach fails",
            fontsize=FONT_SIZE, va="top"
        )

    # Compute arrow anchor positions
    if has_holds and has_fails:
        (x1, y1), (x2, y2) = arrow_positions_for_labels(ax, text_holds, text_fails)
    elif has_holds:
        fig = ax.figure
        fig.canvas.draw()
        bbox = text_holds.get_window_extent(fig.canvas.get_renderer())
        trans = ax.transData.inverted()
        x1, y1 = trans.transform((bbox.x1, bbox.y0 + bbox.height/2))
        xlim = ax.get_xlim()
        offset = (xlim[1] - xlim[0]) * 0.005  # reduced from 0.015 to move arrow origins left
        x1 = x1 + offset
        x2, y2 = None, None
    elif has_fails:
        fig = ax.figure
        fig.canvas.draw()
        bbox = text_fails.get_window_extent(fig.canvas.get_renderer())
        trans = ax.transData.inverted()
        x2, y2 = trans.transform((bbox.x1, bbox.y0 + bbox.height/2))
        xlim = ax.get_xlim()
        offset = (xlim[1] - xlim[0]) * 0.005  # reduced from 0.015 to move arrow origins left
        x2 = x2 + offset
        x1, y1 = None, None

    # Draw arrows to appropriate text boxes
    for i, (a_star, u_star) in enumerate(targets):
        if foa_flags[i] and has_holds:
            ax.annotate("", xy=(a_star, u_star), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=ARROW_LINEWIDTH, alpha=1.0))
        elif not foa_flags[i] and has_fails:
            ax.annotate("", xy=(a_star, u_star), xytext=(x2, y2),
                        arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=ARROW_LINEWIDTH, alpha=1.0))
    
    # Set x-axis limits to match action_grid_plot range
    ax.set_xlim(action_grid_plot.min(), action_grid_plot.max())
    ax.set_xlabel("Action (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Agent expected utility (certain equivalent, USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)

    # Set y-axis ticks at natural certain equivalent values
    # First, get the current y-axis range in utility values
    ylim = ax.get_ylim()
    u_min, u_max = ylim
    
    # Convert to certain equivalent range
    ce_min = mhp.k(u_min)
    ce_max = mhp.k(u_max)
    
    # Find all multiples of 10 in the certain equivalent range
    # Start from the smallest multiple of 10 >= ce_min
    start_tick = np.ceil(ce_min / 10) * 10
    # End at the largest multiple of 10 <= ce_max
    end_tick = np.floor(ce_max / 10) * 10
    
    # Generate all multiples of 10 in the range [start_tick, end_tick]
    # np.arange works correctly with negative numbers
    if start_tick <= end_tick:
        # Include end_tick by going one step beyond
        ce_ticks = np.arange(start_tick, end_tick + 10, 10)
        # start_tick and end_tick are already the correct bounds, so all values are valid
    else:
        # If no multiples of 10 in range, use empty array
        ce_ticks = np.array([])
    
    # Convert certain equivalent ticks back to utility values for positioning
    if len(ce_ticks) > 0:
        u_ticks = utility_cfg["u"](ce_ticks)
        # Set the ticks and format labels
        ax.set_yticks(u_ticks)
        ax.set_yticklabels([f'{ce:.0f}' for ce in ce_ticks], fontsize=FONT_SIZE)


def plot_agent_utilities(
    filename, action_grid_plot, agent_utilities, targets,
    reservation_wage_grid, foa_flags, title, mhp, utility_cfg
):
    """Plot agent utilities and save to file."""
    cmap = COLORMAP
    norm = plt.Normalize(reservation_wage_grid.min(), reservation_wage_grid.max())

    fig, ax = plt.subplots(figsize=(6, 3.375))  # 6 inches wide, 16:9 aspect ratio

    _plot_utility_basic(ax, action_grid_plot, agent_utilities, targets,
                       reservation_wage_grid, foa_flags, title, mhp, utility_cfg)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Reservation Wage (USD 1,000s)")

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_stacked_wage_and_utility(
    filename, y_grid, wage_functions, action_grid_plot, agent_utilities, targets,
    reservation_wage_grid, foa_flags, title_wage, title_utility, mhp, utility_cfg,
    distribution_type="continuous"
):
    """
    Create a stacked figure with optimal contract (top) and agent utility (bottom).
    The x-axis range for the wage plot is restricted to match the action_grid_plot range.
    Color scale is shared between both plots.
    """
    cmap = COLORMAP
    norm = plt.Normalize(reservation_wage_grid.min(), reservation_wage_grid.max())

    # Create figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6.75), sharex=False)  # 6" wide, double height for 2 plots
    
    # Get the x-axis range from action_grid_plot (for the utility plot)
    action_xmin = action_grid_plot.min()
    action_xmax = action_grid_plot.max()
    
    # TOP PLOT: Optimal Contract (wage functions)
    _plot_wage_basic(ax1, y_grid, wage_functions, reservation_wage_grid, foa_flags, 
                     title_wage, xlim=(action_xmin, action_xmax), 
                     distribution_type=distribution_type)
    
    # BOTTOM PLOT: Agent Utility
    _plot_utility_basic(ax2, action_grid_plot, agent_utilities, targets,
                       reservation_wage_grid, foa_flags, title_utility, mhp, utility_cfg)
    
    # Set x-axis limits to match the top plot exactly
    ax2.set_xlim(action_xmin, action_xmax)
    
    # Adjust layout first to make room for colorbar at the bottom (including label)
    fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.3)
    
    # Shared colorbar at the bottom of the figure
    # Create a dedicated axes for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Position colorbar with one more line of spacing from bottom graph
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.set_label("Reservation Wage (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------
# Numerical routines
# -------------------------------------------------------------
def compute_principal_results(mhp, utility_cfg, reservation_wage_grid, a_min, a_max, 
                              n_a_iterations, action_grid_plot):
    """Solve principal problem over reservation wage grid."""
    wage_functions = []
    utilities = []
    optimal_actions = []
    optimal_utils = []
    foa = []
    Ew = []
    Ew_relaxed = []

    ru_grid = [utility_cfg["u"](w) for w in reservation_wage_grid]

    for ru in ru_grid:
        sol = mhp.solve_principal_problem(
            revenue_function=lambda a: a,
            reservation_utility=ru,
            a_min=a_min, a_max=a_max,
            a_ic_lb=a_min, a_ic_ub=a_max,
            n_a_iterations=n_a_iterations
        )

        sol_rel = mhp.solve_principal_problem(
            revenue_function=lambda a: a,
            reservation_utility=ru,
            a_min=a_min, a_max=a_max,
            a_ic_lb=a_min, a_ic_ub=a_max,
            n_a_iterations=0
        )

        c = sol.cmp_result.optimal_contract
        wage_functions.append(mhp.k(c))
        U = mhp.U(c, action_grid_plot)
        utilities.append(U)

        a_star = sol.optimal_action
        optimal_actions.append(a_star)
        optimal_utils.append(mhp.U(c, a_star))

        foa.append(sol.cmp_result.first_order_approach_holds)
        Ew.append(float(sol.cmp_result.constraints["Ewage"]))
        Ew_relaxed.append(float(sol_rel.cmp_result.constraints["Ewage"]))

    return wage_functions, utilities, optimal_actions, optimal_utils, foa, Ew, Ew_relaxed


def compute_cost_minimization_results(mhp, utility_cfg, reservation_wage_grid, 
                                      intended_action, n_a_iterations, action_grid_plot):
    wage_functions = []
    utilities = []
    foa = []
    util_at_intended = []

    ru_grid = [utility_cfg["u"](w) for w in reservation_wage_grid]

    for ru in ru_grid:
        sol = mhp.solve_cost_minimization_problem(
            intended_action=intended_action,
            reservation_utility=ru,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=n_a_iterations
        )
        c = sol.optimal_contract

        wage_functions.append(mhp.k(c))
        utilities.append(mhp.U(c, action_grid_plot))
        util_at_intended.append(mhp.U(c, intended_action))
        foa.append(sol.first_order_approach_holds)

    return wage_functions, utilities, foa, util_at_intended


def plot_pareto_frontier(filename, mhp, utility_cfg, reservation_wage_grid_pareto, 
                         intended_action, n_a_iterations):
    """Plot the Pareto frontier."""
    pareto_ru = utility_cfg["u"](reservation_wage_grid_pareto)

    Ew_pf = []
    Ew_pf_rel = []

    for ru in pareto_ru:
        sol = mhp.solve_cost_minimization_problem(
            intended_action=intended_action,
            reservation_utility=ru,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=n_a_iterations
        )
        Ew_pf.append(float(sol.constraints["Ewage"]))

        sol_rel = mhp.solve_cost_minimization_problem(
            intended_action=intended_action,
            reservation_utility=ru,
            a_ic_lb=0.0,
            a_ic_ub=100.0,
            n_a_iterations=0
        )
        Ew_pf_rel.append(float(sol_rel.constraints["Ewage"]))

    fig, ax = plt.subplots(figsize=(6, 3.375))  # 6 inches wide, 16:9 aspect ratio

    # Convert to numpy arrays for easier manipulation
    pareto_ru = np.array(pareto_ru)
    Ew_pf = np.array(Ew_pf)
    Ew_pf_rel = np.array(Ew_pf_rel)

    # Define colors
    ghibli_palette_deep_teal = (59/255.0, 105/255.0, 120/255.0)  # RGB(59, 105, 120) normalized
    light_gray = (0.85, 0.85, 0.85)  # Light gray

    # Set x-axis limits to match CE range from -20 to 50 (converted to utility values)
    ce_min, ce_max = -20.0, 50.0
    u_min = utility_cfg["u"](ce_min)
    u_max = utility_cfg["u"](ce_max)
    ax.set_xlim(u_min, u_max)

    # Set y-axis limits with padding, then use top for fill_between
    y_min = 0.0  # Start from 0
    y_max = max(Ew_pf_rel.max(), Ew_pf.max()) * 1.1  # 10% padding above
    ax.set_ylim(y_min, y_max)
    # Get the actual top of the plot area after setting ylim
    y_top = ax.get_ylim()[1]

    # Plot relaxed problem (larger frontier) - black solid line
    # Fill area above the curve with light gray (all the way to top of plot)
    ax.fill_between(pareto_ru, Ew_pf_rel, y_top, color=light_gray, alpha=1.0, zorder=1)
    ax.plot(pareto_ru, Ew_pf_rel, color='black', linestyle='-', linewidth=PLOT_LINEWIDTH * 2, 
            label="Relaxed Problem", zorder=2)

    # Plot full problem (smaller frontier) - dashed teal line with alpha=0.5
    # Fill area above the curve with light teal (alpha=0.5, all the way to top of plot)
    ax.fill_between(pareto_ru, Ew_pf, y_top, color=ghibli_palette_deep_teal, alpha=0.5, zorder=3)
    ax.plot(pareto_ru, Ew_pf, color=ghibli_palette_deep_teal, linestyle='--', linewidth=PLOT_LINEWIDTH * 2, 
            label="Full Problem", alpha=0.5, zorder=4)

    ax.set_xlabel("Agent expected utility (certain equivalent, USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Expected Wage Cost (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax.set_title("Pareto Frontier: Expected Wage Cost vs Agent Expected Utility", fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, loc='lower right')

    # Set x-axis ticks at natural certain equivalent values
    # Use a subset of the CE range for nice tick locations
    # Select evenly spaced values that are nice round numbers
    # (ce_min and ce_max already defined above)
    n_ticks = 8
    ce_ticks = np.linspace(ce_min, ce_max, n_ticks)
    # Round to nearest nice values
    tick_spacing = (ce_max - ce_min) / (n_ticks - 1)
    if tick_spacing >= 10:
        round_to = 10
    elif tick_spacing >= 5:
        round_to = 5
    else:
        round_to = 1

    ce_ticks = np.round(ce_ticks / round_to) * round_to
    ce_ticks = np.unique(ce_ticks)  # Remove duplicates
    ce_ticks = ce_ticks[(ce_ticks >= ce_min) & (ce_ticks <= ce_max)]  # Keep within range

    # Convert certain equivalent ticks to utility values for positioning
    u_ticks = utility_cfg["u"](ce_ticks)

    # Set the ticks and format labels
    ax.set_xticks(u_ticks)
    ax.set_xticklabels([f'{ce:.0f}' for ce in ce_ticks], fontsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def make_example_figures(mhp, utility_cfg, n_a_iterations, reservation_wage_grid,
                         reservation_wage_grid_pareto, a_min, a_max, action_grid_plot,
                         intended_action, dirname="figures"):
    """
    Generate all example figures for a given specification.
    
    Parameters:
    -----------
    mhp : MoralHazardProblem
        The moral hazard problem instance
    utility_cfg : dict
        Utility configuration dictionary
    n_a_iterations : int
        Number of action iterations for solving
    reservation_wage_grid : array
        Grid of reservation wages for principal and cost minimization problems
    reservation_wage_grid_pareto : array
        Grid of reservation wages for Pareto frontier
    a_min : float
        Minimum action value
    a_max : float
        Maximum action value
    action_grid_plot : array
        Grid of actions for plotting utility curves
    intended_action : float
        Intended action for cost minimization problem
    dirname : str
        Directory name where figures will be saved (default: "figures")
    """
    # Create output directory
    os.makedirs(dirname, exist_ok=True)
    
    # Get y_grid from mhp
    y_grid = mhp._y_grid
    
    # PART 1 — Principal Problem
    (
        wage_pp,
        util_pp,
        a_star_pp,
        u_star_pp,
        foa_pp,
        Ew_pp,
        Ew_rel_pp,
    ) = compute_principal_results(mhp, utility_cfg, reservation_wage_grid, a_min, a_max,
                                   n_a_iterations, action_grid_plot)

    # Check whether distribution is discrete or continuous
    if np.allclose(mhp._w, 1.0):
        distribution_type = "discrete"
    else:
        distribution_type = "continuous"

    plot_wage_functions(
        f"{dirname}/pp_wage.png",
        y_grid,
        wage_pp,
        reservation_wage_grid,
        foa_pp,
        "Optimal Contract",
        distribution_type=distribution_type
    )

    plot_agent_utilities(
        f"{dirname}/pp_utility.png",
        action_grid_plot,
        util_pp,
        list(zip(a_star_pp, u_star_pp)),
        reservation_wage_grid,
        foa_pp,
        "Agent Utility vs Action given Optimal Contract",
        mhp,
        utility_cfg
    )

    plot_stacked_wage_and_utility(
        f"{dirname}/pp_stacked.png",
        y_grid,
        wage_pp,
        action_grid_plot,
        util_pp,
        list(zip(a_star_pp, u_star_pp)),
        reservation_wage_grid,
        foa_pp,
        "Optimal Contract",
        "Agent Utility vs Action given Optimal Contract",
        mhp,
        utility_cfg,
        distribution_type=distribution_type
    )

    # PART 2 — Cost Minimization
    (
        wage_cm,
        util_cm,
        foa_cm,
        util_intended_cm
    ) = compute_cost_minimization_results(mhp, utility_cfg, reservation_wage_grid,
                                          intended_action, n_a_iterations, action_grid_plot)

    plot_wage_functions(
        f"{dirname}/cm_wage.png",
        y_grid,
        wage_cm,
        reservation_wage_grid,
        foa_cm,
        "Optimal Contract",
        distribution_type=distribution_type
    )

    plot_agent_utilities(
        f"{dirname}/cm_utility.png",
        action_grid_plot,
        util_cm,
        [(intended_action, u) for u in util_intended_cm],
        reservation_wage_grid,
        foa_cm,
        "Agent Utility vs Action given Optimal Contract",
        mhp,
        utility_cfg
    )

    plot_stacked_wage_and_utility(
        f"{dirname}/cm_stacked.png",
        y_grid,
        wage_cm,
        action_grid_plot,
        util_cm,
        [(intended_action, u) for u in util_intended_cm],
        reservation_wage_grid,
        foa_cm,
        "Optimal Contract",
        "Agent Utility vs Action given Optimal Contract",
        mhp,
        utility_cfg,
        distribution_type=distribution_type
    )

    # PART 3 — Pareto Frontier
    plot_pareto_frontier(
        f"{dirname}/pareto_frontier.png",
        mhp,
        utility_cfg,
        reservation_wage_grid_pareto,
        intended_action,
        n_a_iterations
    )
