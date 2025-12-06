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

os.makedirs('figures', exist_ok=True)

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

# -------------------------------------------------------------
# Primitives and configuration
# -------------------------------------------------------------
initial_wealth = 50
first_best_effort = 100
sigma = 10.0
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)
def C(a): return theta * a ** 2 / 2

def Cprime(a): return theta * a

utility_cfg = make_utility_cfg("log", w0=initial_wealth)
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)

comp_cfg = {
    "distribution_type": "continuous",
    "y_min": 0.0   - 6 * sigma,
    "y_max": 180.0 + 6 * sigma,
    "n": 201,  # must be odd
}

cfg = {
    "problem_params": {**utility_cfg, **dist_cfg, "C": C, "Cprime": Cprime},
    "computational_params": comp_cfg
}

reservation_wage_grid = np.linspace(-1.0, 50.0, 6)
reservation_wage_grid_pareto = np.linspace(-20.0, 50.0, 100)
a_min, a_max = 0.0, 180.0
action_grid_plot = np.linspace(a_min, a_max, 100)

n_a_iterations = 10

mhp = MoralHazardProblem(cfg)


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


def plot_wage_functions(
    filename, y_grid, wage_functions, reservation_wage_grid, foa_flags, title
):
    cmap = COLORMAP
    norm = plt.Normalize(reservation_wage_grid.min(), reservation_wage_grid.max())

    fig, ax = plt.subplots(figsize=(6, 3.375))  # 6 inches wide, 16:9 aspect ratio

    for i, wf in enumerate(wage_functions):
        ax.plot(
            y_grid, wf,
            color=cmap(norm(reservation_wage_grid[i])),
            linestyle="-" if foa_flags[i] else "--",
            alpha=0.6, linewidth=PLOT_LINEWIDTH
        )

    ax.set_xlabel("Output (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Wage (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Reservation Wage (USD 1,000s)")  # colorbar fix

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_agent_utilities(
    filename, action_grid_plot, agent_utilities, targets,
    reservation_wage_grid, foa_flags, title
):
    cmap = COLORMAP
    norm = plt.Normalize(reservation_wage_grid.min(), reservation_wage_grid.max())

    fig, ax = plt.subplots(figsize=(6, 3.375))  # 6 inches wide, 16:9 aspect ratio

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

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Reservation Wage (USD 1,000s)")  # colorbar fix

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_stacked_wage_and_utility(
    filename, y_grid, wage_functions, action_grid_plot, agent_utilities, targets,
    reservation_wage_grid, foa_flags, title_wage, title_utility
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
    # Restrict x-axis to match action_grid_plot range
    for i, wf in enumerate(wage_functions):
        # Only plot the portion of wage functions within the action range
        mask = (y_grid >= action_xmin) & (y_grid <= action_xmax)
        ax1.plot(
            y_grid[mask], wf[mask],
            color=cmap(norm(reservation_wage_grid[i])),
            linestyle="-" if foa_flags[i] else "--",
            alpha=0.6, linewidth=PLOT_LINEWIDTH
        )
    
    ax1.set_xlim(action_xmin, action_xmax)
    ax1.set_xlabel("Output (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel("Wage (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax1.set_title(title_wage, fontsize=TITLE_FONT_SIZE)
    ax1.tick_params(labelsize=FONT_SIZE)
    
    # BOTTOM PLOT: Agent Utility
    for i, U in enumerate(agent_utilities):
        ax2.plot(
            action_grid_plot, U,
            color=cmap(norm(reservation_wage_grid[i])),
            linestyle="-" if foa_flags[i] else "--",
            alpha=0.6, linewidth=PLOT_LINEWIDTH
        )
        a_star, u_star = targets[i]
        ax2.scatter(a_star, u_star, color=SCATTER_COLOR, s=MARKER_SIZE, zorder=5)
    
    # Add horizontal lines through each red dot (only when first order approach fails)
    for i, (a_star, u_star) in enumerate(targets):
        if not foa_flags[i]:
            ax2.axhline(y=u_star, color=HORIZONTAL_LINE_COLOR, linestyle='-', 
                       linewidth=HORIZONTAL_LINE_WIDTH, alpha=0.7, zorder=1)
    
    # Check which conditions exist for annotations
    has_holds = any(foa_flags)
    has_fails = any(not f for f in foa_flags)
    
    xlim, ylim = ax2.get_xlim(), ax2.get_ylim()
    x_text = xlim[0] + 0.08*(xlim[1]-xlim[0])
    y_text1 = ylim[1] - 0.05*(ylim[1]-ylim[0])
    y_text2 = ylim[1] - 0.12*(ylim[1]-ylim[0])
    
    text_holds = None
    text_fails = None
    
    if has_holds:
        text_holds = ax2.text(
            x_text, y_text1, "first order approach holds",
            fontsize=FONT_SIZE, va="top"
        )
    
    if has_fails:
        y_pos = y_text1 if not has_holds else y_text2
        text_fails = ax2.text(
            x_text, y_pos, "first order approach fails",
            fontsize=FONT_SIZE, va="top"
        )
    
    # Compute arrow anchor positions
    if has_holds and has_fails:
        (x1, y1), (x2, y2) = arrow_positions_for_labels(ax2, text_holds, text_fails)
    elif has_holds:
        fig_temp = ax2.figure
        fig_temp.canvas.draw()
        bbox = text_holds.get_window_extent(fig_temp.canvas.get_renderer())
        trans = ax2.transData.inverted()
        x1, y1 = trans.transform((bbox.x1, bbox.y0 + bbox.height/2))
        xlim = ax2.get_xlim()
        offset = (xlim[1] - xlim[0]) * 0.005  # reduced from 0.015 to move arrow origins left
        x1 = x1 + offset
        x2, y2 = None, None
    elif has_fails:
        fig_temp = ax2.figure
        fig_temp.canvas.draw()
        bbox = text_fails.get_window_extent(fig_temp.canvas.get_renderer())
        trans = ax2.transData.inverted()
        x2, y2 = trans.transform((bbox.x1, bbox.y0 + bbox.height/2))
        xlim = ax2.get_xlim()
        offset = (xlim[1] - xlim[0]) * 0.005  # reduced from 0.015 to move arrow origins left
        x2 = x2 + offset
        x1, y1 = None, None
    
    # Draw arrows to appropriate text boxes
    for i, (a_star, u_star) in enumerate(targets):
        if foa_flags[i] and has_holds:
            ax2.annotate("", xy=(a_star, u_star), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=ARROW_LINEWIDTH, alpha=1.0))
        elif not foa_flags[i] and has_fails:
            ax2.annotate("", xy=(a_star, u_star), xytext=(x2, y2),
                        arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, lw=ARROW_LINEWIDTH, alpha=1.0))
    
    # Set x-axis limits to match the top plot exactly
    ax2.set_xlim(action_xmin, action_xmax)
    ax2.set_xlabel("Action (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax2.set_ylabel("Agent expected utility (certain equivalent, USD 1,000s)", fontsize=LABEL_FONT_SIZE)
    ax2.set_title(title_utility, fontsize=TITLE_FONT_SIZE)
    ax2.tick_params(labelsize=FONT_SIZE)
    
    # Set y-axis ticks at natural certain equivalent values for bottom plot
    ylim = ax2.get_ylim()
    u_min, u_max = ylim
    ce_min = mhp.k(u_min)
    ce_max = mhp.k(u_max)
    start_tick = np.ceil(ce_min / 10) * 10
    end_tick = np.floor(ce_max / 10) * 10
    
    if start_tick <= end_tick:
        ce_ticks = np.arange(start_tick, end_tick + 10, 10)
    else:
        ce_ticks = np.array([])
    
    if len(ce_ticks) > 0:
        u_ticks = utility_cfg["u"](ce_ticks)
        ax2.set_yticks(u_ticks)
        ax2.set_yticklabels([f'{ce:.0f}' for ce in ce_ticks], fontsize=FONT_SIZE)
    
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
def compute_principal_results():
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


def compute_cost_minimization_results(intended_action):
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


# -------------------------------------------------------------
# PART 1 — Principal Problem
# -------------------------------------------------------------
(
    wage_pp,
    util_pp,
    a_star_pp,
    u_star_pp,
    foa_pp,
    Ew_pp,
    Ew_rel_pp,
) = compute_principal_results()

y_grid = mhp._y_grid

plot_wage_functions(
    "figures/pp_wage.png",
    y_grid,
    wage_pp,
    reservation_wage_grid,
    foa_pp,
    "Optimal Contract"
)

plot_agent_utilities(
    "figures/pp_utility.png",
    action_grid_plot,
    util_pp,
    list(zip(a_star_pp, u_star_pp)),
    reservation_wage_grid,
    foa_pp,
    "Agent Utility vs Action given Optimal Contract"
)

plot_stacked_wage_and_utility(
    "figures/pp_stacked.png",
    y_grid,
    wage_pp,
    action_grid_plot,
    util_pp,
    list(zip(a_star_pp, u_star_pp)),
    reservation_wage_grid,
    foa_pp,
    "Optimal Contract",
    "Agent Utility vs Action given Optimal Contract"
)


# -------------------------------------------------------------
# PART 2 — Cost Minimization
# -------------------------------------------------------------
intended_action = first_best_effort

(
    wage_cm,
    util_cm,
    foa_cm,
    util_intended_cm
) = compute_cost_minimization_results(intended_action)

plot_wage_functions(
    "figures/cm_wage.png",
    y_grid,
    wage_cm,
    reservation_wage_grid,
    foa_cm,
    "Optimal Contract"
)

plot_agent_utilities(
    "figures/cm_utility.png",
    action_grid_plot,
    util_cm,
    [(intended_action, u) for u in util_intended_cm],
    reservation_wage_grid,
    foa_cm,
    "Agent Utility vs Action given Optimal Contract"
)

plot_stacked_wage_and_utility(
    "figures/cm_stacked.png",
    y_grid,
    wage_cm,
    action_grid_plot,
    util_cm,
    [(intended_action, u) for u in util_intended_cm],
    reservation_wage_grid,
    foa_cm,
    "Optimal Contract",
    "Agent Utility vs Action given Optimal Contract"
)


# -------------------------------------------------------------
# PART 3 — Pareto Frontier
# -------------------------------------------------------------
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
ax.plot(pareto_ru, Ew_pf, label="Full Problem", linewidth=PLOT_LINEWIDTH)
ax.plot(pareto_ru, Ew_pf_rel, label="Relaxed Problem", linewidth=PLOT_LINEWIDTH, linestyle="--")
ax.set_xlabel("Agent expected utility (certain equivalent, USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax.set_ylabel("Expected Wages (USD 1,000s)", fontsize=LABEL_FONT_SIZE)
ax.set_title("Pareto Frontier: Expected Wages vs Agent Expected Utility", fontsize=TITLE_FONT_SIZE)
ax.tick_params(labelsize=FONT_SIZE)
ax.legend(fontsize=FONT_SIZE)

# Set x-axis ticks at natural certain equivalent values from reservation_wage_grid_pareto
# Use a subset of reservation_wage_grid_pareto for nice tick locations
# Select evenly spaced values that are nice round numbers
ce_min, ce_max = reservation_wage_grid_pareto.min(), reservation_wage_grid_pareto.max()
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
fig.savefig("figures/pareto_frontier.png", dpi=300)
plt.close(fig)
