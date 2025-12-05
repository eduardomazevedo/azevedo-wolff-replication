"""
Gaussian log utility example — refactored and cleaned version.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------
# Primitives and configuration
# -------------------------------------------------------------
initial_wealth = 50
sigma = 10.0
first_best_effort = 100
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

reservation_wage_grid = np.linspace(-1.0, 50.0, 10)
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
    offset = (xlim[1] - xlim[0]) * 0.015

    return (x1 + offset, y1), (x2 + offset, y2)


def plot_wage_functions(
    filename, y_grid, wage_functions, reservation_wage_grid, foa_flags, title
):
    cmap = plt.cm.viridis
    norm = plt.Normalize(reservation_wage_grid.min(), reservation_wage_grid.max())

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, wf in enumerate(wage_functions):
        ax.plot(
            y_grid, wf,
            color=cmap(norm(reservation_wage_grid[i])),
            linestyle="-" if foa_flags[i] else "--",
            alpha=0.6, linewidth=0.8
        )

    ax.set_xlabel("y")
    ax.set_ylabel("Wage Function")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Reservation Wage")  # colorbar fix

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_agent_utilities(
    filename, action_grid_plot, agent_utilities, targets,
    reservation_wage_grid, foa_flags, title
):
    cmap = plt.cm.viridis
    norm = plt.Normalize(reservation_wage_grid.min(), reservation_wage_grid.max())

    fig, ax = plt.subplots(figsize=(10, 6))

    # curves and red dots
    for i, U in enumerate(agent_utilities):
        ax.plot(
            action_grid_plot, U,
            color=cmap(norm(reservation_wage_grid[i])),
            linestyle="-" if foa_flags[i] else "--",
            alpha=0.6, linewidth=0.8
        )
        a_star, u_star = targets[i]
        ax.scatter(a_star, u_star, color="red", s=5, zorder=5)

    # label positions
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    # Add horizontal lines through each red dot (only when first order approach fails)
    for i, (a_star, u_star) in enumerate(targets):
        if not foa_flags[i]:
            ax.axhline(y=u_star, color='gray', linestyle='-', linewidth=0.5, alpha=0.7, zorder=1)
    
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
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10, va="top"
        )
    
    if has_fails:
        y_pos = y_text1 if not has_holds else y_text2
        text_fails = ax.text(
            x_text, y_pos, "first order approach fails",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10, va="top"
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
        offset = (xlim[1] - xlim[0]) * 0.015
        x1 = x1 + offset
        x2, y2 = None, None
    elif has_fails:
        fig = ax.figure
        fig.canvas.draw()
        bbox = text_fails.get_window_extent(fig.canvas.get_renderer())
        trans = ax.transData.inverted()
        x2, y2 = trans.transform((bbox.x1, bbox.y0 + bbox.height/2))
        xlim = ax.get_xlim()
        offset = (xlim[1] - xlim[0]) * 0.015
        x2 = x2 + offset
        x1, y1 = None, None

    # Draw arrows to appropriate text boxes
    for i, (a_star, u_star) in enumerate(targets):
        if foa_flags[i] and has_holds:
            ax.annotate("", xy=(a_star, u_star), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.8, alpha=0.5))
        elif not foa_flags[i] and has_fails:
            ax.annotate("", xy=(a_star, u_star), xytext=(x2, y2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.8, alpha=0.5))

    ax.set_xlabel("Action")
    ax.set_ylabel("Agent Utility")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Reservation Wage")  # colorbar fix

    fig.tight_layout()
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
    "Optimal Wage Function vs y"
)

plot_agent_utilities(
    "figures/pp_utility.png",
    action_grid_plot,
    util_pp,
    list(zip(a_star_pp, u_star_pp)),
    reservation_wage_grid,
    foa_pp,
    "Agent Utility vs Action"
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
    "Optimal Wage Function vs y (Cost Minimization)"
)

plot_agent_utilities(
    "figures/cm_utility.png",
    action_grid_plot,
    util_cm,
    [(intended_action, u) for u in util_intended_cm],
    reservation_wage_grid,
    foa_cm,
    "Agent Utility vs Action (Cost Minimization)"
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

plt.figure(figsize=(10, 6))
plt.plot(pareto_ru, Ew_pf, label="Full Problem", linewidth=2)
plt.plot(pareto_ru, Ew_pf_rel, label="Relaxed Problem", linewidth=2, linestyle="--")
plt.xlabel("Reservation Utility")
plt.ylabel("Expected Wages")
plt.title("Pareto Frontier: Expected Wages vs Reservation Utility")
plt.legend()
plt.tight_layout()
plt.savefig("figures/pareto_frontier.png", dpi=300)
plt.close()
