"""
Gaussian log utility example.
Graphs plotted:
  - Principal problem
    - Optimal wage function vs y.
    - Agent utility vs a.
  - Cost minimization problem (using intended action = first best effort)
    - Optimal effort vs y.
    - Agent utility vs a.
    - Pareto frontier (expected wage vs agent reservation utility, for relaxed problem and non relaxed problem).
"""

# Load Modules
import numpy as np
import matplotlib.pyplot as plt
import os
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

os.makedirs('figures', exist_ok=True)

# ---- primitives ----
initial_wealth = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)

def C(a): return theta * a ** 2 / 2
def Cprime(a): return theta * a

# Create utility functions (log utility with baseline wealth x0)
utility_cfg = make_utility_cfg("log", w0=initial_wealth)
# Create distribution functions (gaussian with sigma)
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)
comp_cfg = {
    "distribution_type": "continuous",
    "y_min": 0.0   - 3 * sigma,
    "y_max": 180.0 + 3 * sigma,
    "n": 201,  # must be odd
}

cfg = {
    "problem_params": {
        **utility_cfg,  # u, k, link_function
        **dist_cfg,     # f, score
        "C": C,
        "Cprime": Cprime,
    },
    "computational_params": comp_cfg
}

reservation_wage_grid = np.linspace(-1.0, 10.0, 10)
a_min = 0.0
a_max = 180.0
a_ic_lb = 0.0
a_ic_ub = 180.0
action_grid_plot = np.linspace(a_min, a_max, 100)
y_grid_plot = np.linspace(0.0, 180.0, 100)

mhp = MoralHazardProblem(cfg)

# Part 1: Principal problem
# Loop over each reservation utility in the grid
wage_functions = []
agent_utilities = []
optimal_actions = []
optimal_action_utilities = []
first_order_approach_holds_list = []

for reservation_wage in reservation_wage_grid:
    reservation_utility = utility_cfg["u"](reservation_wage)
    
    results_principal = mhp.solve_principal_problem(
        revenue_function=lambda a: a,
        reservation_utility=reservation_utility,
        a_min=a_min,
        a_max=a_max,
        a_ic_lb=a_ic_lb,
        a_ic_ub=a_ic_ub
    )
    
    optimal_contract = results_principal.cmp_result.optimal_contract
    agent_utility = mhp.U(optimal_contract, action_grid_plot)
    wage_function = mhp.k(optimal_contract)
    optimal_action = results_principal.optimal_action
    optimal_action_u = mhp.U(optimal_contract, optimal_action)
    first_order_approach_holds = results_principal.cmp_result.first_order_approach_holds
    
    wage_functions.append(wage_function)
    agent_utilities.append(agent_utility)
    optimal_actions.append(optimal_action)
    optimal_action_utilities.append(optimal_action_u)
    first_order_approach_holds_list.append(first_order_approach_holds)

# Get y_grid for plotting
y_grid = mhp._y_grid

# Set up colormap for reservation wages
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=reservation_wage_grid.min(), vmax=reservation_wage_grid.max())

# Plot 1: wage function vs mhp._y_grid
plt.figure(figsize=(10, 6))
for i, wage_function in enumerate(wage_functions):
    color = cmap(norm(reservation_wage_grid[i]))
    linestyle = '-' if first_order_approach_holds_list[i] else '--'
    plt.plot(y_grid, wage_function, alpha=0.6, linewidth=0.8, color=color, linestyle=linestyle)
plt.xlabel('y')
plt.ylabel('Wage Function')
plt.title('Optimal Wage Function vs y')
plt.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Reservation Wage')
plt.tight_layout()
plt.savefig('figures/pp_wage.png', dpi=300)
plt.close()

# Plot 2: agent_utility vs action_grid_plot
plt.figure(figsize=(10, 6))
for i, agent_utility in enumerate(agent_utilities):
    color = cmap(norm(reservation_wage_grid[i]))
    linestyle = '-' if first_order_approach_holds_list[i] else '--'
    plt.plot(action_grid_plot, agent_utility, alpha=0.6, linewidth=0.8, color=color, linestyle=linestyle)
    plt.scatter(optimal_actions[i], optimal_action_utilities[i], color='red', s=5, zorder=5)
plt.xlabel('Action')
plt.ylabel('Agent Utility')
plt.title('Agent Utility vs Action')
plt.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Reservation Wage')
plt.tight_layout()
plt.savefig('figures/pp_utility.png', dpi=300)
plt.close()

print(f"First order approach holds: {first_order_approach_holds_list}")