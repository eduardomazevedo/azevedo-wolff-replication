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

reservation_wage_grid = np.linspace(-1.0, 50.0, 4)
a_min = 0.0
a_max = 180.0
a_ic_lb = 0.0
a_ic_ub = 180.0
action_grid_plot = np.linspace(a_min, a_max, 100)
y_grid_plot = np.linspace(0.0, 180.0, 100)

mhp = MoralHazardProblem(cfg)

# Part 1: Principal problem
# Loop over each reservation utility in the grid
reservation_utility_grid = [utility_cfg["u"](reservation_wage) for reservation_wage in reservation_wage_grid]
wage_functions = []
agent_utilities = []
optimal_actions = []
optimal_action_utilities = []
first_order_approach_holds_list = []
expected_wages = []
expected_wages_relaxed = []

for i in range(len(reservation_wage_grid)):
    results_principal = mhp.solve_principal_problem(
        revenue_function=lambda a: a,
        reservation_utility=reservation_utility_grid[i],
        a_min=a_min,
        a_max=a_max,
        a_ic_lb=a_ic_lb,
        a_ic_ub=a_ic_ub
    )

    results_relaxed = mhp.solve_principal_problem(
        revenue_function=lambda a: a,
        reservation_utility=reservation_utility_grid[i],
        a_min=a_min,
        a_max=a_max,
        a_ic_lb=a_ic_lb,
        a_ic_ub=a_ic_ub,
        n_a_iterations=0,
    )

    optimal_contract = results_principal.cmp_result.optimal_contract
    agent_utility = mhp.U(optimal_contract, action_grid_plot)
    wage_function = mhp.k(optimal_contract)
    optimal_action = results_principal.optimal_action
    optimal_action_u = mhp.U(optimal_contract, optimal_action)
    first_order_approach_holds = results_principal.cmp_result.first_order_approach_holds
    expected_wages.append(float(results_principal.cmp_result.constraints['Ewage']))
    expected_wages_relaxed.append(float(results_relaxed.cmp_result.constraints['Ewage']))
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
plt.xlim(a_min, a_max)
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
ax = plt.gca()
for i, agent_utility in enumerate(agent_utilities):
    color = cmap(norm(reservation_wage_grid[i]))
    linestyle = '-' if first_order_approach_holds_list[i] else '--'
    plt.plot(action_grid_plot, agent_utility, alpha=0.6, linewidth=0.8, color=color, linestyle=linestyle)
    plt.scatter(optimal_actions[i], optimal_action_utilities[i], color='red', s=5, zorder=5)

# Get axis limits for text positioning
xlim = ax.get_xlim()
ylim = ax.get_ylim()
x_text = xlim[0] + 0.08 * (xlim[1] - xlim[0])  # Position moved to the right
y_text_holds = ylim[1] - 0.05 * (ylim[1] - ylim[0])  # First line
y_text_fails = ylim[1] - 0.12 * (ylim[1] - ylim[0])  # Second line

# Add text boxes and get their positions
text_holds = plt.text(x_text, y_text_holds, 'first order approach holds', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontsize=10, verticalalignment='top')
text_fails = plt.text(x_text, y_text_fails, 'first order approach fails', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontsize=10, verticalalignment='top')

# Get bounding boxes of text objects to find middle right positions
# Need to draw first to get accurate bbox
fig = plt.gcf()
fig.canvas.draw()
bbox_holds = text_holds.get_window_extent(fig.canvas.get_renderer())
bbox_fails = text_fails.get_window_extent(fig.canvas.get_renderer())

# Convert from display coordinates to data coordinates
trans = ax.transData.inverted()
x_holds_right, y_holds_mid = trans.transform((bbox_holds.x1, bbox_holds.y0 + bbox_holds.height/2))
x_fails_right, y_fails_mid = trans.transform((bbox_fails.x1, bbox_fails.y0 + bbox_fails.height/2))

# Add offset to push arrows a bit more to the right (about 3 characters worth)
# Estimate character width in data coordinates based on font size
char_width_data = (xlim[1] - xlim[0]) * 0.015  # Roughly 3 characters worth
x_holds_right += char_width_data
x_fails_right += char_width_data

# Draw arrows from text boxes to corresponding red dots
for i in range(len(optimal_actions)):
    if first_order_approach_holds_list[i]:
        # Arrow from "holds" text middle right to red dot
        plt.annotate('', xy=(optimal_actions[i], optimal_action_utilities[i]),
                    xytext=(x_holds_right, y_holds_mid),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8, alpha=0.5))
    else:
        # Arrow from "fails" text middle right to red dot
        plt.annotate('', xy=(optimal_actions[i], optimal_action_utilities[i]),
                    xytext=(x_fails_right, y_fails_mid),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8, alpha=0.5))

plt.xlabel('Action')
plt.ylabel('Agent Utility')
plt.title('Agent Utility vs Action')
plt.xlim(a_min, a_max)
plt.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Reservation Wage')
plt.tight_layout()
plt.savefig('figures/pp_utility.png', dpi=300)
plt.close()

print(f"First order approach holds: {first_order_approach_holds_list}")

# Check FOA behavior in the grid
foa_holds_count = sum(first_order_approach_holds_list)
foa_fails_count = len(first_order_approach_holds_list) - foa_holds_count

print(f"\nFOA analysis in grid:")
print(f"  - FOA holds for {foa_holds_count} reservation wages")
print(f"  - FOA fails for {foa_fails_count} reservation wages")

# Check if FOA fails at the lowest wages
if foa_fails_count > 0:
    fails_at_lowest = not first_order_approach_holds_list[0]  # Check first (lowest) wage
    print(f"  - FOA fails at lowest reservation wage: {fails_at_lowest}")
    
    # Find indices where FOA fails
    fails_indices = [i for i, holds in enumerate(first_order_approach_holds_list) if not holds]
    if fails_indices:
        print(f"  - FOA fails at indices: {fails_indices}")
        print(f"  - Corresponding reservation wages: {reservation_wage_grid[fails_indices]}")

# Binary search to find highest reservation wage where FOA fails
if foa_holds_count > 0 and foa_fails_count > 0:
    print(f"\nPerforming binary search to find highest reservation wage where FOA fails...")
    
    # Find bounds: we know FOA fails at some point and holds at others
    # We'll search in a wider range than the grid
    min_wage = reservation_wage_grid.min()
    max_wage = reservation_wage_grid.max()
    
    # Expand search range slightly to ensure we capture the boundary
    search_min = min_wage - 1.0
    search_max = max_wage + 1.0
    tolerance = 1e-6
    
    def check_foa(reservation_wage):
        """Check if FOA holds for a given reservation wage."""
        reservation_utility = utility_cfg["u"](reservation_wage)
        results_principal = mhp.solve_principal_problem(
            revenue_function=lambda a: a,
            reservation_utility=reservation_utility,
            a_min=a_min,
            a_max=a_max,
            a_ic_lb=a_ic_lb,
            a_ic_ub=a_ic_ub
        )
        return results_principal.cmp_result.first_order_approach_holds
    
    # Binary search: find the highest wage where FOA fails
    # Strategy: find the transition point where FOA goes from fails to holds
    # We'll search between the lowest wage where we know FOA fails and highest where it holds
    
    # Find the range where we know FOA fails and holds
    fails_wages = [reservation_wage_grid[i] for i in range(len(reservation_wage_grid)) 
                   if not first_order_approach_holds_list[i]]
    holds_wages = [reservation_wage_grid[i] for i in range(len(reservation_wage_grid)) 
                   if first_order_approach_holds_list[i]]
    
    if fails_wages and holds_wages:
        # We have both fails and holds, so there's a transition point
        low_bound = max(fails_wages)  # Highest wage we know FOA fails
        high_bound = min(holds_wages)  # Lowest wage we know FOA holds
        
        # Binary search in this range
        low = low_bound
        high = high_bound
        tolerance = 1e-6
        
        # Find the highest wage where FOA still fails
        while high - low > tolerance:
            mid = (low + high) / 2
            if check_foa(mid):
                # FOA holds at mid, so the boundary is below mid
                high = mid
            else:
                # FOA fails at mid, so the boundary might be above mid
                low = mid
        
        highest_fail_wage = low
        print(f"  Highest reservation wage where FOA fails: {highest_fail_wage:.6f}")
        # Verify: check slightly above
        verify_wage = highest_fail_wage + tolerance
        verify_holds = check_foa(verify_wage)
        print(f"  Verification: At reservation wage {verify_wage:.6f}, FOA holds: {verify_holds}")
    elif fails_wages and not holds_wages:
        # FOA fails for all wages in grid, check if it holds at higher wages
        low = max(fails_wages)
        high = search_max
        tolerance = 1e-6
        
        # Check if FOA holds at the high end
        if check_foa(high):
            # Binary search for transition
            while high - low > tolerance:
                mid = (low + high) / 2
                if check_foa(mid):
                    high = mid
                else:
                    low = mid
            highest_fail_wage = low
            print(f"  Highest reservation wage where FOA fails: {highest_fail_wage:.6f}")
        else:
            print(f"  FOA fails for all reservation wages up to {high:.6f}")
            print(f"  Highest reservation wage checked where FOA fails: {low:.6f}")
    else:
        print("  Cannot determine highest fail wage from grid data.")
else:
    print(f"\nCannot perform binary search:")
    if foa_holds_count == 0:
        print("  - FOA fails for all reservation wages in the grid")
    if foa_fails_count == 0:
        print("  - FOA holds for all reservation wages in the grid")


# Part 2: Cost minimization problem
# Loop over each reservation utility in the grid
intended_action = first_best_effort
wage_functions = []
agent_utilities = []
first_order_approach_holds_list = []
utility_at_intended_action = []

for i in range(len(reservation_wage_grid)):
    results = mhp.solve_cost_minimization_problem(
    intended_action=intended_action,
    reservation_utility=reservation_utility_grid[i],
    a_ic_lb=0.0,
    a_ic_ub=100.0
    )

    optimal_contract = results.optimal_contract
    agent_utility = mhp.U(optimal_contract, action_grid_plot)
    wage_function = mhp.k(optimal_contract)
    utility_at_intended = mhp.U(optimal_contract, intended_action)
    first_order_approach_holds = results.first_order_approach_holds

    wage_functions.append(wage_function)
    agent_utilities.append(agent_utility)
    utility_at_intended_action.append(utility_at_intended)
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
plt.title('Optimal Wage Function vs y (Cost Minimization)')
plt.xlim(a_min, a_max)
plt.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Reservation Wage')
plt.tight_layout()
plt.savefig('figures/cm_wage.png', dpi=300)
plt.close()

# Plot 2: agent_utility vs action_grid_plot
plt.figure(figsize=(10, 6))
ax = plt.gca()
for i, agent_utility in enumerate(agent_utilities):
    color = cmap(norm(reservation_wage_grid[i]))
    linestyle = '-' if first_order_approach_holds_list[i] else '--'
    plt.plot(action_grid_plot, agent_utility, alpha=0.6, linewidth=0.8, color=color, linestyle=linestyle)
    plt.scatter(intended_action, utility_at_intended_action[i], color='red', s=5, zorder=5)

# Get axis limits for text positioning
xlim = ax.get_xlim()
ylim = ax.get_ylim()
x_text = xlim[0] + 0.08 * (xlim[1] - xlim[0])  # Position moved to the right
y_text_holds = ylim[1] - 0.05 * (ylim[1] - ylim[0])  # First line
y_text_fails = ylim[1] - 0.12 * (ylim[1] - ylim[0])  # Second line

# Add text boxes and get their positions
text_holds = plt.text(x_text, y_text_holds, 'first order approach holds', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontsize=10, verticalalignment='top')
text_fails = plt.text(x_text, y_text_fails, 'first order approach fails', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontsize=10, verticalalignment='top')

# Get bounding boxes of text objects to find middle right positions
# Need to draw first to get accurate bbox
fig = plt.gcf()
fig.canvas.draw()
bbox_holds = text_holds.get_window_extent(fig.canvas.get_renderer())
bbox_fails = text_fails.get_window_extent(fig.canvas.get_renderer())

# Convert from display coordinates to data coordinates
trans = ax.transData.inverted()
x_holds_right, y_holds_mid = trans.transform((bbox_holds.x1, bbox_holds.y0 + bbox_holds.height/2))
x_fails_right, y_fails_mid = trans.transform((bbox_fails.x1, bbox_fails.y0 + bbox_fails.height/2))

# Add offset to push arrows a bit more to the right (about 3 characters worth)
# Estimate character width in data coordinates based on font size
char_width_data = (xlim[1] - xlim[0]) * 0.015  # Roughly 3 characters worth
x_holds_right += char_width_data
x_fails_right += char_width_data

# Draw arrows from text boxes to corresponding red dots
for i in range(len(utility_at_intended_action)):
    if first_order_approach_holds_list[i]:
        # Arrow from "holds" text middle right to red dot
        plt.annotate('', xy=(intended_action, utility_at_intended_action[i]),
                    xytext=(x_holds_right, y_holds_mid),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8, alpha=0.5))
    else:
        # Arrow from "fails" text middle right to red dot
        plt.annotate('', xy=(intended_action, utility_at_intended_action[i]),
                    xytext=(x_fails_right, y_fails_mid),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8, alpha=0.5))

plt.xlabel('Action')
plt.ylabel('Agent Utility')
plt.title('Agent Utility vs Action (Cost Minimization)')
plt.xlim(a_min, a_max)
plt.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Reservation Wage')
plt.tight_layout()
plt.savefig('figures/cm_utility.png', dpi=300)
plt.close()


# Part 3: Pareto frontier
# Use cost minimization problem with 100 reservation utilities for detailed plot
pareto_reservation_utility_grid = np.linspace(
    utility_cfg["u"](reservation_wage_grid.min()),
    utility_cfg["u"](reservation_wage_grid.max()),
    100
)
expected_wages_pareto = []
expected_wages_relaxed_pareto = []

for reservation_utility in pareto_reservation_utility_grid:
    # Full problem
    results = mhp.solve_cost_minimization_problem(
        intended_action=intended_action,
        reservation_utility=reservation_utility,
        a_ic_lb=0.0,
        a_ic_ub=100.0
    )
    expected_wages_pareto.append(float(results.constraints['Ewage']))
    
    # Relaxed problem
    results_relaxed = mhp.solve_cost_minimization_problem(
        intended_action=intended_action,
        reservation_utility=reservation_utility,
        a_ic_lb=0.0,
        a_ic_ub=100.0,
        n_a_iterations=0
    )
    expected_wages_relaxed_pareto.append(float(results_relaxed.constraints['Ewage']))

# Plot expected wages vs reservation utility
plt.figure(figsize=(10, 6))
plt.plot(pareto_reservation_utility_grid, expected_wages_pareto, label='Expected Wages (Full Problem)', linewidth=2)
plt.plot(pareto_reservation_utility_grid, expected_wages_relaxed_pareto, label='Expected Wages (Relaxed Problem)', linewidth=2, linestyle='--')
plt.xlabel('Reservation Utility')
plt.ylabel('Expected Wages')
plt.title('Pareto Frontier: Expected Wages vs Reservation Utility (Cost Minimization)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('figures/pareto_frontier.png', dpi=300)
plt.close()
