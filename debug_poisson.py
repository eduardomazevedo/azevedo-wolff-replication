"""
Make figures for the examples.
"""
import numpy as np
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg
from figures import make_example_figures

# -------------------------------------------------------------
# Log-gaussian example
# -------------------------------------------------------------
initial_wealth = 50
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)
theta_cara = theta * 10.0
def C_cara(a): return theta_cara * a ** 2 / 2

def Cprime_cara(a): return theta_cara * a

utility_cfg = make_utility_cfg("cara", w0=initial_wealth, alpha=1.0 / initial_wealth)
reservation_wage_grid = np.linspace(-1.0, 50.0, 20)
reservation_wage_grid_pareto = np.linspace(-20.0, 50.0, 100)
a_min, a_max = 0.0, 180.0
action_grid_plot = np.linspace(a_min, a_max, 100)
n_a_iterations = 10

for sigma in [10.0, 20.0, 50.0]:
    dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)
    
    comp_cfg = {
        "distribution_type": "continuous",
        "y_min": 0.0   - 6 * sigma,
        "y_max": 180.0 + 6 * sigma,
        "n": 201,  # must be odd
    }
    
    cfg = {
        "problem_params": {**utility_cfg, **dist_cfg, "C": C_cara, "Cprime": Cprime_cara},
        "computational_params": comp_cfg
    }
    
    mhp = MoralHazardProblem(cfg)
    
    # Generate all figures
    make_example_figures(
        mhp=mhp,
        utility_cfg=utility_cfg,
        n_a_iterations=n_a_iterations,
        reservation_wage_grid=reservation_wage_grid,
        reservation_wage_grid_pareto=reservation_wage_grid_pareto,
        a_min=a_min,
        a_max=a_max,
        action_grid_plot=action_grid_plot,
        intended_action=first_best_effort,
        dirname=f"figures/cara-gaussian-sigma={sigma}"
    )
