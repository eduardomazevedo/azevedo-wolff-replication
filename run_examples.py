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
def C(a): return theta * a ** 2 / 2

def Cprime(a): return theta * a

utility_cfg = make_utility_cfg("log", w0=initial_wealth)
reservation_wage_grid = np.linspace(-1.0, 50.0, 5)
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
        "problem_params": {**utility_cfg, **dist_cfg, "C": C, "Cprime": Cprime},
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
        dirname=f"figures/log-gaussian-sigma={sigma}"
    )


# CARA gaussian example
theta_cara = theta * 10.0
def C_cara(a): return theta_cara * a ** 2 / 2

def Cprime_cara(a): return theta_cara * a

utility_cfg_cara = make_utility_cfg("cara", w0=initial_wealth, alpha=1.0 / initial_wealth)

sigma = 50.0
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)

cfg_cara = {
    "problem_params": {**utility_cfg_cara, **dist_cfg, "C": C_cara, "Cprime": Cprime_cara},
    "computational_params": comp_cfg
}

mhp_cara = MoralHazardProblem(cfg_cara)

# Generate all figures
make_example_figures(
    mhp=mhp_cara,
    utility_cfg=utility_cfg_cara,
    n_a_iterations=n_a_iterations,
    reservation_wage_grid=reservation_wage_grid,
    reservation_wage_grid_pareto=reservation_wage_grid_pareto,
    a_min=a_min,
    a_max=a_max,
    action_grid_plot=action_grid_plot,
    intended_action=first_best_effort,
    dirname=f"figures/cara-gaussian-sigma={sigma}"
)



# t distribution example for sigma = 10 and sigma = 20
for sigma in [10.0, 20.0, 50.0]:
    dist_cfg = make_distribution_cfg("Student_t", sigma=sigma, nu=1.15)
    comp_cfg = {
        "distribution_type": "continuous",
        "y_min": a_min - 10 * sigma,
        "y_max": a_max + 10 * sigma,
        "n": 201,  # must be odd
    }
    cfg = {
        "problem_params": {**utility_cfg, **dist_cfg, "C": C, "Cprime": Cprime},
        "computational_params": comp_cfg
    }
    
    mhp = MoralHazardProblem(cfg)
    
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
        dirname=f"figures/log-t-sigma={sigma}"
    )




# log-Poisson example
a_min, a_max = 0.0, 30.0
action_grid_plot = np.linspace(a_min, a_max, 100)

first_best_effort_poisson = 20.0
theta_poisson = 1.0 / first_best_effort_poisson / (first_best_effort_poisson + initial_wealth)
def C_poisson(a): return theta_poisson * a ** 2 / 2
def Cprime_poisson(a): return theta_poisson * a

dist_cfg = make_distribution_cfg("poisson")
comp_cfg = {
    "distribution_type": "discrete",
    "y_min": 0.0,
    "y_max": a_max + 6 * np.floor(np.sqrt(a_max)),
    "step_size": 1.0,
}

cfg = {
    "problem_params": {**utility_cfg, **dist_cfg, "C": C_poisson, "Cprime": Cprime_poisson},
    "computational_params": comp_cfg
}

mhp = MoralHazardProblem(cfg)

make_example_figures(
    mhp=mhp,
    utility_cfg=utility_cfg,
    n_a_iterations=n_a_iterations,
    reservation_wage_grid=reservation_wage_grid,
    reservation_wage_grid_pareto=reservation_wage_grid_pareto,
    a_min=a_min,
    a_max=a_max,
    action_grid_plot=action_grid_plot,
    intended_action=first_best_effort_poisson,
    dirname="figures/log-poisson"
)


#log-exponential example
a_min, a_max = 10.0, 180.0
action_grid_plot = np.linspace(a_min, a_max, 100)
intended_action = 50.0 # Action of 100 was extremely not worth it for principal.

dist_cfg = make_distribution_cfg("exponential")
comp_cfg = {
    "distribution_type": "continuous",
    "y_min": 0.1,
    "y_max": a_max + 6 * np.floor(np.sqrt(a_max)),
    "n": 201,  # must be odd
}

cfg = {
    "problem_params": {**utility_cfg, **dist_cfg, "C": C, "Cprime": Cprime},
    "computational_params": comp_cfg
}

mhp = MoralHazardProblem(cfg)

make_example_figures(
    mhp=mhp,
    utility_cfg=utility_cfg,
    n_a_iterations=n_a_iterations,
    reservation_wage_grid=reservation_wage_grid,
    reservation_wage_grid_pareto=reservation_wage_grid_pareto,
    a_min=a_min,
    a_max=a_max,
    action_grid_plot=action_grid_plot,
    intended_action=intended_action,
    dirname="figures/log-exponential"
)