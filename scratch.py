import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# --------------------
# Primitives
# --------------------
initial_wealth = 50
sigma = 10.0
first_best_effort = 100
theta = 1.0 / first_best_effort / (first_best_effort + initial_wealth)
reservation_wages = np.linspace(-1.0, 100.0, 10)

utility_cfg = make_utility_cfg("log", w0=initial_wealth)
dist_cfg = make_distribution_cfg("gaussian", sigma=sigma)

def C(a):
    return theta * a**2 / 2

def Cprime(a):
    return theta * a

computational_params = {
    "distribution_type": "continuous",
    "y_min": 0.0 - 3 * sigma,
    "y_max": 180.0 + 3 * sigma,
    "n": 201,
}

cfg = {
    "problem_params": {
        **utility_cfg,
        **dist_cfg,
        "C": C,
        "Cprime": Cprime,
    },
    "computational_params": computational_params,
}

mhp = MoralHazardProblem(cfg)
u_fun = cfg["problem_params"]["u"]

a_min = 0.0
a_max = 130.0
a_init = 100.0
a_hat = np.array([0.0])

# --------------------
# Output DataFrames
# --------------------
principal_cols = [
    "reservation_wage",
    "optimal_action",
    "profit",
    "lam",
    "mu",
    "mu_hat",
]

agent_cols = [
    "reservation_wage",
    "cost",
    "lam",
    "mu",
    "mu_hat",
]

principal_records = []
agent_records = []

# --------------------
# Solve for each reservation wage
# --------------------
for w in reservation_wages:

    # --- Principal problem ---
    pr = mhp.solve_principal_problem(
        revenue_function=lambda a: a,
        reservation_utility=u_fun(w),
        a_min=a_min,
        a_max=a_max,
        a_init=a_init,
        a_hat=a_hat,
    )

    pr_mult = pr.multipliers
    lam = pr_mult["lam"]
    mu = pr_mult["mu"]
    mu_hat = float(pr_mult["mu_hat"][0])

    principal_records.append({
        "reservation_wage": w,
        "optimal_action": pr.optimal_action,
        "profit": pr.profit,
        "lam": lam,
        "mu": mu,
        "mu_hat": mu_hat,
        "optimal_contract": pr.optimal_contract,
    })

    # --- Cost-minimization problem ---
    cm = mhp.solve_cost_minimization_problem(
        intended_action=first_best_effort,
        reservation_utility=u_fun(w),
        a_hat=a_hat,
    )

    cm_mult = cm.multipliers
    cm_lam = cm_mult["lam"]
    cm_mu = cm_mult["mu"]
    cm_mu_hat = float(cm_mult["mu_hat"][0])

    agent_records.append({
        "reservation_wage": w,
        "cost": cm.expected_wage,
        "lam": cm_lam,
        "mu": cm_mu,
        "mu_hat": cm_mu_hat,
        "optimal_contract": cm.optimal_contract,
    })

# --------------------
# Create DataFrames (excluding optimal_contract columns for CSV)
# --------------------
df_principal = pd.DataFrame([
    {col: record[col] for col in principal_cols}
    for record in principal_records
])
df_cost_minimization = pd.DataFrame([
    {col: record[col] for col in agent_cols}
    for record in agent_records
])

# --------------------
# Save DataFrames
# --------------------
os.makedirs("./tables", exist_ok=True)
df_principal.to_csv("./tables/principal.csv", index=False)
df_cost_minimization.to_csv("./tables/cost_minimization.csv", index=False)

# --------------------
# Create Plots
# --------------------
os.makedirs("./figures", exist_ok=True)

# Get grids
y_grid = mhp.y_grid
a_grid = np.linspace(0, 140, 100)

# Get colormap for reservation wages
w_min = min(record["reservation_wage"] for record in principal_records)
w_max = max(record["reservation_wage"] for record in principal_records)
cmap = plt.cm.viridis

# Plot 1: Principal - Wage schedule k(v*(y)) vs y
fig, ax = plt.subplots(figsize=(8, 5))
for record in principal_records:
    w = record["reservation_wage"]
    v = record["optimal_contract"]
    wage = mhp.k(v)
    color = cmap((w - w_min) / (w_max - w_min))
    ax.plot(y_grid, wage, color=color, alpha=0.6, linewidth=0.5)
ax.set_xlabel("Output ($y$)")
ax.set_ylabel("Wage")
ax.set_title("Principal: Wage Schedule $k(v^*(y))$ vs $y$")
plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max)),
    ax=ax,
    label="Reservation Wage"
)
plt.tight_layout()
plt.savefig("./figures/principal_wage_schedule_vs_y.png")
plt.close()

# Plot 2: Principal - Utility function U(v, a) vs a
fig, ax = plt.subplots(figsize=(8, 5))
for record in principal_records:
    w = record["reservation_wage"]
    v = record["optimal_contract"]
    U_grid = mhp.U(v, a_grid)
    color = cmap((w - w_min) / (w_max - w_min))
    ax.plot(a_grid, U_grid, color=color, alpha=0.6, linewidth=0.5)
ax.set_xlabel("Action ($a$)")
ax.set_ylabel("Utility")
ax.set_title("Principal: Utility $U(v, a)$ vs $a$")
plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max)),
    ax=ax,
    label="Reservation Wage"
)
plt.tight_layout()
plt.savefig("./figures/principal_utility_vs_a.png")
plt.close()

# Plot 3: Cost Minimization - Wage schedule k(v*(y)) vs y
fig, ax = plt.subplots(figsize=(8, 5))
for record in agent_records:
    w = record["reservation_wage"]
    v = record["optimal_contract"]
    wage = mhp.k(v)
    color = cmap((w - w_min) / (w_max - w_min))
    ax.plot(y_grid, wage, color=color, alpha=0.6, linewidth=0.5)
ax.set_xlabel("Output ($y$)")
ax.set_ylabel("Wage")
ax.set_title("Cost Minimization: Wage Schedule $k(v^*(y))$ vs $y$")
plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max)),
    ax=ax,
    label="Reservation Wage"
)
plt.tight_layout()
plt.savefig("./figures/cost_minimization_wage_schedule_vs_y.png")
plt.close()

# Plot 4: Cost Minimization - Utility function U(v, a) vs a
fig, ax = plt.subplots(figsize=(8, 5))
for record in agent_records:
    w = record["reservation_wage"]
    v = record["optimal_contract"]
    U_grid = mhp.U(v, a_grid)
    color = cmap((w - w_min) / (w_max - w_min))
    ax.plot(a_grid, U_grid, color=color, alpha=0.6, linewidth=0.5)
ax.set_xlabel("Action ($a$)")
ax.set_ylabel("Utility")
ax.set_title("Cost Minimization: Utility $U(v, a)$ vs $a$")
plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max)),
    ax=ax,
    label="Reservation Wage"
)
plt.tight_layout()
plt.savefig("./figures/cost_minimization_utility_vs_a.png")
plt.close()
