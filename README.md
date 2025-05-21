# Replication code for Azevedo and Wolff, 2025

## Replication instructions
- Run `setup.sh` to install python.
- Run `produce_results_cost_min.py` and `produce_results_opt_action.py`.

## Requirements
- Python 3.13. Tested on OSX 15.3.2.

## Core classes

The following files define the main classes used in the project:

- **prob_class.py**: Defines the basic principal/agent problem with:
  - Error function
  - Cost function
  - Utility function
  - Reservation utility

- **err_class.py**: Defines error functions. Instances of `prob_class` are initialized with a member of `err_class`

- **contract.py**: Defines contracts. Solutions to cost minimization problems return a contract, including:
  - Value function
  - Wage function method

- **plotter.py**: Provides plotting methods for `prob_class` members to quickly generate visualizations. Every `prob` instance is automatically instantiated with a plotter

- **cost_min.py**: Implements the cost minimization problem. Inherits from `prob` and adds:
  - Intended action
  - Methods for solving the cost minimization problem

## Replication

The following files produce results for the paper:

- **produce_results_opt_action.py**: Generates the graphs with optimal actions shown in the introduction

- **produce_results_cost_min.py**: Produces the cost minimization graphs presented in section 5
