.PHONY: setup results clean

# Default target
all: setup results

# Setup environment
setup:
	@echo "Setting up environment..."
	chmod +x setup.sh
	./setup.sh

# Run all result generation scripts
results: results_cost_min results_opt_action

# Generate cost minimization results
results_cost_min:
	@echo "Generating cost minimization results..."
	@source .venv/bin/activate && python -c "import matplotlib; matplotlib.use('Agg'); from py import produce_results_cost_min" || { echo "Error generating cost minimization results"; exit 1; }

# Generate optimal action results
results_opt_action:
	@echo "Generating optimal action results..."
	@source .venv/bin/activate && python -c "import matplotlib; matplotlib.use('Agg'); from py import produce_results_opt_action" || { echo "Error generating optimal action results"; exit 1; }

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -f *.pyc
	rm -rf __pycache__
	rm -rf output/* 