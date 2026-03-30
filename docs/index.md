# RKO - Random-Key Optimizer (Python Framework)

Welcome to the documentation for the **RKO (Random-Key Optimizer)**.

The Random-Key Optimizer is a versatile and efficient metaheuristic framework designed for a wide range of combinatorial optimization problems. Its core paradigm is the encoding of solutions as vectors of random keys—real numbers uniformly distributed in the interval [0, 1). This representation maps the discrete, and often complex, search space of a combinatorial problem to a continuous n-dimensional unit hypercube.

## Installation

You can download the package using pip:
```bash
pip install rko
```

## Getting Started

To learn how to use RKO, please refer to the Github repository's [README](https://github.com/RKO-solver/RKO_Python) which details how to create your own `Environment` and run the `RKO` solver.

## API Reference

Explore the detailed API Reference using the navigation menu:

- **[RKO](reference/rko.md)**: The core solver class, encapsulating all search operators and metaheuristics.
- **[Environment](reference/environment.md)**: The abstract base class (`RKOEnvAbstract`) enabling the integration of problem-specific logic.
- **[LogStrategy](reference/logstrategy.md)**: Mechanisms for logging search progress.
- **[Plots](reference/plots.md)**: Visualization utilities for convergence and other metrics.

---

### Maintainers
- Felipe Silvestre Cardoso Roberto - [Linkedin](https://www.linkedin.com/in/felipesilvestrecr/)
- João Victor Assaoka Ribeiro - [Linkedin](https://www.linkedin.com/in/assaoka/)
