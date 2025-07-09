# GRNsim-demo

**GRNsim** uses genetic algorithms developed with the DEAP Python framework to simulate the evolution of gene regulatory networks (GRNs). Individuals within the population consist of biochemically-informed GRNs that model transcription factor binding and target gene expression. The simulation tracks fitness, regulatory complexity, and optionally population diversity to explore the role of recombination in the evolution of regulatory complexity. 

This repository is a **demonstration subset** of the simulation framework developed as part of my graduate studies. It includes the essential functions and core logic, but does not contain the full range of simulation paramters or final experimental results. A manuscript with full results and analysis is currently in preparation.

## Installation 
Clone repository:
```bash 
git clone https://github.com/mach-2/GRNsim-demo.git 
cd GRNsim-demo
```
Install dependencies: 
`pip install -r requirements.txt`

## Directory structure 
```bash
GRNsim-demo/
├── evolution.py             # Main simulation script
├── evoHelpers.py            # Helper functions for network creation, mutation, evaluation, etc.
├── visualize.py             # Visualization script for plotting fitness and complexity
├── requirements.txt         # Python package dependencies
├── README.md                # Project overview and usage
├── replicate-inis/          # Configuration files for simulation replicates
│   ├── s0.ini               # Example replicate config
│   └── example-config.ini   # Template config for new runs
├── checkpoints/             # Checkpoint files for saving simulation progress (can be large)
├── logs/                    # Log files tracking progress during runs
└── plots/                   # Optional: auto-generated plots or results
```

## Running a simulation 

### 1. Create config file 
The `replicate-inis/` directory contains configuration files that specify simulation parameters. The `[network]`, `[evolution]`, `[tracking]`, and `[checkpointing]` sections are required. The `[initializations]` section is optional. If not provided, values will be initialized when the simulation begins. 

A completed `s0.ini` file is provided, or you can modify the blank `example-config.ini` file with your desired parameters. 

### 2. Run simulation
```
python evolution.py config-name
``` 
where `config-name` is the name of the config file (i.e., `s0`, `example-config`)

### 3. Visualize results 
Use the visualization script to view population fitness and complexity over time: 
```bash
python visualize.py checkpoints/ --labels label
```

To compare two runs: 
```bash 
python visualize.py checkpoints/cp_name1 checkpoints/cp_name2 --labels experiment1 experiment2
```
When comparing two runs, use the `cp_name` parameter specified in the config file. 

An example plot is included in `plots/example.png`.

### Notes 
* Checkpoint files are thinned to reduce disk usage. Only the most-recent checkpoint contains a DEAP logbook, but all contain populations from those save points 
* If a simulation is interrupted before completion, running `python evolution.py config-name` again will resume it from the most-recent checkpoint 

## Acknowledgements
Built using [DEAP](https://github.com/DEAP/deap) (Distributed Evolutionary Algorithms in Python)