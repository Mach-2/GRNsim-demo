# GRNsim-demo

**GRNsim** uses genetic algorithms developed with the DEAP Python framework to simulate the evolution of gene regulatory networks (GRNs). Individuals within the population consist of biochemically-informed GRNs that model transcription factor binding and target gene expression. The simulation tracks fitness, regulatory complexity, and optionally population diversity to explore the role of recombination in the evolution of regulatory complexity. 

This repository is a **demonstration subset** of the simulation framework developed as part of my graduate studies. It includes the essential functions and core logic, but does not contain the full range of simulation parameters or final experimental results. A manuscript with full results and analysis is currently in preparation.

## Simulation framework 
### GRN Model
In this simulation, there are two gene types: target genes, the expression of which determines fitness, and TFs, which regulate the target genes and each other, but do not directly impact fitness. The 200 target genes and 20 TFs are randomly arranged on a single linear “chromosome”, and the binding affinities between them are captured in a 20 x 220 matrix. A gene’s expression is the sum of TF binding probabilities (determined by the combined effect of their affinity and expression), each weighted by their effect on expression (+1 for activators, -1 for repressors; n=10 of each). Each target gene has a randomly initialized ‘optimal’ expression; its fitness is normally distributed with respect to the log(expression), centred at this optimum. An individual’s fitness is the product of the fitness measures for all target genes, simulating a system where all target genes are equal and essential. 

### Evolution simulation
Populations are evolved using a genetic algorithm. In each generation, individuals are selected with probability proportional to their fitness and allowed to reproduce. Without recombination, offspring are a clonal replicate of a parent genome. With recombination, offspring are a hybrid of both parents, with a single randomly selected recombination site determining which genotypes are passed on. The TF affinities for each gene are inherited as a unit from a single parent, simulating inheritance of a single cis-regulatory region for each gene. For both with and without recombination, the TF affinities of offspring are subject to mutation. A new generation then begins, and the process repeats.

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

## Notes 
* Checkpoint files are thinned to reduce disk usage. Only the most-recent checkpoint contains a DEAP logbook, but all contain populations from those save points 
* If a simulation is interrupted before completion, running `python evolution.py config-name` again will resume it from the most-recent checkpoint 

## Acknowledgements
Built using [DEAP](https://github.com/DEAP/deap) (Distributed Evolutionary Algorithms in Python)