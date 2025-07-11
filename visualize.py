import argparse 
import os 
import glob
import pickle 
import tarfile
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from natsort import natsorted, ns

from deap import base, creator

# Recreate the same DEAP types as in simulation
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Complexity", float)
creator.create("Expression", list)
creator.create("Individual", list,
               fitness=creator.Fitness,
               complexity=creator.Complexity,
               expression=creator.Expression)

def load_checkpoint(path): 
    # Loads either gzipped or pkled files (older versions of evolution.py output pkled)
    if path.endswith('.pkl'):
        
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".pkl"):
                    f = tar.extractfile(member)
                    return pickle.load(f)
    raise ValueError("Unsupported file format")

def extract_data(directory):
    fitnesses = []
    complexities = []
    for rep_path in natsorted(glob.glob(os.path.join(directory, "*_gen*")), alg=ns.IGNORECASE):
        checkpoint = load_checkpoint(rep_path)
        print(rep_path)
        
        log = checkpoint['logbook']
        
        # Clean any duplicate generations 
        genlist = log.select('gen') 
        seen = set()
        dupes = [x for x in genlist if x in seen or seen.add(x)]
        for x in dupes:
            log.pop(x)

        fitnesses.append(log.select('meanFit'))
        complexities.append(log.select('meanComp'))
    return fitnesses, complexities

def plot_all(fits_list, comps_list, labels, ax_fitness, ax_complexity):
    color_cycle = ['orange', 'blue', 'green', 'red']
    custom_lines = []

    for dir_idx, (fit_runs, comp_runs, label) in enumerate(zip(fits_list, comps_list, labels)):
        color = color_cycle[dir_idx % len(color_cycle)]
        custom_lines.append(Line2D([0], [0], color=color, linewidth=2))

        for fit, comp in zip(fit_runs, comp_runs):
            ax_fitness.plot(fit[:1000000], color=color, alpha=0.1)
            ax_complexity.plot(comp[:1000000], color=color, alpha=0.1)

    ax_fitness.set_title("Fitness")
    ax_fitness.set_ylabel("Fitness (% of maximum)")
    ax_fitness.set_xlabel("Generation")

    ax_complexity.set_title("Complexity")
    ax_complexity.set_ylabel("Complexity (1 - Gini)")
    ax_complexity.set_xlabel("Generation")
    ax_complexity.legend(custom_lines, labels)

def main():
    parser = argparse.ArgumentParser
    parser = argparse.ArgumentParser(description="Compare evolution fitness and complexity curves.")
    parser.add_argument("dirs", nargs='+', help="One or two checkpoint directories")
    parser.add_argument("--labels", nargs='*', help="Optional labels for the datasets")
    args = parser.parse_args()

    labels = args.labels if args.labels else args.dirs

    fitness_all = []
    complexity_all = []

    for d in args.dirs:
        fits, comps = extract_data(d)
        fitness_all.append(fits)
        complexity_all.append(comps)

    fig, (fitness_ax, complexity_ax) = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    fitness_ax.set_xscale('log')
    # fitness_ax.set_xlim(left=100)
    fitness_ax.set_ylim(-2.23, 102.23)
    complexity_ax.set_ylim(-0.022, 1.022)
    
    plot_all(fitness_all, complexity_all, labels, fitness_ax, complexity_ax)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()