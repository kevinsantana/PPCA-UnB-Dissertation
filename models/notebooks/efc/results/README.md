# Reproducible Results

Each `experiment_x` folder represents one of the experiments executed on research, containing a `experiment.py` file with a `main` Python function. Which can be executed to reproduce a given experiment.

## Running the Experiments

These scripts should be executed from the `efc` directory. 

PPCA-UnB-Dissertation/
├── models/
│   └── notebooks/
│       └── efc/
│           ├── results/
│           │   ├── __init__.py
│           │   ├── common/
│           │   │   ├── __init__.py
│           │   │   └── constants.py
│           │   ├── experiment_3/
│           │   │   ├── __init__.py
│           │   │   └── unbalanced_techniques.py
│           │   ├── experiment_4/
│           │   │   ├── __init__.py
│           │   │   └── feature_selection_selectkbest_f_classif.py
│           │   ├── experiment_5/
│           │   │   ├── __init__.py
│           │   │   ├── experiment.py
│           │   │   └── feature_selection_with_balanced_datasets.py
│           │   └── ... other experiments
│           └── ... other files

1. Run the script using the module notation `(-m)`

```bash
python -m results.experiment_5.experiment
```
