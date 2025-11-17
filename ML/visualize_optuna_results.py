#!/usr/bin/env python3
"""
Analyze and visualize Optuna study results.
Run this after your optimization to get insights into the hyperparameter search.
"""

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice,
    plot_edf,
)
import pandas as pd
from pathlib import Path


def load_study(study_name: str = "U235_DNN_optimization", 
               storage: str = "sqlite:///optuna_U235_study.db"):
    """Load an existing Optuna study."""
    return optuna.load_study(study_name=study_name, storage=storage)


def print_study_summary(study: optuna.Study):
    """Print a comprehensive summary of the study."""
    print("=" * 80)
    print("OPTUNA STUDY SUMMARY")
    print("=" * 80)
    
    print(f"\nStudy name: {study.study_name}")
    print(f"Direction: {study.direction}")
    print(f"Total trials: {len(study.trials)}")
    
    # Count trials by state
    states = {}
    for trial in study.trials:
        state = trial.state.name
        states[state] = states.get(state, 0) + 1
    
    print("\nTrials by state:")
    for state, count in states.items():
        print(f"  {state}: {count}")
    
    # Best trial info
    if study.best_trial:
        print(f"\n{'=' * 40}")
        print("BEST TRIAL")
        print('=' * 40)
        print(f"Trial number: {study.best_trial.number}")
        print(f"Value (val_loss): {study.best_trial.value:.6f}")
        print(f"\nHyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
    
    # Top 5 trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) >= 5:
        print(f"\n{'=' * 40}")
        print("TOP 5 TRIALS")
        print('=' * 40)
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:5]
        for i, trial in enumerate(sorted_trials, 1):
            print(f"\n{i}. Trial {trial.number}: {trial.value:.6f}")
            print(f"   Params: {trial.params}")


def create_visualizations(study: optuna.Study, output_dir: str = "optuna_plots"):
    """Create and save all visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations in {output_dir}/...")
    
    try:
        # 1. Optimization History
        fig = plot_optimization_history(study)
        fig.write_html(output_path / "optimization_history.html")
        print("  ✓ Optimization history")
        
        # 2. Parameter Importances
        fig = plot_param_importances(study)
        fig.write_html(output_path / "param_importances.html")
        print("  ✓ Parameter importances")
        
        # 3. Parallel Coordinate Plot
        fig = plot_parallel_coordinate(study)
        fig.write_html(output_path / "parallel_coordinate.html")
        print("  ✓ Parallel coordinate plot")
        
        # 4. Slice Plot (shows individual parameter effects)
        fig = plot_slice(study)
        fig.write_html(output_path / "slice_plot.html")
        print("  ✓ Slice plot")
        
        # 5. Contour Plot (shows parameter interactions)
        fig = plot_contour(study)
        fig.write_html(output_path / "contour_plot.html")
        print("  ✓ Contour plot")
        
        # 6. EDF (Empirical Distribution Function)
        fig = plot_edf(study)
        fig.write_html(output_path / "edf_plot.html")
        print("  ✓ EDF plot")
        
        print(f"\nAll plots saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Make sure plotly is installed: pip install plotly")


def export_trials_to_csv(study: optuna.Study, output_file: str = "optuna_trials.csv"):
    """Export all trial results to a CSV file."""
    # Extract trial data
    trials_data = []
    for trial in study.trials:
        trial_dict = {
            "trial_number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        }
        # Add all parameters
        trial_dict.update(trial.params)
        trials_data.append(trial_dict)
    
    # Create DataFrame and save
    df = pd.DataFrame(trials_data)
    df.to_csv(output_file, index=False)
    print(f"\nTrial data exported to {output_file}")
    print(f"Total rows: {len(df)}")


def compare_parameter_ranges(study: optuna.Study):
    """Analyze which parameter ranges work best."""
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("No completed trials to analyze.")
        return
    
    # Sort trials by value
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)
    
    # Get top 20% and bottom 20%
    n_top = max(1, len(sorted_trials) // 5)
    top_trials = sorted_trials[:n_top]
    bottom_trials = sorted_trials[-n_top:]
    
    print(f"\n{'=' * 80}")
    print("PARAMETER ANALYSIS: TOP 20% vs BOTTOM 20%")
    print('=' * 80)
    
    # Get all parameter names
    param_names = list(top_trials[0].params.keys())
    
    for param_name in param_names:
        print(f"\n{param_name}:")
        
        # Collect values
        top_values = [t.params[param_name] for t in top_trials]
        bottom_values = [t.params[param_name] for t in bottom_trials]
        
        # Categorical parameters
        if isinstance(top_values[0], (str, bool)):
            from collections import Counter
            top_counts = Counter(top_values)
            bottom_counts = Counter(bottom_values)
            
            print("  Top 20% distribution:")
            for val, count in top_counts.most_common():
                pct = 100 * count / len(top_values)
                print(f"    {val}: {count} ({pct:.1f}%)")
            
            print("  Bottom 20% distribution:")
            for val, count in bottom_counts.most_common():
                pct = 100 * count / len(bottom_values)
                print(f"    {val}: {count} ({pct:.1f}%)")
        
        # Numerical parameters
        else:
            import numpy as np
            top_mean = np.mean(top_values)
            top_std = np.std(top_values)
            bottom_mean = np.mean(bottom_values)
            bottom_std = np.std(bottom_values)
            
            print(f"  Top 20%: {top_mean:.6f} ± {top_std:.6f}")
            print(f"  Bottom 20%: {bottom_mean:.6f} ± {bottom_std:.6f}")


def analyze_study(study_name: str = "U235_DNN_optimization",
                  storage: str = "sqlite:///optuna_U235_study.db",
                  create_plots: bool = True,
                  export_csv: bool = True):
    """Complete analysis of an Optuna study."""
    
    print("Loading study...")
    study = load_study(study_name, storage)
    
    # Print summary
    print_study_summary(study)
    
    # Compare parameter ranges
    compare_parameter_ranges(study)
    
    # Export to CSV
    if export_csv:
        export_trials_to_csv(study)
    
    # Create visualizations
    if create_plots:
        create_visualizations(study)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Run complete analysis
    analyze_study(
        study_name="U235_DNN_optimization",
        storage="sqlite:///optuna_U235_study.db",
        create_plots=True,
        export_csv=True,
    )