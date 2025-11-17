#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization for DNN models.
Uses intelligent sampling instead of exhaustive grid search.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from omegaconf import OmegaConf
import torch
from pathlib import Path

# Personal imports
import ML.datamodule.DNN_Datamodule as DNN_Datamodule
import ML.models.DNN_Model as DNN_Model
import ML.models.modes as modes

from sweeper import ConfigSweeper


class OptunaObjective:
    """
    Objective function for Optuna optimization.
    Wraps the training process and returns validation metrics.
    """
    
    def __init__(self, base_config_path: str):
        """
        Args:
            base_config_path: Path to the base YAML configuration
        """
        self.base_config_path = base_config_path
        self.sweeper = ConfigSweeper(base_config_path)
        
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function that Optuna will optimize.
        
        Args:
            trial: Optuna trial object for suggesting hyperparameters
            
        Returns:
            Validation R² score to maximize (ranges from -inf to 1, where 1 is perfect)
        """
        # Reset to base config for each trial
        self.sweeper.reset()
        cfg = self.sweeper.get_cfg()
        
        # ============================================================
        # Define hyperparameter search space using Optuna
        # ============================================================
        
        # Input scaling for U235
        u235_input_scaling = trial.suggest_categorical(
            "u235_input_scaling", 
            ["MinMax", "robust", "standard", "quantile", "log"]
        )
        cfg.dataset.inputs.U235 = u235_input_scaling
        
        # Target scaling for U235
        u235_target_scaling = trial.suggest_categorical(
            "u235_target_scaling", 
            ["MinMax", "robust", "standard", "quantile", "log"]
        )
        cfg.dataset.targets.U235 = u235_target_scaling
        
        # Model architecture
        layer_config = trial.suggest_categorical(
            "layers",
            ["64_64", "128_64", "64_32_64", "128_128", "128_64_32"]
        )
        layer_map = {
            "64_64": [64, 64],
            "128_64": [128, 64],
            "64_32_64": [64, 32, 64],
            "128_128": [128, 128],
            "128_64_32": [128, 64, 32],
        }
        cfg.model.layers = layer_map[layer_config]

        # Dropout
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        cfg.model.dropout_probability = dropout

        # Activation function
        activation = trial.suggest_categorical(
            "activation", 
            ["relu", "gelu", "tanh", "elu"]
        )
        cfg.model.activation = activation

        # Only suggest residual if architecture supports it
        supports_residual = layer_config in ["64_64", "128_128"]
        if supports_residual:
            residual = trial.suggest_categorical("residual_connections", [True, False])
        else:
            residual = False  # No point enabling it
        cfg.model.residual_connections = residual

        # Loss function
        loss = trial.suggest_categorical(
            "loss", 
            ["mse", "mae", "huber", "smooth_l1"]
        )
        cfg.train.loss = loss
        
        # Learning rate (log scale)
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        cfg.train.learning_rate = lr
        
        # Weight decay (log scale)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        cfg.train.weight_decay = weight_decay
        
        # Batch size
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
        cfg.dataset.train.batch_size = batch_size
        
        # ============================================================
        # Configure training
        # ============================================================
        
        # Force training mode
        cfg.runtime.mode = "train"
        
        # Update model name to include trial number
        cfg.model.name = f"optuna_trial_{trial.number}"
        
        # Optionally reduce epochs for faster trials
        # cfg.train.num_epochs = 100  # Uncomment to limit epochs during search
        
        # ============================================================
        # Train model and get validation metric
        # ============================================================
        
        try:
            # Create datamodule
            datamodule = DNN_Datamodule.DNN_Datamodule(cfg)
            
            # Train and get results
            results = modes.train_and_test(datamodule, DNN_Model.DNN_Model, cfg)
            
            # Extract R² metric
            if isinstance(results, dict):
                val_r2 = results.get("val_r2", -float("inf"))
            elif hasattr(results, "val_r2"):
                val_r2 = results.val_r2
            else:
                print(f"Warning: Could not extract R² metric for trial {trial.number}")
                val_r2 = -float("inf")
            
            # Transform R² to amplify differences near 1
            # 0.99992 → ~4.1, 0.9999 → ~4.0, 0.999 → ~3.0
            if val_r2 > 0:
                transformed = -math.log10(1 - val_r2 + 1e-12)
            else:
                transformed = val_r2  # Keep negative R² as-is
            
            print(f"Trial {trial.number}: R² = {val_r2:.6f}, transformed = {transformed:.4f}")
            return transformed
            
        except Exception as e:
            import traceback
            print(f"Trial {trial.number} failed with error: {e}")
            traceback.print_exc()
            return -float("inf")


def run_optuna_study(
    base_config: str = "base_simple_U235.yaml",
    study_name: str = "U235_hyperparameter_optimization",
    storage: str = "sqlite:///optuna_U235.db",
    n_trials: int = 100,
    n_jobs: int = 1,
    timeout: int = None,
    load_if_exists: bool = True,
):
    """
    Run Optuna hyperparameter optimization study.
    
    Args:
        base_config: Path to base YAML configuration
        study_name: Name for the Optuna study
        storage: Database URL for storing study results
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (set to 1 to avoid GPU conflicts)
        timeout: Time limit in seconds (None for no limit)
        load_if_exists: Whether to continue an existing study
    """
    
    # Set PyTorch matmul precision
    torch.set_float32_matmul_precision("high")
    
    # Create study with TPE sampler and median pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(
        n_startup_trials=5,  # Don't prune first 5 trials
        n_warmup_steps=10,   # Don't prune for first 10 epochs
    )
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",  # Maximize R² score (ranges from -inf to 1)
        load_if_exists=load_if_exists,
    )
    
    # Create objective
    objective = OptunaObjective(base_config)
    
    print(f"Starting Optuna study: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Storage: {storage}")
    print("=" * 80)
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout,
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("Optimization finished!")
    print("=" * 80)
    
    # Convert back from transformed value to actual R²
    best_transformed = study.best_trial.value
    if best_transformed > 0:
        best_r2 = 1 - 10**(-best_transformed)
    else:
        best_r2 = best_transformed
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best R²: {best_r2:.6f}")
    print(f"Best transformed value: {best_transformed:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters to YAML
    best_config_path = Path(f"best_config_trial_{study.best_trial.number}.yaml")
    sweeper = ConfigSweeper(base_config)
    sweeper.reset()
    cfg = sweeper.get_cfg()
    
    # Apply best parameters
    cfg.dataset.inputs.U235 = study.best_trial.params["u235_input_scaling"]
    cfg.dataset.targets.U235 = study.best_trial.params["u235_target_scaling"]

    layer_map = {
        "64_64": [64, 64],
        "128_64": [128, 64],
        "64_32_64": [64, 32, 64],
        "128_128": [128, 128],
        "128_64_32": [128, 64, 32],
    }
    cfg.model.layers = layer_map[study.best_trial.params["layers"]]
    cfg.model.dropout_probability = study.best_trial.params["dropout"]
    cfg.model.activation = study.best_trial.params["activation"]

    # Handle conditional residual_connections
    cfg.model.residual_connections = study.best_trial.params.get("residual_connections", False)

    cfg.train.loss = study.best_trial.params["loss"]
    cfg.train.learning_rate = study.best_trial.params["learning_rate"]
    cfg.train.weight_decay = study.best_trial.params["weight_decay"]
    cfg.dataset.train.batch_size = study.best_trial.params["batch_size"]
    cfg.model.name = f"best_trial_{study.best_trial.number}"
    
    OmegaConf.save(cfg, best_config_path)
    print(f"\nBest configuration saved to: {best_config_path}")
    
    # Print trial statistics
    print(f"\nTotal trials: {len(study.trials)}")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Completed trials: {len(completed_trials)}")
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Pruned trials: {len(pruned_trials)}")
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"Failed trials: {len(failed_trials)}")
    
    return study


if __name__ == "__main__":
    # Run the optimization
    study = run_optuna_study(
        base_config="base_simple_U235.yaml",
        study_name="U235_DNN_optimization",
        storage="sqlite:///optuna_U235_study.db",
        n_trials=300,  # Adjust based on computational budget
        n_jobs=1,      # Keep at 1 to avoid GPU conflicts
        timeout=None,  # No time limit
    )
    
    # Optional: Generate visualizations
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html("optuna_optimization_history.html")
        
        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html("optuna_param_importances.html")
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html("optuna_parallel_coordinate.html")
        
        print("\nVisualization plots saved as HTML files.")
        
    except ImportError:
        print("\nInstall plotly for visualization: pip install plotly")