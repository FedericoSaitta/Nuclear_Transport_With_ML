"""
Optuna-based hyperparameter optimization for DNN models.
Uses intelligent sampling instead of exhaustive grid search.
GENERALIZED VERSION: Automatically optimizes scaling for all isotope features.
"""
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from omegaconf import OmegaConf
import torch
from pathlib import Path
import math
import re

# Personal imports
import ML.datamodule.DNN_Datamodule as DNN_Datamodule
import ML.models.DNN_Model as DNN_Model
import ML.models.modes as modes

from ML.parameter_tuners.sweeper import ConfigSweeper

class OptunaObjective:
    """
    Objective function for Optuna optimization.
    Wraps the training process and returns validation metrics.
    """
    
    def __init__(self, base_config_path, fixed_operational_features = None, optimize_isotope_scaling = True):
        """
        Args:
            base_config_path: Path to the base YAML configuration
            fixed_operational_features: List of feature names with fixed scaling (e.g., ["power_W_g", "fuel_temp_K"])
            optimize_isotope_scaling: Whether to optimize scaling for isotope features
        """
        self.base_config_path = base_config_path
        self.sweeper = ConfigSweeper(base_config_path)
        self.optimize_isotope_scaling = optimize_isotope_scaling
        
        # Define operational features that should NOT be optimized
        # (these have well-established physical scalings)
        if fixed_operational_features is None:
            self.fixed_operational_features = [
                "power_W_g",
                "fuel_temp_K", 
                "mod_temp_K",
                "mod_density_g_cm3",
                "boron_ppm",
                "clad_temp_K",
            ]
        else:
            self.fixed_operational_features = fixed_operational_features
        
        # Identify optimizable features from base config
        self._identify_optimizable_features()
        
    def _identify_optimizable_features(self):
        """Identify which input/target features should have their scaling optimized."""
        cfg = self.sweeper.get_cfg()
        
        # Features that can be optimized (not in fixed list)
        self.optimizable_inputs = []
        self.optimizable_targets = []
        
        if self.optimize_isotope_scaling:
            # Check inputs
            if hasattr(cfg.dataset, 'inputs'):
                for feature_name in cfg.dataset.inputs.keys():
                    if feature_name not in self.fixed_operational_features:
                        self.optimizable_inputs.append(feature_name)
            
            # Check targets
            if hasattr(cfg.dataset, 'targets'):
                for feature_name in cfg.dataset.targets.keys():
                    if feature_name not in self.fixed_operational_features:
                        self.optimizable_targets.append(feature_name)
        
        print(f"📊 Optimizable input features: {self.optimizable_inputs}")
        print(f"🎯 Optimizable target features: {self.optimizable_targets}")
        
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function that Optuna will optimize.
        
        Args:
            trial: Optuna trial object for suggesting hyperparameters
            
        Returns:
            Validation R² score to maximize (transformed for better optimization)
        """
        # Reset to base config for each trial
        self.sweeper.reset()
        cfg = self.sweeper.get_cfg()
        
        # ============================================================
        # DYNAMIC SCALING OPTIMIZATION
        # Suggest scalers for all optimizable features
        # ============================================================
        
        scaler_choices = ["MinMax", "robust", "standard", "quantile", "power"] 
        
        # Optimize input scaling for isotopes/non-fixed features
        for feature in self.optimizable_inputs:
            suggested_scaler = trial.suggest_categorical(
                f"input_scaling_{feature}",
                scaler_choices
            )
            cfg.dataset.inputs[feature] = suggested_scaler
        
        # Optimize target scaling for isotopes/non-fixed features
        for feature in self.optimizable_targets:
            suggested_scaler = trial.suggest_categorical(
                f"target_scaling_{feature}",
                scaler_choices
            )
            cfg.dataset.targets[feature] = suggested_scaler
        
        # ============================================================
        # MODEL ARCHITECTURE OPTIMIZATION
        # ============================================================
        
        # Layer configuration
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

        # Residual connections (only if architecture supports it)
        supports_residual = layer_config in ["64_64", "128_128"]
        if supports_residual:
            residual = trial.suggest_categorical("residual_connections", [True, False])
        else:
            residual = False
        cfg.model.residual_connections = residual

        # ============================================================
        # TRAINING OPTIMIZATION
        # ============================================================
        
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
        # CONFIGURE TRAINING RUN
        # ============================================================
        
        # Force training mode
        cfg.runtime.mode = "train"
        
        # Update model name to include trial number
        cfg.model.name = f"optuna_trial_{trial.number}"
        
        # Optionally reduce epochs for faster trials
        # cfg.train.num_epochs = 100  # Uncomment to limit epochs during search
        
        # ============================================================
        # TRAIN MODEL AND GET VALIDATION METRIC
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
                print(f"⚠️  Warning: Could not extract R² metric for trial {trial.number}")
                val_r2 = -float("inf")
            
            # Transform R² to amplify differences near 1
            # This helps Optuna distinguish between very good models
            # 0.99992 → ~4.1, 0.9999 → ~4.0, 0.999 → ~3.0
            if val_r2 > 0:
                transformed = -math.log10(1 - val_r2 + 1e-12)
            else:
                transformed = val_r2  # Keep negative R² as-is
            
            print(f"✅ Trial {trial.number}: R² = {val_r2:.6f}, transformed = {transformed:.4f}")
            return transformed
            
        except Exception as e:
            import traceback
            print(f"❌ Trial {trial.number} failed with error: {e}")
            traceback.print_exc()
            return -float("inf")


def run_optuna_study(base_config, study_name, storage, n_trials, n_jobs, timeout, load_if_exists, 
                    fixed_operational_features,optimize_isotope_scaling,):
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
        fixed_operational_features: List of features with fixed scaling (None = use defaults)
        optimize_isotope_scaling: Whether to optimize isotope scaling
    """
    
    # Set PyTorch matmul precision
    torch.set_float32_matmul_precision("high")
    
    # Create study with TPE sampler and median pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5,  n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name, storage=storage, sampler=sampler, pruner=pruner, direction="maximize", load_if_exists=load_if_exists,
    )
    
    # Create objective
    objective = OptunaObjective(
        base_config,
        fixed_operational_features=fixed_operational_features,
        optimize_isotope_scaling=optimize_isotope_scaling,
    )
    
    print("=" * 80)
    print(f"🚀 Starting Optuna study: {study_name}")
    print(f"📊 Number of trials: {n_trials}")
    print(f"💾 Storage: {storage}")
    print(f"🔬 Base config: {base_config}")
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
    print("🎉 Optimization finished!")
    print("=" * 80)
    
    # Convert back from transformed value to actual R²
    best_transformed = study.best_trial.value
    if best_transformed > 0:
        best_r2 = 1 - 10**(-best_transformed)
    else:
        best_r2 = best_transformed
    
    print(f"\n🏆 Best trial: {study.best_trial.number}")
    print(f"📈 Best R²: {best_r2:.6f}")
    print(f"📊 Best transformed value: {best_transformed:.4f}")
    print("\n⚙️  Best hyperparameters:")
    
    # Group parameters by category for better readability
    scaling_params = {}
    model_params = {}
    training_params = {}
    
    for key, value in study.best_trial.params.items():
        if "scaling" in key:
            scaling_params[key] = value
        elif key in ["layers", "dropout", "activation", "residual_connections"]:
            model_params[key] = value
        else:
            training_params[key] = value
    
    if scaling_params:
        print("\n  🔬 Feature Scaling:")
        for key, value in scaling_params.items():
            print(f"    {key}: {value}")
    
    if model_params:
        print("\n  🏗️  Model Architecture:")
        for key, value in model_params.items():
            print(f"    {key}: {value}")
    
    if training_params:
        print("\n  🎓 Training Config:")
        for key, value in training_params.items():
            print(f"    {key}: {value}")
    
    # Save best parameters to YAML
    best_config_path = Path(f"best_config_trial_{study.best_trial.number}.yaml")
    sweeper = ConfigSweeper(base_config)
    sweeper.reset()
    cfg = sweeper.get_cfg()
    
    # Apply best scaling parameters
    for key, value in scaling_params.items():
        if key.startswith("input_scaling_"):
            feature = key.replace("input_scaling_", "")
            cfg.dataset.inputs[feature] = value
        elif key.startswith("target_scaling_"):
            feature = key.replace("target_scaling_", "")
            cfg.dataset.targets[feature] = value
    
    # Apply best model parameters
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
    cfg.model.residual_connections = study.best_trial.params.get("residual_connections", False)
    
    # Apply best training parameters
    cfg.train.loss = study.best_trial.params["loss"]
    cfg.train.learning_rate = study.best_trial.params["learning_rate"]
    cfg.train.weight_decay = study.best_trial.params["weight_decay"]
    cfg.dataset.train.batch_size = study.best_trial.params["batch_size"]
    cfg.model.name = f"best_trial_{study.best_trial.number}"
    
    OmegaConf.save(cfg, best_config_path)
    print(f"\n💾 Best configuration saved to: {best_config_path}")
    
    # Print trial statistics
    print(f"\n📊 Trial Statistics:")
    print(f"  Total trials: {len(study.trials)}")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  ✅ Completed trials: {len(completed_trials)}")
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  ✂️  Pruned trials: {len(pruned_trials)}")
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"  ❌ Failed trials: {len(failed_trials)}")
    
    return study


if __name__ == "__main__":
  # Run the optimization with default settings
  # Automatically detects and optimizes all isotope features
  study = run_optuna_study(
    base_config="parameter_tuners/base_simple_chain.yaml",
    study_name="isotope_DNN_optimization",
    storage="sqlite:///parameter_tuners/optuna_isotope_study.db",
    n_trials=1_000,
    n_jobs=1,
    timeout=None,
    load_if_exists=True,  
    fixed_operational_features=None,
    optimize_isotope_scaling=True,
  )
