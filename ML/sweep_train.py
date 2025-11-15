# sweep_train.py

from omegaconf import OmegaConf
import torch

# Personal imports (same as your main)
import ML.datamodule.DNN_Datamodule as DNN_Datamodule
import ML.models.DNN_Model as DNN_Model
import ML.models.modes as modes

from sweeper import ConfigSweeper


if __name__ == "__main__":
    # Allow use of tensor cores if present
    torch.set_float32_matmul_precision("high")

    # This object holds the base config and the helpers:
    model = ConfigSweeper("test_config.yaml")  # you can call it 'model' if you like

    # ----- define your search space -----
    layer_grid = [
        [64, 64],
        [128, 64],
        [128, 128, 64],
    ]

    input_scaling_grid = ["MinMax", "robust"]
    train_batch_grid = [512, 1024]
    val_batch_grid = [4096, 8192]

    # Example: ensure all targets use robust scaling
    # (you can also sweep this if you want)
    # model.set_scaling("targets", "robust")   # per-run inside the loop if you want

    run_id = 0

    for layers in layer_grid:
        for input_scaling in input_scaling_grid:
            for train_bs in train_batch_grid:
                for val_bs in val_batch_grid:
                    run_id += 1

                    # ---- reset to the original config for each run ----
                    model.reset()

                    # ---- apply modifications for this run ----
                    model.set_layers(layers)
                    model.set_scaling("inputs", input_scaling)
                    model.set_scaling("targets", "robust")  # fixed here, but could be a grid too
                    model.set_batch_sizes(train_bs, val_bs)

                    # Example of adding or overriding a single input:
                    # model.add_input("time_days", input_scaling)
                    # model.add_input("U235", "robust")

                    cfg = model.get_cfg()
                    cfg.runtime.mode = "train"  # force train mode

                    # Optional: adjust model name to encode the setup
                    cfg.model.name = (
                        f"layers_{'-'.join(map(str, layers))}"
                        f"_in_{input_scaling}_bs_{train_bs}_{val_bs}_run_{run_id}"
                    )

                    datamodule = DNN_Datamodule.DNN_Datamodule(cfg)
                    modes.train_and_test(datamodule, DNN_Model.DNN_Model, cfg)

                    # You can inspect logs/checkpoints later using cfg.model.name
