import importlib
import os
import pkgutil
import sys

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


not_exp_modules = ["smoke_test", "path_length_settings"]
modules = list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)])
               if module not in not_exp_modules)
all_experiments = []
for m_str in modules:
    m = importlib.import_module(m_str)
    all_experiments.append((m_str, m.experiment, m.variant))

smoke_algo_kwargs = dict(
        batch_size=100,
        min_num_steps_before_training=200,
        num_epochs=1,
        num_train_loops_per_epoch=1,
        num_expl_steps_per_train_loop=200,
        num_trains_per_train_loop=1,
        num_eval_steps_per_epoch=200
    )

if __name__ == '__main__':
    for name, experiment, variant in all_experiments:
        print(f"Started: {name}")
        with HiddenPrints():
            variant["algo_kwargs"] = smoke_algo_kwargs
            
            if "z_what_dim" in variant:
                from rlkit.torch.scalor import (common, discovery, model,
                                                modules, propagation, scalor)
                for module in (common, discovery, model,
                               modules, propagation, scalor):
                    module.z_what_dim = variant["z_what_dim"]

            run_experiment(
                experiment,
                exp_prefix='smoke_test_runs',
                mode='local',
                variant=variant,
                use_gpu=True)

        print(f"Finished: {name}")
