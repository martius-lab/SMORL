from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.scalor_training import scalor_training

experiment = scalor_training
variant = dict(
        algorithm='SCALOR',
        imsize=64,

        generate_scalor_dataset_kwargs=dict(
            init_camera=sawyer_init_camera_zoomed_in,
            env_id='SawyerMultiobjectRearrangeEnv-TwoObj-v0',
            N=101,
            rollout_length=100,
            test_p=.9,
            use_cached=True,
            show=False,
        ),
        scalor_params=dict(
            n_itr=6001,
            lr=0.0001, 
            batch_size=11,
            num_cell_h=4, 
            num_cell_w=4, 
            max_num_obj=5,
            explained_ratio_threshold=0.1, 
            sigma=0.1, 
            ratio_anc=1.0, 
            var_anc=0.3, 
            size_anc=0.22, 
            var_s=0.12,  
            z_pres_anneal_end_value=0.0001, 
        ),
        save_period=25,
        )

from rlkit.torch.scalor import common
common.z_what_dim = 4

if __name__ == "__main__":
    mode = 'local'
    exp_prefix = 'scalor_training_2_objects'
    run_experiment(
                scalor_training,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                )
