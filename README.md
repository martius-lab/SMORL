# Self-Supervised Visual Reinforcement Learning with Object-Centric Representations (SMORL)

This repository contains the code release for the paper [Self-Supervised Visual Reinforcement Learning with Object-Centric Representations](https://arxiv.org/abs/2011.14381) by Andrii Zadaianchuk, Maximilian Seitzer, and Georg Martius, published as a *spotlight at ICLR 2021*. Please use the [provided citation](#citation) when making use of our code or ideas.

The code is an adaptation of [RLkit](https://github.com/vitchyr/rlkit) for RL code and [multiworld](https://github.com/vitchyr/multiworld) for environments, both under MIT license.

## Abstract

Autonomous agents need large repertoires of skills to act reasonably on new tasks that they have not seen before. However, acquiring these skills using only a stream of high-dimensional, unstructured, and unlabeled observations is a tricky challenge for any autonomous agent. Previous methods have used variational autoencoders to encode a scene into a low-dimensional vector that can be used as a goal for an agent to discover new skills. Nevertheless, in compositional/multi-object environments it is difficult to disentangle all the factors of variation into such a fixed-length representation of the whole scene. We propose to use object-centric representations as a modular and structured observation space, which is learned with a compositional generative world model. We show that the structure in the representations in combination with goal-conditioned attention policies helps the autonomous agent to discover and learn useful skills. These skills can be further combined to address compositional tasks like the manipulation of several different objects. 

## Install

First, create and activate a new conda environment with 

```
conda env create -f smorl_env.yml
conda activate smorl
```

Next, install out modification of `rlkit` and `multiworld` packages. For this, change your dir to the cloned repo and run 

```
python -m pip install -e ./multiworld
python -m pip install -e ./rlkit
```

Finally, to test that everything installed correctly you can run tests for SMORL:

```
python -W ignore  ./rlkit/settings/smorl/smoke_test.py
```

and for baselines: 

```
python -W ignore  ./rlkit/settings/baselines/sac/smoke_test.py
python -W ignore  ./rlkit/settings/baselines/rig/smoke_test.py
python -W ignore  ./rlkit/settings/baselines/skewfit/smoke_test.py
```
## Usage 

### SMORL training
All the hyperparameters files are stored in `./rlkit/settings` folder. 

To run SMORL training, just execute the corresponding `exp_name.py` file. For example, for SMORL training on GT representations in 3 objects Rearrange environment run 


```
python  ./rlkit/settings/smorl/gt_rearrange_3_object.py
``` 


### SCALOR training for visual environments

For Visual Environments, we have provided a trained SCALOR model. If you want you can train SCALOR model from scratch by running

```
python  ./rlkit/settings/scalor_training/exp_name.py
```
After SCALOR training, you can use the saved checkpoint path in the corresponding SMORL training setting by  
```
variant["scalor_path"]="path/to/checkpoint"
```

### Visualization of the results 
The visualization of the results is similar to the original `rlkit` package. 
So please install `viskit` package from [here](https://github.com/vitchyr/viskit). 

After this, you can run a visualization of the results by 

```
python viskit/frontend.py ./rlkit/data/exp_name/
```

Also, you can visualize learned policy by running 

```
python rlkit/scripts/run_goal_conditioned_policy.py ./rlkit/data/exp_name/params.pkl
```

## Citation

Please use the following bibtex entry to cite us:

```
@inproceedings{Zadaianchuk2021SelfSupervisedVRL,
  title={Self-supervised Visual Reinforcement Learning with Object-centric Representations},
  author={Andrii Zadaianchuk and Maximilian Seitzer and Georg Martius},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=xppLmXCbOw1}
}
```

## Credits

We used [RLkit](https://github.com/vitchyr/rlkit) and [multiworld](https://github.com/vitchyr/multiworld) by Vitchyr Pong for RL infrastructure and SAC+HER training implementation as well as environments implementations. Also, the SCALOR implementation is an adapted version of the official [SCALOR implementation](https://github.com/JindongJiang/SCALOR) by Jindong Jiang.
