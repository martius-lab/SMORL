from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.envs.scalor_wrapper import SCALORWrappedEnv

class WrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            goal_sampling_mode,
            env,
            policy,
            decode_goals=False,
            **kwargs):
        super().__init__(env, policy, **kwargs)
        self._goal_sampling_mode = goal_sampling_mode
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):

        if isinstance(self._env, (SCALORWrappedEnv)):
            return dict(
                policy=self._policy,
            )
        else:
            return dict(
                env=self._env,
                policy=self._policy,
            )
