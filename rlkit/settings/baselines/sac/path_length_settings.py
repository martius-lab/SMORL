

class _PathConf:
    def __init__(self, n_items, subtask_len, train_path_len=None):
        assert subtask_len is not None
        self._n_items = n_items

        self._subtask_len = subtask_len
        if train_path_len is None:
            self._train_path_len = subtask_len
        else:
            self._train_path_len = train_path_len

        self._eval_path_len = subtask_len * self.attempts

    @property
    def n_items(self):
        return self._n_items

    @property
    def subtask_length(self):
        return self._subtask_len

    @property
    def train_path_length(self):
        return self._train_path_len

    @property
    def eval_path_length(self):
        return self._eval_path_len

    @property
    def attempts(self):
        return 2 * self._n_items - 1


assert _PathConf(2, 20).subtask_length == 20
assert _PathConf(2, 20).train_path_length == 20
assert _PathConf(2, 20).eval_path_length == 60
assert _PathConf(3, 15, 50).subtask_length == 15
assert _PathConf(3, 15, 50).train_path_length == 50
assert _PathConf(3, 15, 50).eval_path_length == 75

_PATH_CONFIGS = {
    'SAC-Push-1obj': _PathConf(2, subtask_len=15, train_path_len=50),
    'SAC-Push-2obj': _PathConf(3, subtask_len=15, train_path_len=50),
    'SAC-Push-3obj': _PathConf(4, subtask_len=15, train_path_len=50),
    'SAC-Rearrange-1obj': _PathConf(2, subtask_len=20, train_path_len=50),
    'SAC-Rearrange-2obj': _PathConf(3, subtask_len=20, train_path_len=50),
    'SAC-Rearrange-3obj': _PathConf(4, subtask_len=20, train_path_len=50),
    'SAC-Rearrange-4obj': _PathConf(5, subtask_len=20, train_path_len=50),

    'RIG-Push-1obj': _PathConf(2, subtask_len=15, train_path_len=50),
    'RIG-Push-2obj': _PathConf(3, subtask_len=15, train_path_len=50),
    'RIG-Push-3obj': _PathConf(4, subtask_len=15, train_path_len=50),
    'RIG-Rearrange-1obj': _PathConf(2, subtask_len=20, train_path_len=50),
    'RIG-Rearrange-2obj': _PathConf(3, subtask_len=20, train_path_len=50),
    'RIG-Rearrange-3obj': _PathConf(4, subtask_len=20, train_path_len=50),

    'Skewfit-Push-1obj': _PathConf(2, subtask_len=15, train_path_len=50),
    'Skewfit-Push-2obj': _PathConf(3, subtask_len=15, train_path_len=50),
    'Skewfit-Push-3obj': _PathConf(4, subtask_len=15, train_path_len=50),
    'Skewfit-Rearrange-1obj': _PathConf(2, subtask_len=20, train_path_len=50),
    'Skewfit-Rearrange-2obj': _PathConf(3, subtask_len=20, train_path_len=50),
    'Skewfit-Rearrange-3obj': _PathConf(4, subtask_len=20, train_path_len=50),

    'MOURL-Push-1obj': _PathConf(2, subtask_len=15),
    'MOURL-Push-2obj': _PathConf(3, subtask_len=15),
    'MOURL-Push-3obj': _PathConf(4, subtask_len=15),
    'MOURL-Rearrange-1obj': _PathConf(2, subtask_len=20),
    'MOURL-Rearrange-2obj': _PathConf(3, subtask_len=20),
    'MOURL-Rearrange-3obj': _PathConf(4, subtask_len=20),
    'MOURL-Rearrange-4obj': _PathConf(5, subtask_len=20),

    'SCALOR-Push-1obj': _PathConf(2, subtask_len=15),
    'SCALOR-Push-2obj': _PathConf(3, subtask_len=15),
    'SCALOR-Push-3obj': _PathConf(4, subtask_len=15),
    'SCALOR-Rearrange-1obj': _PathConf(2, subtask_len=20),
    'SCALOR-Rearrange-2obj': _PathConf(3, subtask_len=20),
    'SCALOR-Rearrange-3obj': _PathConf(4, subtask_len=20),
    'SCALOR-Rearrange-4obj': _PathConf(5, subtask_len=20),
}


def get_path_settings(method, task, n_objects):
    key = '{}-{}-{}obj'.format(method, task, n_objects)
    return _PATH_CONFIGS[key]
