import numpy as np

# TODO 参考GPT最后一次的答案，试着平滑由噪音带来的不稳定，state normalization而奖励不，噪声恰恰体现在奖励，这所出现的问题
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, init_mean=None, init_std=None, init_n=1):
        """
        Initialize RunningMeanStd with optional initial values.
        :param shape: The shape of the features (e.g., number of numerical features).
        :param init_mean: Initial mean for each feature (default: zeros).
        :param init_std: Initial std for each feature (default: ones).
        :param init_n: Initial number of samples (default: 1).
        """
        self.n = init_n  # 初始样本数，默认为1
        self.mean = np.zeros(shape, dtype=np.float32) if init_mean is None else np.array(init_mean, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32) if init_std is None else np.array(init_std, dtype=np.float32)
        # 根据std和样本数反推S
        self.S = (self.std ** 2) * self.n

    def update(self, x):
        x = np.array(x, dtype=np.float32)
        self.n += 1

        old_mean = self.mean.copy()
        self.mean += (x - old_mean) / self.n
        self.S += (x - old_mean) * (x - self.mean)

        # 确保标准差始终有一个下限，避免归一化结果出现异常
        self.std = np.sqrt(np.maximum(self.S / max(self.n, 1), 1e-6))


class Normalization:
    def __init__(self, num_categorical, num_numerical, running_ms=None):
        """
        :param num_categorical: Number of categorical features.
        :param num_numerical: Number of numerical features.
        :param running_ms: Pre-initialized RunningMeanStd object (optional).
        """
        self.num_categorical = num_categorical  # Categorical feature count
        self.num_numerical = num_numerical  # Numerical feature count
        self.running_ms = running_ms if running_ms else RunningMeanStd(shape=(num_numerical,))

    def __call__(self, x, update=True):
        # Separate categorical and numerical features
        x_categorical = x[:self.num_categorical]
        x_numerical = np.array(x[self.num_categorical:], dtype=np.float32)

        if update:
            self.running_ms.update(x_numerical)

        # Normalize only the numerical features
        x_numerical = (x_numerical - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        # Concatenate categorical and normalized numerical features
        x_normalized = np.concatenate([x_categorical, x_numerical])
        return x_normalized

    def denormal(self, x):
        x_categorical = x[:self.num_categorical]
        x_numerical = np.array(x[self.num_categorical:], dtype=np.float32)

        # Denormalize numerical features
        x_numerical = x_numerical * (self.running_ms.std + 1e-8) + self.running_ms.mean

        # Concatenate categorical and denormalized numerical features
        x_denormalized = np.concatenate([x_categorical, x_numerical])
        return x_denormalized


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x[0]

    def reset(self):  # When an episode is done, we should reset 'self.R'
        self.R = np.zeros(self.shape)
