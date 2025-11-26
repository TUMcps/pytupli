from typing import Optional, Any, Tuple
from pytupli.dataset import TupliDataset
import warnings
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from hyperloglog import HyperLogLog

# Optional torch import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes to avoid import errors
    torch = None
    nn = None
    optim = None


class QualityMetric:
    def evaluate(self, data: TupliDataset) -> float:
        raise NotImplementedError('Subclasses should implement this method.')


class SACoMetric(QualityMetric):
    def __init__(
        self,
        environment: gym.Env,
        reference_states: np.ndarray = None,
        reference_actions: np.ndarray = None,
        use_hyperloglog: bool = True,
        observation_preprocessor: Optional[Any] = None,
    ):
        """
        Compute SACo (State-Action Coverage) metric for a dataset according to:
        Schweighofer, Kajetan, et al. "A dataset perspective on offline reinforcement learning."
        Conference on Lifelong Learning Agents. PMLR, 2022.

        SACo measures exploration by computing the normalized count of unique
        state-action pairs. It's defined as:

        SACo(D) = u_s,a(D) / u_s,a(D_ref)

        where u_s,a(D) is the number of unique state-action pairs in dataset D,
        and D_ref is a reference dataset (typically the replay dataset from online training).
        Args:
            reference_states (np.ndarray): States from the reference dataset.
            reference_actions (np.ndarray): Actions from the reference dataset.
            use_hyperloglog (bool): Whether to use HyperLogLog for approximate counting.
            observation_preprocessor (Optional[Any]): Optional preprocessor for observations, e.g., if they are continuous.
        """
        super().__init__()
        assert isinstance(environment.action_space, Discrete), (
            'SACo currently only supports discrete action spaces.'
        )
        if (
            not isinstance(environment.observation_space, Discrete)
            and observation_preprocessor is None
        ):
            warnings.warn(
                'SACo is designed for discrete state spaces. '
                'Please provide an observation_preprocessor to discretize or encode continuous states.'
            )
        self.observation_space = environment.observation_space

        if reference_states is not None:
            assert reference_states.shape[0] == reference_actions.shape[0], (
                'Reference states and actions must have the same number of samples.'
            )
            if len(reference_states.shape) == 1:
                reference_states = reference_states.reshape(-1, 1)
            if len(reference_actions.shape) == 1:
                reference_actions = reference_actions.reshape(-1, 1)
        self.reference_states = reference_states
        self.reference_actions = reference_actions
        self.use_hyperloglog = use_hyperloglog
        self.observation_preprocessor = observation_preprocessor

    def _count_unique_state_actions(
        self, states: np.ndarray, actions: np.ndarray, use_hyperloglog: bool = True
    ) -> int:
        """
        Count unique state-action pairs in a dataset.

        Args:
            states: Array of shape (n_samples, state_dim)
            actions: Array of shape (n_samples, action_dim)
            use_hyperloglog: If True, uses probabilistic HyperLogLog counting (memory efficient).
                            If False, uses exact set-based counting.

        Returns:
            Number of unique state-action pairs
        """
        if self.observation_preprocessor is not None:
            if not TORCH_AVAILABLE:
                raise ImportError(
                    'PyTorch is required when using observation_preprocessor. '
                    'Please install it with: pip install torch'
                )
            with torch.no_grad():
                states_tensor = torch.tensor(states, dtype=torch.float32)
                encoded_states = self.observation_preprocessor(states_tensor).cpu().numpy()
            states = encoded_states
        # Concatenate states and actions for pairing
        state_actions = np.concatenate([states, actions.reshape(-1, 1)], axis=1)

        if use_hyperloglog:
            return self._count_with_hyperloglog(state_actions)
        else:
            return self._count_exact(state_actions)

    def _count_exact(self, state_actions: np.ndarray) -> int:
        """
        Count unique state-action pairs exactly using a set.

        Args:
            state_actions: Array of shape (n_samples, feature_dim)

        Returns:
            Exact count of unique state-action pairs
        """
        unique = set()
        for sa_pair in state_actions:
            # Convert to tuple for hashability
            unique.add(tuple(sa_pair))
        return len(unique)

    def _count_with_hyperloglog(self, state_actions: np.ndarray, error_rate: float = 0.01) -> int:
        """
        Count unique state-action pairs using HyperLogLog probabilistic data structure.

        This is more memory-efficient for very large datasets. Typical error rate is ~2%
        for error_rate=0.01 even for N > 10^9 samples.

        Args:
            state_actions: Array of shape (n_samples, feature_dim)
            error_rate: Desired error rate for HyperLogLog. Lower values use more memory.
                    Common values: 0.01 (1%), 0.001 (0.1%)

        Returns:
            Approximate count of unique state-action pairs
        """
        hll = HyperLogLog(error_rate)

        for sa_pair in state_actions:
            # Convert to string for hashing
            hll.add(','.join([str(x) for x in sa_pair]))

        return len(hll)

    def evaluate(self, data: TupliDataset) -> float:
        states = np.array(data.observations)
        actions = np.array(data.actions)
        # Compute unique state-action pairs in dataset
        unique_sa = self._count_unique_state_actions(states, actions, self.use_hyperloglog)

        # If no reference provided, no normalization happens
        if self.reference_states is None or self.reference_actions is None:
            saco = unique_sa
        else:
            unique_sa_ref = self._count_unique_state_actions(
                self.reference_states, self.reference_actions, self.use_hyperloglog
            )
            saco = unique_sa / unique_sa_ref if unique_sa_ref > 0 else 0.0

        return saco


class AverageReturnMetric(QualityMetric):
    def __init__(
        self, gamma: float = 0.99, normalization_range: Optional[Tuple[float, float]] = None
    ):
        """
        Cumulative return metric (also termed "trajectory quality", TQ) according to:
        Schweighofer, Kajetan, et al. "A dataset perspective on offline reinforcement learning."
        Conference on Lifelong Learning Agents. PMLR, 2022.
        Args:
            gamma (float): Discount factor for future rewards.
            normalization_range (Optional[Tuple[float, float]]): Range for normalizing the mean
                return. Schweighofer et al. use a random policy for generating the lower bound
                and the best policy from online training to generate the upper bound.
        """
        super().__init__()
        self.gamma = gamma
        self.normalization_range = normalization_range

    def evaluate(self, data: TupliDataset) -> float:
        """
        Evaluate the cumulative return metric on the provided data.

        Args:
            data (TupliDataset): Dataset containing trajectories.
        Returns:
            float: The cumulative return.
        """
        assert data.episodes, 'Dataset must contain episodes to compute average return.'

        total_return = 0.0
        for episode in data.episodes:
            episode_return = 0.0
            discount = 1.0
            for step in episode.tuples:
                episode_return += step.reward * discount
                discount *= self.gamma
            total_return += episode_return
        mean_return = total_return / len(data.episodes) if data.episodes else 0.0
        return self.normalize_mean_return(mean_return)

    def normalize_mean_return(self, mean_return: float) -> float:
        """
        Normalize the mean return based on the specified normalization range.
        Args:
            mean_return (float): The mean return to normalize.
        Returns:
            float: The normalized mean return.
        """
        if self.normalization_range is None:
            return mean_return
        min_val, max_val = self.normalization_range
        return (mean_return - min_val) / (max_val - min_val)


class EstimatedReturnImprovementMetric(QualityMetric):
    def __init__(self, gamma: float = 0.99, min_return: Optional[float] = None):
        """
        Estimated relative return improvement (ERI) metric according to:
        Swazinna, Phillip, Steffen Udluft, and Thomas Runkler.
        "Measuring data quality for dataset selection in offline reinforcement learning."
        2021 IEEE Symposium Series on Computational Intelligence (SSCI).

        The ERI requires returns to be lower-bound by 0. Therefore, we need to know the minimum reward.
        If this is not provided, we will estimate it from the dataset.

        Args:
            gamma (float): Discount factor for future rewards.
            min_reward (Optional[float]): Minimum reward value for shifting.
        """
        super().__init__()
        self.gamma = gamma
        self.min_return = min_return

    def evaluate(self, data: TupliDataset) -> float:
        """
        Evaluate the estimated return improvement metric on the provided data.

        Args:
            data (TupliDataset): Dataset containing trajectories.
        Returns:
            float: The estimated return improvement.
        """
        assert data.episodes, 'Dataset must contain episodes to compute ERI.'

        returns = []
        for episode in data.episodes:
            episode_return = 0.0
            discount = 1.0
            for step in episode.tuples:
                episode_return += step.reward * discount
                discount *= self.gamma
            returns.append(episode_return)
        if self.min_return is None:
            self.min_return = min(returns)
        shifted_returns = [r - self.min_return for r in returns]
        mean_shifted_return = sum(shifted_returns) / len(shifted_returns)
        if mean_shifted_return == 0:
            return 0.0
        eri = (max(shifted_returns) - mean_shifted_return) / mean_shifted_return
        return eri


class GeneralizedBehavioralEntropyMetric(QualityMetric):
    def __init__(
        self,
        rep_dim: int,
        alpha: float = 0.7,
        num_knn: int = 10,
        use_mean_normalization: bool = True,
        knn_clip: float = 0.0,
        use_knn_avg: bool = False,
        device: str = 'cpu',
        observation_encoder: Optional[Any] = None,
    ):
        """
        Generalized Behavioral Entropy (GBE) metric according to:
        Suttle, Wesley A., Aamodh Suresh, and Carlos Nieto-Granda.
        "Behavioral Entropy-Guided Dataset Generation for Offline Reinforcement Learning."
        arXiv preprint arXiv:2502.04141 (2025).

        For alpha=1.0, this reduces to Shannon Entropy.

        Args:
            rep_dim (int): Dimensionality of the representation space. Either state space dim or latent space dim.
            alpha (float): Controls probability distortion under Prelec weighting function.
                Recommended values from paper: {0.2, 0.5, 0.7, 0.9, 1.5, 2.0, 3.0, 5.0}
            num_knn (int): Number of nearest neighbors to consider.
                Recommended values: {
                5-15 for robotics state vectors (as in URLB, BE paper, APT paper),
                3-5 for small datasets,
                10-20 for >50k points
                }
            use_mean_normalization (bool): Whether to use mean and std of the dataset for normalization.
            knn_clip (float): Clipping value for k-NN distances.
            use_knn_avg (bool): Whether to average over k-NN distances.
            device (str): Device to perform computations on ('cpu' or 'cuda'). Only used if observation_encoder is provided.
            observation_encoder (Optional[torch.nn.Module]): Optional encoder to obtain representations from states.
                If provided, PyTorch is required.
        """
        super().__init__()
        if observation_encoder is not None and not TORCH_AVAILABLE:
            raise ImportError(
                'PyTorch is required when using observation_encoder. '
                'Please install it with: pip install torch'
            )
        self.rep_dim = rep_dim
        self.alpha = alpha
        # compute beta according to paper
        self.beta = np.power(np.log(rep_dim), 1 - alpha)
        self.num_knn = num_knn
        self.use_mean_normalization = use_mean_normalization
        self.knn_clip = knn_clip
        self.use_knn_avg = use_knn_avg
        self.device = device
        self.observation_encoder = observation_encoder.to(device) if observation_encoder else None

    def evaluate(self, data: TupliDataset) -> float:
        """
        Evaluate the Generalized Behavioral Entropy metric on the provided data.
        Args:
            data (TupliDataset): Dataset containing trajectories.
        Returns:
            float: The Generalized Behavioral Entropy value.
        """
        assert data.episodes, 'Dataset must contain episodes to compute GBE.'
        # Extract representations from states if needed.
        if self.observation_encoder:
            with torch.no_grad():
                observations = torch.tensor(
                    data.observations, device=self.device, dtype=torch.float32
                )
                rep = self.observation_encoder(observations).cpu().numpy()
                if rep.shape[-1] != self.rep_dim:
                    raise ValueError(
                        f'Encoded representation dimension {rep.shape[-1]} does not match specified rep_dim {self.rep_dim}.'
                    )
        else:
            rep = np.array(data.observations, dtype=np.float32)
            if len(rep.shape) > 2:
                raise ValueError(
                    'Observation encoder must be provided for high-dimensional observations (e.g., images).'
                )
        entropies = self.compute_behavioral_entropy(rep)
        return float(np.mean(entropies))

    def compute_mean(self, rep: np.ndarray) -> float:
        # numerically stable computation according to code from [Wesley et al., 2025]
        mean = 0.0
        eps = 1e-4
        bs = rep.shape[0]
        delta = np.mean(rep) - mean
        new_M = mean + delta * bs / (eps + bs)
        return new_M

    def compute_behavioral_entropy(self, rep: np.ndarray) -> np.ndarray:
        """
        Compute behavioral entropy using k-nearest neighbors.

        Args:
            rep: Array of shape (n_samples, feature_dim)

        Returns:
            Array of shape (n_samples, 1) with entropy values
        """
        source = target = rep
        b1, b2 = source.shape[0], target.shape[0]

        # Compute pairwise L2 distances
        # (b1, 1, c) - (1, b2, c) -> (b1, b2)
        source_expanded = source[:, np.newaxis, :]  # (b1, 1, c)
        target_expanded = target[np.newaxis, :, :]  # (1, b2, c)
        diff = source_expanded - target_expanded  # (b1, b2, c)
        sim_matrix = np.linalg.norm(diff, axis=-1, ord=2)  # (b1, b2)

        # Get top-k nearest neighbors (smallest distances)
        # np.partition is faster than full sort for k-th element
        if self.num_knn < b2:
            # Use argpartition to find k smallest indices
            kth_indices = np.argpartition(sim_matrix, self.num_knn, axis=1)[:, : self.num_knn]
            # Get the actual distances
            reward = np.take_along_axis(sim_matrix, kth_indices, axis=1)  # (b1, k)
            # Sort the k nearest neighbors
            reward = np.sort(reward, axis=1)  # (b1, k)
        else:
            # If k >= dataset size, just sort all
            reward = np.sort(sim_matrix, axis=1)[:, : self.num_knn]  # (b1, k)

        if not self.use_knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.compute_mean(reward) if self.use_mean_normalization else 1.0
            reward = (
                np.maximum(reward - self.knn_clip, 0.0) if self.knn_clip >= 0.0 else reward
            )  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.compute_mean(reward) if self.use_mean_normalization else 1.0
            reward = np.maximum(reward - self.knn_clip, 0.0) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.num_knn))  # (b1, k)
            reward = reward.mean(axis=1, keepdims=True)  # (b1, 1)

        reward_exponent = self.beta * np.power(np.log(reward + 1), self.alpha)
        reward = reward * np.exp(-reward_exponent) * reward_exponent
        return reward


class MLP(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_dim, out_dim, hidden=(256, 256)):
        """
        Simple feedforward MLP.
        Args:
            in_dim (int): Input dimensionality. For discrete actions, this is just state dim.
            out_dim (int): Output dimensionality. For discrete actions, this is |A|.
            hidden (tuple): Hidden layer sizes.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                'PyTorch is required for MLP. Please install it with: pip install torch'
            )
        super().__init__()
        layers = []
        dim = in_dim
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QFunctionMetric(QualityMetric):
    def __init__(
        self,
        env: gym.Env,
        network_arch: MLP,
        network_kwargs: dict = None,
        gamma: float = 0.99,
        lr: float = 3e-4,
        batch_size: int = 256,
        iterations: int = 20000,
        target_update_rate: float = 0.005,
        device: str = 'cpu',
        evaluation_policy: Optional[Any] = None,  # π(a|s) for FQE; None → FQI
    ):
        """
        Trains a Q-function for the behavior policy and computes the mean Q-values over the dataset.
        Essentially, we fit
        $Q_\theta \leftarrow \argmin_\theta E_{(s,a,r,s') \sim D} (Q_\theta(s,a) - (r + \gamma V(s')))^2$,
        For discrete action spaces, we can perform fitted Q-iteration (FQI). In continuous action spaces,
        we have to do fitted Q-evaluation (FQE) with an evaluation policy. This policy could be extracted
        from the dataset using behavioral cloning or a k-nearest neighbor approach. Then we get
            - FQI: $V(s') = \max_a Q_{\overline{\theta}}(s', a)$
            - FQE: $V(s') = \mathbb{E}_{a' \sim \pi(. | s')} [Q_{\overline{\theta}}(s', a')]$
        which makes use of a target Q-function.
        Args:
            device (str): Device to perform computations on ('cpu' or 'cuda').
        """
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError(
                'PyTorch is required for QFunctionMetric. Please install it with: pip install torch'
            )
        self.discrete_actions = isinstance(env.action_space, Discrete)
        if not self.discrete_actions and evaluation_policy is None:
            raise ValueError(
                'For continuous action spaces, you have to provide an evaluation policy.'
            )
        # Initialize Q-function and target Q-function
        self.q_function = network_arch(**network_kwargs)
        self.q_function_target = network_arch(**network_kwargs)
        self.q_function_target.load_state_dict(self.q_function.state_dict())

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.iterations = iterations
        self.tau = target_update_rate
        self.evaluation_policy = evaluation_policy
        self.device = device

        self.optimizer = optim.Adam(self.q_function.parameters(), lr=self.lr)

    def evaluate(self, data: TupliDataset) -> float:
        """
        Evaluate the average Q-value on the provided data after training the Q-function.
        Args:
            data (TupliDataset): Dataset containing trajectories.
        Returns:
            float: The average Q-value over the dataset.
        """
        self.train_q_functions(data)
        with torch.no_grad():
            observations = torch.tensor(data.observations, device=self.device, dtype=torch.float32)
            actions = torch.tensor(data.actions, device=self.device, dtype=torch.float32)
            if self.discrete_actions:
                q_values = self.q_function(observations).gather(1, actions.long().view(-1, 1))
            else:
                q_values = self.q_function(torch.cat([observations, actions], dim=-1))
            mean_q_value = q_values.mean().item()
        return mean_q_value

    def sample_batch(self, data: dict) -> Tuple:
        """
        Sample a batch of transitions from the dataset.
        Args:
            data (dict): Dataset in d4rl format.
        Returns:
            Tuple: Batch of (obs, act, rew, next_obs, done) tensors.
        """
        dataset_size = data['observations'].shape[0]
        idx = torch.randint(0, dataset_size, (self.batch_size,))
        obs = torch.tensor(data['observations'][idx], device=self.device)
        act = torch.tensor(data['actions'][idx], device=self.device)
        rew = torch.tensor(data['rewards'][idx], device=self.device)
        next_obs = torch.tensor(data['next_observations'][idx], device=self.device)
        done = torch.tensor(data['terminals'][idx], device=self.device)
        return (obs, act, rew, next_obs, done)

    def train_q_functions(self, data: TupliDataset):
        """
        Train the Q-function using FQI or FQE on the provided dataset.
        Args:
            data (TupliDataset): Dataset containing trajectories.
        Returns:
            Trained Q-function.
        """
        data_dict = data.convert_to_d4rl_format()
        for it in range(self.iterations):
            s, a, r, s_next, d = self.sample_batch(data_dict)

            # ---- Compute target y = r + γ * ( ... ) ----
            with torch.no_grad():
                if self.evaluation_policy is None:
                    # ---------- FQI (max over actions) ----------
                    q_next = self.q_function_target(s_next)  # (B, |A|)
                    q_next = q_next.max(dim=1, keepdim=True)[0]  # max_a Q(s',a)

                else:
                    # ---------- FQE (policy evaluation) ----------
                    if self.discrete_actions:
                        # π(a|s') is a categorical distribution
                        pi = self.evaluation_policy(s_next)  # shape (B, |A|)
                        q_next_all = self.q_function_target(s_next)  # (B, |A|)
                        q_next = (pi * q_next_all).sum(dim=1, keepdim=True)
                    else:
                        # π(s') → a', continuous
                        a_next = self.evaluation_policy(s_next)
                        q_next = self.q_target(torch.cat([s_next, a_next], dim=-1))

                target = (
                    r.reshape(q_next.size())
                    + self.gamma * (1.0 - d.int()).reshape(q_next.size()) * q_next
                )

            # ---- Compute Q(s,a) ----
            if self.discrete_actions:
                q_pred = self.q_function(s).gather(
                    1, a.long().view(-1, 1)
                )  # standard Q-learning target
            else:
                q_pred = self.q_function(torch.cat([s, a], dim=-1))

            # ---- Loss ----
            loss = ((q_pred - target) ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---- Soft target update ----
            for param, target_param in zip(
                self.q_function.parameters(), self.q_function_target.parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if (it + 1) % 500 == 0:
                print(f'Iteration {it + 1}, loss={loss.item():.4f}')

        return self.q_function
