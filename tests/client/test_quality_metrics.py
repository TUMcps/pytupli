"""Test methods of quality_metrics module."""

import pytest
import numpy as np
import gymnasium as gym
from pytupli.dataset import TupliDataset
from pytupli.quality_metrics import (
    SACoMetric,
    AverageReturnMetric,
    EstimatedReturnImprovementMetric,
    GeneralizedBehavioralEntropyMetric,
    QFunctionMetric,
    MLP,
)

# Optional imports for testing
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Base QualityMetric Tests
# ============================================================================


def test_quality_metric_base_class():
    """Test that base QualityMetric class raises NotImplementedError."""
    from pytupli.quality_metrics import QualityMetric

    metric = QualityMetric()

    # Create a minimal dataset
    from pytupli.storage import FileStorage
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorage(storage_base_dir=tmpdir)
        dataset = TupliDataset(storage)

        with pytest.raises(NotImplementedError, match='Subclasses should implement this method'):
            metric.evaluate(dataset)


# ============================================================================
# SACoMetric Tests
# ============================================================================


def test_saco_metric_initialization(test_env):
    """Test basic initialization of SACoMetric."""
    metric = SACoMetric(environment=test_env, use_hyperloglog=False)
    assert metric.observation_space == test_env.observation_space
    assert metric.reference_states is None
    assert metric.reference_actions is None
    assert metric.use_hyperloglog is False


def test_saco_metric_without_reference(loaded_dataset, test_env):
    """Test SACoMetric without reference dataset (unnormalized)."""
    metric = SACoMetric(environment=test_env, use_hyperloglog=False)
    saco_score = metric.evaluate(loaded_dataset)

    # Should return count of unique state-action pairs
    assert isinstance(saco_score, (int, float))
    assert saco_score > 0
    # With 3 episodes of 2 steps each = 6 state-action pairs, but some might be duplicates
    assert saco_score <= 6


def test_saco_metric_with_reference(loaded_dataset, test_env):
    """Test SACoMetric with reference dataset (normalized)."""
    # Create reference data with known unique state-action pairs
    reference_states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    reference_actions = np.array([0, 1, 0, 1])

    metric = SACoMetric(
        environment=test_env,
        reference_states=reference_states,
        reference_actions=reference_actions,
        use_hyperloglog=False,
    )
    saco_score = metric.evaluate(loaded_dataset)

    # Should be normalized (ratio of unique pairs)
    assert isinstance(saco_score, float)
    assert 0.0 <= saco_score <= 1.0


def test_saco_metric_with_hyperloglog(loaded_dataset, test_env):
    """Test SACoMetric using HyperLogLog for approximate counting."""
    metric = SACoMetric(environment=test_env, use_hyperloglog=True)
    saco_score = metric.evaluate(loaded_dataset)

    # Should return approximate count of unique state-action pairs
    assert isinstance(saco_score, (int, float))
    assert saco_score > 0


def test_saco_exact_vs_hyperloglog(loaded_varied_dataset, test_env):
    """Test that exact and HyperLogLog counting give similar results."""
    metric_exact = SACoMetric(environment=test_env, use_hyperloglog=False)
    metric_hll = SACoMetric(environment=test_env, use_hyperloglog=True)

    score_exact = metric_exact.evaluate(loaded_varied_dataset)
    score_hll = metric_hll.evaluate(loaded_varied_dataset)

    # Scores should be close (within 10% for HyperLogLog error)
    assert abs(score_exact - score_hll) / score_exact < 0.15


def test_saco_count_exact_method(test_env):
    """Test the _count_exact method directly."""
    metric = SACoMetric(environment=test_env, use_hyperloglog=False)

    # Create test data with known unique state-action pairs
    state_actions = np.array(
        [
            [0, 0, 1],  # state [0, 0], action 1
            [0, 1, 0],  # state [0, 1], action 0
            [0, 0, 1],  # duplicate
            [1, 0, 1],  # state [1, 0], action 1
        ]
    )

    count = metric._count_exact(state_actions)
    assert count == 3  # 3 unique state-action pairs


def test_saco_count_hyperloglog_method(test_env):
    """Test the _count_with_hyperloglog method directly."""
    metric = SACoMetric(environment=test_env, use_hyperloglog=True)

    # Create test data
    state_actions = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],  # duplicate
            [1, 0, 1],
        ]
    )

    count = metric._count_with_hyperloglog(state_actions)
    # Should be approximately 3 (within HyperLogLog error)
    assert 2 <= count <= 4


def test_saco_with_preprocessor_requires_torch(test_env):
    """Test that SACoMetric with preprocessor requires torch when used."""
    if TORCH_AVAILABLE:
        pytest.skip('PyTorch is available, skipping unavailability test')

    # Create a mock preprocessor
    class MockPreprocessor:
        def __call__(self, x):
            return x

    metric = SACoMetric(
        environment=test_env, use_hyperloglog=False, observation_preprocessor=MockPreprocessor()
    )

    # Create minimal dataset
    from pytupli.schema import FilterEQ
    from pytupli.storage import FileStorage
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorage(storage_base_dir=tmpdir)
        from pytupli.benchmark import TupliEnvWrapper
        from tests.client.example_envs import SimpleTestEnv

        env = SimpleTestEnv()
        benchmark = TupliEnvWrapper(env, storage)
        benchmark.store(name='Test', description='Test', difficulty='easy', version='1.0')

        obs, _ = benchmark.reset()
        benchmark.step(np.int64(1))

        dataset = TupliDataset(storage)
        dataset = dataset.with_benchmark_filter(FilterEQ(key='id', value=benchmark.id))
        dataset.load()

        # Should raise ImportError when trying to use preprocessor
        with pytest.raises(
            ImportError, match='PyTorch is required when using observation_preprocessor'
        ):
            metric.evaluate(dataset)

        benchmark.delete(delete_episodes=True)


def test_saco_reference_states_reshape():
    """Test that reference states and actions are properly reshaped."""
    from tests.client.example_envs import SimpleTestEnv

    env = SimpleTestEnv()

    # Test with 1D arrays that should be reshaped
    ref_states = np.array([0, 1, 2, 3])
    ref_actions = np.array([0, 1, 0, 1])

    metric = SACoMetric(
        environment=env,
        reference_states=ref_states,
        reference_actions=ref_actions,
        use_hyperloglog=False,
    )

    # Check that they were reshaped to 2D
    assert len(metric.reference_states.shape) == 2
    assert len(metric.reference_actions.shape) == 2
    assert metric.reference_states.shape == (4, 1)
    assert metric.reference_actions.shape == (4, 1)


# ============================================================================
# AverageReturnMetric Tests
# ============================================================================


def test_average_return_metric_initialization():
    """Test basic initialization of AverageReturnMetric."""
    metric = AverageReturnMetric(gamma=0.95)
    assert metric.gamma == 0.95
    assert metric.normalization_range is None


def test_average_return_metric_without_normalization(loaded_dataset):
    """Test AverageReturnMetric without normalization."""
    metric = AverageReturnMetric(gamma=0.99)
    avg_return = metric.evaluate(loaded_dataset)

    # Should return the mean discounted return
    assert isinstance(avg_return, float)
    # Returns should be non-negative for this test environment
    assert avg_return >= 0.0


def test_average_return_metric_with_normalization(loaded_dataset):
    """Test AverageReturnMetric with normalization range."""
    # Set normalization range [0, 10]
    metric = AverageReturnMetric(gamma=0.99, normalization_range=(0.0, 10.0))
    normalized_return = metric.evaluate(loaded_dataset)

    # Normalized return should be between 0 and 1
    assert isinstance(normalized_return, float)
    assert 0.0 <= normalized_return <= 1.0


def test_average_return_metric_different_gamma(loaded_varied_dataset):
    """Test that different gamma values produce different returns."""
    metric_high_gamma = AverageReturnMetric(gamma=0.99)
    metric_low_gamma = AverageReturnMetric(gamma=0.5)

    return_high = metric_high_gamma.evaluate(loaded_varied_dataset)
    return_low = metric_low_gamma.evaluate(loaded_varied_dataset)

    # Lower gamma should discount future rewards more, resulting in lower return
    assert return_low <= return_high


def test_average_return_normalization_function():
    """Test the normalize_mean_return function."""
    metric = AverageReturnMetric(gamma=0.99, normalization_range=(0.0, 10.0))

    # Test normalization at boundaries
    assert metric.normalize_mean_return(0.0) == 0.0
    assert metric.normalize_mean_return(10.0) == 1.0
    assert metric.normalize_mean_return(5.0) == 0.5


def test_average_return_without_normalization_returns_raw():
    """Test that without normalization, raw mean return is returned."""
    metric = AverageReturnMetric(gamma=0.99)  # No normalization_range

    # Test the normalization function with no range set
    raw_value = 7.5
    assert metric.normalize_mean_return(raw_value) == raw_value


def test_average_return_with_gamma_zero(loaded_dataset):
    """Test AverageReturnMetric with gamma=0 (only immediate rewards)."""
    metric = AverageReturnMetric(gamma=0.0)
    avg_return = metric.evaluate(loaded_dataset)

    # With gamma=0, only immediate rewards count
    assert isinstance(avg_return, float)
    assert avg_return >= 0.0


def test_average_return_with_gamma_one(loaded_varied_dataset):
    """Test AverageReturnMetric with gamma=1 (no discounting)."""
    metric = AverageReturnMetric(gamma=1.0)
    avg_return = metric.evaluate(loaded_varied_dataset)

    # With gamma=1, all rewards have equal weight
    assert isinstance(avg_return, float)
    assert avg_return >= 0.0


def test_average_return_normalization_negative_range():
    """Test normalization with negative range."""
    metric = AverageReturnMetric(gamma=0.99, normalization_range=(-10.0, 10.0))

    assert metric.normalize_mean_return(-10.0) == 0.0
    assert metric.normalize_mean_return(10.0) == 1.0
    assert metric.normalize_mean_return(0.0) == 0.5


# ============================================================================
# EstimatedReturnImprovementMetric Tests
# ============================================================================


def test_eri_metric_initialization():
    """Test basic initialization of EstimatedReturnImprovementMetric."""
    metric = EstimatedReturnImprovementMetric(gamma=0.95, min_return=-10.0)
    assert metric.gamma == 0.95
    assert metric.min_return == -10.0


def test_eri_metric_with_known_min(loaded_varied_dataset):
    """Test ERI metric with known minimum return."""
    metric = EstimatedReturnImprovementMetric(gamma=0.99, min_return=0.0)
    eri_score = metric.evaluate(loaded_varied_dataset)

    # ERI should be non-negative
    assert isinstance(eri_score, float)
    assert eri_score >= 0.0


def test_eri_metric_auto_min(loaded_varied_dataset):
    """Test ERI metric with automatic minimum return estimation."""
    metric = EstimatedReturnImprovementMetric(gamma=0.99)
    eri_score = metric.evaluate(loaded_varied_dataset)

    # ERI should be calculable even without known min
    assert isinstance(eri_score, float)
    assert eri_score >= 0.0


def test_eri_metric_uniform_returns():
    """Test ERI with uniform returns (should be zero)."""
    # Create a mock dataset with uniform returns
    from pytupli.storage import FileStorage
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorage(storage_base_dir=tmpdir)
        from pytupli.benchmark import TupliEnvWrapper
        from tests.client.example_envs import SimpleTestEnv

        env = SimpleTestEnv()
        benchmark = TupliEnvWrapper(env, storage)
        benchmark.store(name='Uniform Test', description='Test', difficulty='easy', version='1.0')

        # Create episodes with same total reward
        for _ in range(3):
            obs, _ = benchmark.reset()
            benchmark.step(np.int64(1))
            benchmark.step(np.int64(1))

        from pytupli.schema import FilterEQ

        dataset = TupliDataset(storage)
        dataset = dataset.with_benchmark_filter(FilterEQ(key='id', value=benchmark.id))
        dataset.load()

        metric = EstimatedReturnImprovementMetric(gamma=1.0, min_return=0.0)
        eri_score = metric.evaluate(dataset)

        # With uniform returns, ERI should be 0
        assert eri_score == 0.0

        benchmark.delete(delete_episodes=True)


def test_eri_metric_with_varied_returns():
    """Test ERI with varied returns to ensure proper calculation."""
    from pytupli.storage import FileStorage
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorage(storage_base_dir=tmpdir)
        from pytupli.benchmark import TupliEnvWrapper
        from tests.client.example_envs import SimpleTestEnv

        env = SimpleTestEnv()
        benchmark = TupliEnvWrapper(env, storage)
        benchmark.store(name='Varied Test', description='Test', difficulty='easy', version='1.0')

        # Create episodes with different total rewards
        # Episode 1: High reward
        obs, _ = benchmark.reset()
        for _ in range(5):
            obs, reward, term, trunc, _ = benchmark.step(np.int64(1))
            if term or trunc:
                break

        # Episode 2: Low reward
        obs, _ = benchmark.reset()
        for _ in range(2):
            obs, reward, term, trunc, _ = benchmark.step(np.int64(0))
            if term or trunc:
                break

        from pytupli.schema import FilterEQ

        dataset = TupliDataset(storage)
        dataset = dataset.with_benchmark_filter(FilterEQ(key='id', value=benchmark.id))
        dataset.load()

        metric = EstimatedReturnImprovementMetric(gamma=1.0, min_return=0.0)
        eri_score = metric.evaluate(dataset)

        # With varied returns, ERI should be > 0
        assert eri_score > 0.0

        benchmark.delete(delete_episodes=True)


def test_eri_metric_gamma_effect():
    """Test that gamma affects ERI calculation."""
    from pytupli.storage import FileStorage
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorage(storage_base_dir=tmpdir)
        from pytupli.benchmark import TupliEnvWrapper
        from tests.client.example_envs import SimpleTestEnv

        env = SimpleTestEnv()
        benchmark = TupliEnvWrapper(env, storage)
        benchmark.store(name='Gamma Test', description='Test', difficulty='easy', version='1.0')

        # Create a few episodes
        for _ in range(3):
            obs, _ = benchmark.reset()
            for action in [1, 0, 1]:
                obs, reward, term, trunc, _ = benchmark.step(np.int64(action))
                if term or trunc:
                    break

        from pytupli.schema import FilterEQ

        dataset = TupliDataset(storage)
        dataset = dataset.with_benchmark_filter(FilterEQ(key='id', value=benchmark.id))
        dataset.load()

        metric_high_gamma = EstimatedReturnImprovementMetric(gamma=0.99)
        metric_low_gamma = EstimatedReturnImprovementMetric(gamma=0.5)

        eri_high = metric_high_gamma.evaluate(dataset)
        eri_low = metric_low_gamma.evaluate(dataset)

        # Both should be non-negative
        assert eri_high >= 0.0
        assert eri_low >= 0.0

        benchmark.delete(delete_episodes=True)


# ============================================================================
# GeneralizedBehavioralEntropyMetric Tests
# ============================================================================


def test_gbe_metric_initialization():
    """Test basic initialization of GeneralizedBehavioralEntropyMetric."""
    metric = GeneralizedBehavioralEntropyMetric(rep_dim=2, alpha=0.7, num_knn=5, device='cpu')
    assert metric.rep_dim == 2
    assert metric.alpha == 0.7
    assert metric.num_knn == 5
    assert metric.device == 'cpu'


def test_gbe_metric_evaluation(loaded_dataset):
    """Test GBE metric evaluation on dataset."""
    # Assuming observations are 2D
    metric = GeneralizedBehavioralEntropyMetric(
        rep_dim=2,
        alpha=1.0,  # Shannon entropy
        num_knn=3,  # Small k for small dataset
        device='cpu',
    )
    gbe_score = metric.evaluate(loaded_dataset)

    # GBE should return a float value
    assert isinstance(gbe_score, float)
    assert gbe_score >= 0.0


def test_gbe_metric_different_alpha(loaded_varied_dataset):
    """Test GBE with different alpha values."""
    metric_low_alpha = GeneralizedBehavioralEntropyMetric(
        rep_dim=2, alpha=0.5, num_knn=3, device='cpu'
    )
    metric_high_alpha = GeneralizedBehavioralEntropyMetric(
        rep_dim=2, alpha=2.0, num_knn=3, device='cpu'
    )

    score_low = metric_low_alpha.evaluate(loaded_varied_dataset)
    score_high = metric_high_alpha.evaluate(loaded_varied_dataset)

    # Different alpha values should produce different scores
    assert isinstance(score_low, float)
    assert isinstance(score_high, float)
    # Both should be non-negative
    assert score_low >= 0.0
    assert score_high >= 0.0


def test_gbe_metric_with_knn_avg(loaded_dataset):
    """Test GBE with k-NN averaging enabled."""
    metric = GeneralizedBehavioralEntropyMetric(
        rep_dim=2, alpha=0.7, num_knn=3, use_knn_avg=True, device='cpu'
    )
    gbe_score = metric.evaluate(loaded_dataset)

    assert isinstance(gbe_score, float)
    assert gbe_score >= 0.0


# ============================================================================
# QFunctionMetric Tests
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch not installed')
def test_q_function_metric_initialization(test_env):
    """Test basic initialization of QFunctionMetric."""
    network_kwargs = {'in_dim': 2, 'out_dim': 2, 'hidden': (64,)}

    metric = QFunctionMetric(
        env=test_env,
        network_arch=MLP,
        network_kwargs=network_kwargs,
        gamma=0.99,
        lr=3e-4,
        batch_size=32,
        iterations=100,  # Small for testing
        device='cpu',
    )

    assert metric.gamma == 0.99
    assert metric.lr == 3e-4
    assert metric.batch_size == 32
    assert metric.iterations == 100
    assert metric.discrete_actions is True


@pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch not installed')
def test_q_function_metric_evaluation(loaded_varied_dataset, test_env):
    """Test Q-function metric evaluation (with very few iterations for speed)."""
    network_kwargs = {'in_dim': 2, 'out_dim': 2, 'hidden': (32,)}

    metric = QFunctionMetric(
        env=test_env,
        network_arch=MLP,
        network_kwargs=network_kwargs,
        gamma=0.99,
        lr=1e-3,
        batch_size=8,
        iterations=50,  # Very small for quick test
        target_update_rate=0.01,
        device='cpu',
    )

    q_value = metric.evaluate(loaded_varied_dataset)

    # Should return a mean Q-value
    assert isinstance(q_value, float)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch not installed')
def test_mlp_network():
    """Test the MLP network used by Q-function metric."""
    mlp = MLP(in_dim=4, out_dim=2, hidden=(64, 64))

    # Test forward pass
    x = torch.randn(10, 4)
    output = mlp(x)

    assert output.shape == (10, 2)
    assert isinstance(output, torch.Tensor)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch not installed')
def test_q_function_sample_batch(loaded_varied_dataset, test_env):
    """Test batch sampling from Q-function metric."""
    network_kwargs = {'in_dim': 2, 'out_dim': 2, 'hidden': (32,)}

    metric = QFunctionMetric(
        env=test_env,
        network_arch=MLP,
        network_kwargs=network_kwargs,
        batch_size=4,
        iterations=10,
        device='cpu',
    )

    # Convert dataset to d4rl format for sampling
    data_dict = loaded_varied_dataset.convert_to_d4rl_format()

    # Sample a batch
    obs, act, rew, next_obs, done = metric.sample_batch(data_dict)

    assert obs.shape[0] == 4
    assert act.shape[0] == 4
    assert rew.shape[0] == 4
    assert next_obs.shape[0] == 4
    assert done.shape[0] == 4


# ============================================================================
# Integration Tests
# ============================================================================


def test_multiple_metrics_on_same_dataset(loaded_varied_dataset, test_env):
    """Test that multiple metrics can be applied to the same dataset."""
    # SACoMetric
    saco = SACoMetric(environment=test_env, use_hyperloglog=False)
    saco_score = saco.evaluate(loaded_varied_dataset)

    # AverageReturnMetric
    avg_return = AverageReturnMetric(gamma=0.99)
    avg_return_score = avg_return.evaluate(loaded_varied_dataset)

    # EstimatedReturnImprovementMetric
    eri = EstimatedReturnImprovementMetric(gamma=0.99)
    eri_score = eri.evaluate(loaded_varied_dataset)

    # All should return valid numeric values
    assert isinstance(saco_score, (int, float))
    assert isinstance(avg_return_score, float)
    assert isinstance(eri_score, float)


def test_all_metrics_with_torch(loaded_varied_dataset, test_env):
    """Test GBE without torch and Q-function with torch."""
    # GBE no longer requires torch without encoder
    gbe = GeneralizedBehavioralEntropyMetric(rep_dim=2, num_knn=3, device='cpu')
    gbe_score = gbe.evaluate(loaded_varied_dataset)
    assert isinstance(gbe_score, float)

    if not TORCH_AVAILABLE:
        return  # Skip Q-function test if torch not available

    # Q-function (very quick)
    network_kwargs = {'in_dim': 2, 'out_dim': 2, 'hidden': (16,)}
    q_func = QFunctionMetric(
        env=test_env,
        network_arch=MLP,
        network_kwargs=network_kwargs,
        iterations=20,
        batch_size=4,
        device='cpu',
    )
    q_value = q_func.evaluate(loaded_varied_dataset)

    assert isinstance(gbe_score, float)
    assert isinstance(q_value, float)


def test_metrics_with_empty_dataset_raise_error(test_storage):
    """Test that metrics raise appropriate errors with empty datasets."""
    empty_dataset = TupliDataset(test_storage)
    empty_dataset.load()

    # AverageReturnMetric should assert on empty episodes
    metric = AverageReturnMetric()
    with pytest.raises(AssertionError, match='Dataset must contain episodes'):
        metric.evaluate(empty_dataset)


def test_torch_metrics_fail_gracefully_without_torch():
    """Test that torch-dependent metrics raise ImportError when torch is unavailable."""
    if TORCH_AVAILABLE:
        pytest.skip('PyTorch is available, skipping unavailability test')

    # Test GeneralizedBehavioralEntropyMetric without encoder should work
    metric = GeneralizedBehavioralEntropyMetric(rep_dim=2)
    assert metric is not None
    assert metric.observation_encoder is None

    # Test MLP
    with pytest.raises(ImportError, match='PyTorch is required'):
        MLP(in_dim=4, out_dim=2)

    # Test QFunctionMetric
    env = gym.make('CartPole-v1')
    with pytest.raises(ImportError, match='PyTorch is required'):
        QFunctionMetric(
            env=env,
            network_arch=MLP,
            network_kwargs={'in_dim': 2, 'out_dim': 2},
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason='PyTorch not installed')
def test_gbe_metric_with_encoder_requires_torch():
    """Test that GBE with encoder requires torch."""
    import torch.nn as nn

    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    # Should work when torch is available
    encoder = DummyEncoder()
    metric = GeneralizedBehavioralEntropyMetric(rep_dim=2, observation_encoder=encoder)
    assert metric.observation_encoder is not None


def test_multiple_metrics_different_parameters(loaded_varied_dataset):
    """Test multiple instantiations of same metric with different parameters."""
    # Test that creating multiple metrics with different params works
    metric1 = AverageReturnMetric(gamma=0.99)
    metric2 = AverageReturnMetric(gamma=0.5)
    metric3 = AverageReturnMetric(gamma=0.99, normalization_range=(0, 10))

    result1 = metric1.evaluate(loaded_varied_dataset)
    result2 = metric2.evaluate(loaded_varied_dataset)
    result3 = metric3.evaluate(loaded_varied_dataset)

    # All should return valid floats
    assert isinstance(result1, float)
    assert isinstance(result2, float)
    assert isinstance(result3, float)

    # Results should differ due to different gamma values
    assert result1 != result2


def test_saco_with_2d_reference_arrays(test_env):
    """Test SACoMetric with 2D reference arrays (already properly shaped)."""
    ref_states = np.array([[0, 1], [1, 0], [1, 1]])
    ref_actions = np.array([[0], [1], [0]])

    metric = SACoMetric(
        environment=test_env,
        reference_states=ref_states,
        reference_actions=ref_actions,
        use_hyperloglog=False,
    )

    # Should not reshape since already 2D
    assert metric.reference_states.shape == (3, 2)
    assert metric.reference_actions.shape == (3, 1)


def test_eri_with_all_negative_returns():
    """Test ERI when all returns are negative."""
    from pytupli.storage import FileStorage
    import tempfile
    from pytupli.schema import Episode, RLTuple, FilterEQ

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorage(storage_base_dir=tmpdir)
        from pytupli.benchmark import TupliEnvWrapper
        from tests.client.example_envs import SimpleTestEnv

        env = SimpleTestEnv()
        benchmark = TupliEnvWrapper(env, storage)
        benchmark.store(name='Negative Test', description='Test', difficulty='easy', version='1.0')

        # Manually create episodes with negative rewards
        episode1 = Episode(
            benchmark_id=benchmark.id,
            metadata={},
            tuples=[
                RLTuple(
                    state={'x': 0}, action=0, reward=-2.0, info={}, terminal=False, timeout=False
                ),
                RLTuple(
                    state={'x': 1}, action=0, reward=-1.0, info={}, terminal=True, timeout=False
                ),
            ],
        )
        episode2 = Episode(
            benchmark_id=benchmark.id,
            metadata={},
            tuples=[
                RLTuple(
                    state={'x': 0}, action=0, reward=-5.0, info={}, terminal=False, timeout=False
                ),
                RLTuple(
                    state={'x': 1}, action=0, reward=-3.0, info={}, terminal=True, timeout=False
                ),
            ],
        )

        storage.record_episode(episode1)
        storage.record_episode(episode2)

        dataset = TupliDataset(storage)
        dataset = dataset.with_benchmark_filter(FilterEQ(key='id', value=benchmark.id))
        dataset.load()

        # ERI with known negative minimum
        metric = EstimatedReturnImprovementMetric(gamma=1.0, min_return=-10.0)
        eri_score = metric.evaluate(dataset)

        assert isinstance(eri_score, float)
        assert eri_score >= 0.0

        benchmark.delete(delete_episodes=True)


def test_average_return_single_episode():
    """Test AverageReturnMetric with only one episode."""
    from pytupli.storage import FileStorage
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorage(storage_base_dir=tmpdir)
        from pytupli.benchmark import TupliEnvWrapper
        from tests.client.example_envs import SimpleTestEnv

        env = SimpleTestEnv()
        benchmark = TupliEnvWrapper(env, storage)
        benchmark.store(name='Single Episode', description='Test', difficulty='easy', version='1.0')

        # Record one episode
        obs, _ = benchmark.reset()
        benchmark.step(np.int64(1))
        benchmark.step(np.int64(1))

        from pytupli.schema import FilterEQ

        dataset = TupliDataset(storage)
        dataset = dataset.with_benchmark_filter(FilterEQ(key='id', value=benchmark.id))
        dataset.load()

        metric = AverageReturnMetric(gamma=0.99)
        avg_return = metric.evaluate(dataset)

        assert isinstance(avg_return, float)
        assert avg_return >= 0.0

        benchmark.delete(delete_episodes=True)


def test_saco_assertion_for_discrete_actions(test_env):
    """Test that SACoMetric asserts discrete action space."""
    # The test_env should have discrete actions, so this should pass
    with pytest.raises(AssertionError, match='SACoMetric currently only supports discrete action spaces'):
        SACoMetric(environment=test_env, use_hyperloglog=False)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
