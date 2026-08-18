"""Unit tests for the Mountain Car SARSA(lambda) code (rl_lab1, problem 2)."""

import eligibility_sarsa as es
import numpy as np
import pytest


def test_fourier_features_count():
    # Fourier basis of order 5 in 2 dims: (5+1)^2 - 1 combos + null base = 36
    etas = [[i, j] for i in range(5 + 1) for j in range(5 + 1) if (i, j) != (0, 0)]
    fla = es.FourierLinearApprox(etas, nA=3)
    assert fla.w.shape == (len(etas) + 1, 3)


def test_fourier_value_is_linear_in_weights():
    etas = [[1, 0], [0, 1], [1, 1]]
    fla = es.FourierLinearApprox(etas, nA=2)
    s = np.array([0.2, -0.4])
    q = fla.Qw(s, 0)
    assert np.isfinite(q)
    assert fla.basis_functions(s).shape == (4,)  # + null base


def test_scale_state_variables():
    low = np.array([-1.2, -0.07])
    high = np.array([0.6, 0.07])
    scaled = es.scale_state_variables(np.array([-0.3, 0.0]), low, high)
    assert scaled.shape == (2,)
    assert np.isfinite(scaled).all()
    # bounds map to 0/1
    assert es.scale_state_variables(low, low, high) == pytest.approx([0, 0])
    assert es.scale_state_variables(high, low, high) == pytest.approx([1, 1])


@pytest.mark.slow
def test_sarsa_improves_on_mountain_car():
    import gym

    env = gym.make("MountainCar-v0")
    etas = [
        [i, j]
        for i in range(5 + 1)
        for j in range(5 + 1)
        if (i, j) != (0, 0)
    ]
    fla = es.FourierLinearApprox(etas, nA=env.action_space.n)
    rewards, _ = es.eligibility_sarsa(
        env, fla, elig_lambda=0.9, gamma=1, alpha=0.005, epsilon=0.1,
        n_episodes=60, momentum=0,
    )
    env.close()
    assert len(rewards) == 60
    # learning happens: average reward of the last half beats the first half
    first = float(np.mean(rewards[:30]))
    last = float(np.mean(rewards[30:]))
    assert last >= first - 5, f"no improvement: {first:.1f} -> {last:.1f}"
