"""Unit tests for the Minotaur Maze environment and algorithms (rl_lab1, problem 1)."""

import minoutaur_maze as mm
import numpy as np
import pytest

MAZE = np.array(
    [
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0],
    ]
)

THOMAS_ST = (0, 0)
MINOTAUR_ST = (6, 5)


@pytest.fixture(scope="module")
def env():
    return mm.MinotaurMaze(MAZE, minotaur_can_wait=False, key_needed=False)


def test_state_space(env):
    thomas_cells = int((MAZE != 1).sum())  # 40
    all_cells = MAZE.size  # 56 (the minotaur may also sit on wall cells)
    assert env.n_states == thomas_cells * all_cells
    assert env.n_actions == 5
    # no key needed -> key flag always 1
    assert all(s[4] == 1 for s in env.states.values())


def test_transitions_are_probabilities(env):
    for s in range(env.n_states):
        (_i_t, _j_t, *_rest) = env.states[s]
        for a in env.actions:
            # Impossible actions (walls) have -inf reward and empty rows
            if env.rewards[s, a] == -np.inf:
                continue
            if env.subset[s] in (-1, 1):
                # terminal states stay put with probability 1
                assert env.transition_probabilities[s, s, a] == 1
            else:
                total = env.transition_probabilities[:, s, a].sum()
                assert total == pytest.approx(1.0, abs=1e-9)


def test_rewards(env):
    for s in range(env.n_states):
        for a in env.actions:
            r = env.rewards[s, a]
            assert r in (-np.inf, -1000, -1, 0, 500), f"unexpected reward {r} at {s},{a}"


def test_value_iteration_converges(env):
    V, policy = mm.value_iteration(env, gamma=0.99, epsilon=1e-3)
    assert V.shape == (env.n_states,)
    assert policy.shape == (env.n_states,)
    # Absorbing terminal states have V = 0 by definition of the self-loop
    for s in range(env.n_states):
        if env.subset[s] in (-1, 1):
            assert V[s] == pytest.approx(0.0, abs=1e-6)
    # Non-terminal states near victory carry the highest value (~500)
    assert V.max() > 400
    assert V.max() <= 500 + 1e-6


def test_vi_policy_reaches_exit(env):
    _V, policy = mm.value_iteration(env, gamma=0.99, epsilon=1e-3)
    # Simulate a few rollouts with the greedy policy: some must reach the exit
    reached = 0
    for _ in range(25):
        s = env.mapping[THOMAS_ST + MINOTAUR_ST + (1,)]
        for _ in range(200):
            if env.subset[s] == 1:
                reached += 1
                break
            if env.subset[s] == -1:
                break
            a = int(policy[s])
            s = env.move(s, a)
    assert reached > 0, "greedy VI policy should win at least some rollouts"


def test_qlearning_runs_and_updates_qtable(env):
    start = THOMAS_ST + MINOTAUR_ST + (1,)
    Q, policy, init_Vs = mm.qLearning(
        env, start, gamma=0.99, epsilon=0.5, n_episodes=50, max_iters=80,
        decay_delta=0.9,
    )
    assert Q.shape == (env.n_states, env.n_actions)
    assert len(policy) == env.n_states
    assert len(init_Vs) == 50
    assert np.isfinite(Q).all()
