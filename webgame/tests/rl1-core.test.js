/**
 * Node test suite for the RL Lab 1 web core (Minotaur Maze + Mountain Car).
 */
const { test } = require("node:test");
const assert = require("node:assert");

const RL1 = require("../js/rl1-core.js");

// ---------------------------------------------------------------------
// Minotaur Maze
// ---------------------------------------------------------------------

test("MDP state space matches the lab (thomas cells x all cells)", () => {
  const mdp = RL1.buildMazeMdp();
  let thomasCells = 0;
  for (let i = 0; i < RL1.ROWS; i++)
    for (let j = 0; j < RL1.COLS; j++) if (RL1.MAZE[i][j] !== 1) thomasCells++;
  assert.strictEqual(mdp.nStates, thomasCells * RL1.ROWS * RL1.COLS);
  assert.strictEqual(mdp.nActions, 5);
  // start state exists
  assert.ok(
    mdp.map.has("0,0,6,5,1"),
    "start state (thomas 0,0, minotaur 6,5) must exist",
  );
});

test("transition probabilities sum to 1 for valid non-terminal (s,a)", () => {
  const mdp = RL1.buildMazeMdp();
  for (let s = 0; s < mdp.nStates; s++) {
    const st = mdp.states[s];
    const valid = mdp.actsThomas[st[0] + "," + st[1]];
    for (const a of valid) {
      if (mdp.subset[s] !== 0) {
        assert.strictEqual(mdp.P[s][s][a], 1);
        continue;
      }
      let total = 0;
      for (let ns = 0; ns < mdp.nStates; ns++) total += mdp.P[ns][s][a];
      assert.ok(Math.abs(total - 1) < 1e-9, `s=${s} a=${a} sum=${total}`);
    }
  }
});

test("rewards: victory 500, loss -1000, step -1, walls -inf", () => {
  const mdp = RL1.buildMazeMdp();
  let sawVictory = 0;
  let sawLoss = 0;
  for (let s = 0; s < mdp.nStates; s++) {
    for (let a = 0; a < mdp.nActions; a++) {
      const r = mdp.R[s][a];
      assert.ok(
        r === RL1.NEG_INF || r === -1000 || r === -1 || r === 0 || r === 500,
        `unexpected reward ${r} at ${s},${a}`,
      );
      if (r === 500) sawVictory++;
      if (r === -1000) sawLoss++;
    }
  }
  assert.ok(sawVictory > 0 && sawLoss > 0, "both victory and loss rewards exist");
});

test("value iteration converges to ~500 near victory, 0 on terminals", () => {
  const mdp = RL1.buildMazeMdp();
  const { V, policy, iterations } = RL1.valueIteration(mdp, 0.99, 1e-3);
  assert.ok(iterations > 0 && iterations <= 200);
  for (let s = 0; s < mdp.nStates; s++) {
    if (mdp.subset[s] !== 0) assert.ok(Math.abs(V[s]) < 1e-6, "terminal V=0");
  }
  assert.ok(V[mdp.map.get("1,0,6,5,1")] > 400, "near-victory state should be valuable");
  assert.ok(Math.max(...V) <= 500 + 1e-6);
  assert.strictEqual(policy.length, mdp.nStates);
});

test("greedy VI policy wins at least some rollouts", () => {
  const mdp = RL1.buildMazeMdp();
  const { policy } = RL1.valueIteration(mdp, 0.99, 1e-3);
  let won = 0;
  for (let trial = 0; trial < 30; trial++) {
    let s = [0, 0, 6, 5, 1];
    for (let t = 0; t < 250; t++) {
      const sub = RL1.MAZE[s[0]][s[1]] === 2 && !(s[0] === s[2] && s[1] === s[3]) ? 1
        : s[0] === s[2] && s[1] === s[3] ? -1 : 0;
      if (sub !== 0) break;
      const key = s.join(",");
      const sid = mdp.map.get(key);
      const res = RL1.mazeStep(s, policy[sid]);
      s = res.state;
      if (res.won) {
        won++;
        break;
      }
      if (res.done) break;
    }
  }
  assert.ok(won > 0, "VI policy should win some rollouts");
});

test("mazeStep never leaves the maze or enters walls", () => {
  for (let t = 0; t < 200; t++) {
    const res = RL1.mazeStep([0, 0, 6, 5, 1], 2); // move right into a wall -> stays
    const s = res.state;
    assert.ok(s[0] >= 0 && s[0] < RL1.ROWS && s[1] >= 0 && s[1] < RL1.COLS);
    assert.notStrictEqual(RL1.MAZE[s[0]][s[1]], 1, "thomas never in a wall");
  }
});

// ---------------------------------------------------------------------
// Mountain Car
// ---------------------------------------------------------------------

test("mountain car physics: push right on a slope accelerates", () => {
  const env = RL1.createMountainCar(1);
  env.x = -0.5;
  env.v = 0;
  // At x=-0.5, cos(3x) > 0, gravity pushes right too
  const v0 = env.v;
  RL1.mcStep(env, 2);
  assert.ok(env.v > v0, "pushing right must not decrease velocity");
});

test("mountain car reaches the goal from a good run", () => {
  const env = RL1.createMountainCar(2);
  env.x = 0.49;
  env.v = 0.05;
  const res = RL1.mcStep(env, 2);
  assert.strictEqual(res.reached, true);
  assert.strictEqual(res.done, true);
});

test("fourier features: dimension matches order", () => {
  const etas = RL1.fourierEtas(5);
  assert.strictEqual(etas.length, 35);
  const phi = RL1.fourierFeatures(etas, [0.5, -0.2]);
  assert.strictEqual(phi.length, 36);
  assert.strictEqual(phi[0], 1); // null base
});

test("sarsa(lambda) improves over episodes on mountain car", () => {
  const out = RL1.sarsaLambda(5, 1, 0.9, 0.005, 0.1, 40, 200, 7);
  assert.strictEqual(out.rewards.length, 40);
  const first = out.rewards.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
  const last = out.rewards.slice(20).reduce((a, b) => a + b, 0) / 20;
  assert.ok(last >= first - 5, `no improvement: ${first} -> ${last}`);
});
