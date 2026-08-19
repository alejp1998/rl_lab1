/**
 * RL Lab 1 web core — Minotaur Maze + Mountain Car, ported from the lab code.
 * Pure JS (no DOM/Pixi): works in the browser and under node:test.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.RL1 = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // =====================================================================
  // MINOTAUR MAZE (faithful port of minoutaur_maze.py)
  // =====================================================================

  // Actions: 0 wait, 1 left, 2 right, 3 up, 4 down
  var ACTIONS = [
    [0, 0],
    [0, -1],
    [0, 1],
    [-1, 0],
    [1, 0],
  ];
  var ACT_NAMES = ["wait", "left", "right", "up", "down"];

  var MAZE = [
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0],
  ];
  var ROWS = MAZE.length;
  var COLS = MAZE[0].length;
  var THOMAS_ST = [0, 0];
  var MINOTAUR_ST = [6, 5];

  var REW = { step: -1, key: 100, victory: 500, loss: -1000 };
  var NEG_INF = -1e9;

  function validMoves(maze, cell, canWait) {
    var moves = [];
    for (var a = 0; a < ACTIONS.length; a++) {
      if (!canWait && a === 0) continue;
      var ni = cell[0] + ACTIONS[a][0];
      var nj = cell[1] + ACTIONS[a][1];
      if (ni < 0 || ni >= ROWS || nj < 0 || nj >= COLS) continue;
      if (maze[ni][nj] === 1) continue;
      moves.push(a);
    }
    return moves;
  }

  /**
   * Builds the full joint MDP: states (thomas, minotaur, key), transitions,
   * rewards — mirroring the lab's MinotaurMaze (key_needed=false so key=1).
   * Returns { states, map, subset, actsThomas, actsMino, P, R, nStates }.
   */
  function buildMazeMdp() {
    var thomasCells = [];
    for (var i = 0; i < ROWS; i++)
      for (var j = 0; j < COLS; j++)
        if (MAZE[i][j] !== 1) thomasCells.push([i, j]);

    var allCells = [];
    for (var i2 = 0; i2 < ROWS; i2++)
      for (var j2 = 0; j2 < COLS; j2++) allCells.push([i2, j2]);

    var states = [];
    var map = new Map();
    var subset = [];
    var actsThomas = {}; // key "i,j" -> valid actions
    var actsMino = {};

    // precompute valid actions per cell (Thomas: no walls; minotaur: no wait)
    thomasCells.forEach(function (c) {
      actsThomas[c[0] + "," + c[1]] = validMoves(MAZE, c, true);
    });
    allCells.forEach(function (c) {
      actsMino[c[0] + "," + c[1]] = validMoves(MAZE, c, false);
    });

    var s = 0;
    for (var it = 0; it < thomasCells.length; it++) {
      var tc = thomasCells[it];
      for (var im = 0; im < allCells.length; im++) {
        var mc = allCells[im];
        var key = 1;
        var sub;
        if (MAZE[tc[0]][tc[1]] === 2 && !(tc[0] === mc[0] && tc[1] === mc[1])) sub = 1;
        else if (tc[0] === mc[0] && tc[1] === mc[1]) sub = -1;
        else sub = 0;
        states.push([tc[0], tc[1], mc[0], mc[1], key]);
        map.set(tc[0] + "," + tc[1] + "," + mc[0] + "," + mc[1] + "," + key, s);
        subset.push(sub);
        s++;
      }
    }

    var nStates = states.length;
    var nActions = ACTIONS.length;
    // P[next][s][a] like the lab tensor
    var P = new Array(nStates);
    for (var i3 = 0; i3 < nStates; i3++) {
      P[i3] = new Array(nStates);
      for (var j3 = 0; j3 < nStates; j3++) {
        P[i3][j3] = new Array(nActions).fill(0);
      }
    }
    var R = new Array(nStates);
    for (var i4 = 0; i4 < nStates; i4++) R[i4] = new Array(nActions).fill(0);

    for (var si = 0; si < nStates; si++) {
      var st = states[si];
      var ti = st[0], tj = st[1], mi = st[2], mj = st[3];
      var terminal = subset[si] === -1 || subset[si] === 1;
      var thomasActs = actsThomas[ti + "," + tj];

      // rewards
      for (var a = 0; a < nActions; a++) {
        if (thomasActs.indexOf(a) === -1) {
          R[si][a] = NEG_INF;
          continue;
        }
        if (terminal) {
          R[si][a] = 0;
          continue;
        }
        var nti = ti + ACTIONS[a][0];
        var ntj = tj + ACTIONS[a][1];
        var minoActs = actsMino[mi + "," + mj];
        R[si][a] = REW.step;
        for (var ma = 0; ma < minoActs.length; ma++) {
          var nmi = mi + ACTIONS[minoActs[ma]][0];
          var nmj = mj + ACTIONS[minoActs[ma]][1];
          var ns = map.get(nti + "," + ntj + "," + nmi + "," + nmj + ",1");
          if (subset[ns] === -1) R[si][a] = REW.loss;
          else if (subset[ns] === 1) R[si][a] = REW.victory;
        }
      }

      // transitions
      for (var a2 = 0; a2 < nActions; a2++) {
        if (terminal) {
          P[si][si][a2] = 1;
          continue;
        }
        if (thomasActs.indexOf(a2) === -1) continue;
        var nti2 = ti + ACTIONS[a2][0];
        var ntj2 = tj + ACTIONS[a2][1];
        var minoActs2 = actsMino[mi + "," + mj];
        var pv = 1 / minoActs2.length;
        for (var ma2 = 0; ma2 < minoActs2.length; ma2++) {
          var nmi2 = mi + ACTIONS[minoActs2[ma2]][0];
          var nmj2 = mj + ACTIONS[minoActs2[ma2]][1];
          var ns2 = map.get(nti2 + "," + ntj2 + "," + nmi2 + "," + nmj2 + ",1");
          P[ns2][si][a2] = pv;
        }
      }
    }

    return {
      states: states,
      map: map,
      subset: subset,
      actsThomas: actsThomas,
      actsMino: actsMino,
      P: P,
      R: R,
      nStates: nStates,
      nActions: nActions,
    };
  }

  /** Value iteration (lab's algorithm). Returns { V, policy, sweeps }. */
  function valueIteration(mdp, gamma, epsilon) {
    var nStates = mdp.nStates;
    var nActions = mdp.nActions;
    var V = new Array(nStates).fill(0);
    var Q = new Array(nStates);
    for (var s = 0; s < nStates; s++) Q[s] = new Array(nActions).fill(0);
    var BV = new Array(nStates).fill(0);
    var tol = ((1 - gamma) * epsilon) / gamma;
    var n = 0;
    var sweeps = [];

    function bellman() {
      for (var s = 0; s < nStates; s++) {
        for (var a = 0; a < nActions; a++) {
          var acc = mdp.R[s][a];
          if (acc === NEG_INF) {
            Q[s][a] = acc;
            continue;
          }
          var dot = 0;
          for (var ns = 0; ns < nStates; ns++) {
            var p = mdp.P[ns][s][a];
            if (p !== 0) dot += p * V[ns];
          }
          Q[s][a] = acc + gamma * dot;
        }
        BV[s] = Math.max.apply(null, Q[s]);
      }
    }

    bellman();
    var err = norm(V, BV);
    sweeps.push(BV.slice());
    while (err >= tol && n < 200) {
      n++;
      for (var s2 = 0; s2 < nStates; s2++) V[s2] = BV[s2];
      bellman();
      err = norm(V, BV);
      sweeps.push(BV.slice());
    }
    var policy = new Array(nStates);
    for (var s3 = 0; s3 < nStates; s3++) policy[s3] = argmax(Q[s3]);
    return { V: BV, policy: policy, sweeps: sweeps, iterations: n };
  }

  function norm(a, b) {
    var acc = 0;
    for (var i = 0; i < a.length; i++) {
      var d = a[i] - b[i];
      acc += d * d;
    }
    return Math.sqrt(acc);
  }

  function argmax(arr) {
    var best = 0;
    for (var i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return best;
  }

  /** Stochastic step: Thomas takes `a`, minotaur moves uniformly at random. */
  function mazeStep(state, a) {
    var ti = state[0], tj = state[1], mi = state[2], mj = state[3];
    var sub = 0;
    if (MAZE[ti][tj] === 2 && !(ti === mi && tj === mj)) sub = 1;
    else if (ti === mi && tj === mj) sub = -1;
    if (sub !== 0) return { state: state, done: true, won: sub === 1 };

    var thomasActs = validMoves(MAZE, [ti, tj], true);
    var nti = ti, ntj = tj;
    if (thomasActs.indexOf(a) !== -1) {
      nti = ti + ACTIONS[a][0];
      ntj = tj + ACTIONS[a][1];
    }
    var minoActs = validMoves(MAZE, [mi, mj], false);
    var ma = minoActs[Math.floor(Math.random() * minoActs.length)];
    var nmi = mi + ACTIONS[ma][0];
    var nmj = mj + ACTIONS[ma][1];

    var won = MAZE[nti][ntj] === 2 && !(nti === nmi && ntj === nmj);
    var lost = nti === nmi && ntj === nmj;
    return {
      state: [nti, ntj, nmi, nmj, 1],
      done: won || lost,
      won: won,
    };
  }

  // =====================================================================
  // MOUNTAIN CAR (SARSA(lambda) with Fourier features — lab problem 2)
  // =====================================================================

  function mountainCarTrack(x) {
    // the TRUE hill potential: the dynamics use -0.0025*cos(3x) = -dV/dx,
    // so V(x) = sin(3x) — the drawn ramp matches the physics exactly
    return Math.sin(3 * x);
  }

  function createMountainCar(seed) {
    var rng = mulberry32(seed || 1);
    return {
      x: -0.5,
      v: 0,
      rng: rng,
      steps: 0,
    };
  }

  function mcStep(env, action) {
    // action: 0 push left, 1 no-op, 2 push right — gym MountainCar-v0 exact:
    //   x += v ; v += 0.0015*a - 0.0025*cos(3x) ; clip v ; goal at x >= 0.5
    var a = action - 1;
    env.x += env.v;
    env.v += a * 0.0015 - 0.0025 * Math.cos(3 * env.x);
    if (env.v > 0.07) env.v = 0.07;
    if (env.v < -0.07) env.v = -0.07;
    if (env.x < -1.2) {
      env.x = -1.2;
      env.v = 0;
    }
    if (env.x >= 0.5) {
      env.x = 0.5;
      return { done: true, reward: 0, reached: true };
    }
    env.steps++;
    return { done: false, reward: -1, reached: false };
  }

  /** Fourier features of order k in 2 dims (+null base). */
  function fourierFeatures(etas, s) {
    var out = [1];
    for (var i = 0; i < etas.length; i++) {
      out.push(Math.cos(Math.PI * (etas[i][0] * s[0] + etas[i][1] * s[1])));
    }
    return out;
  }

  function fourierEtas(order) {
    var etas = [];
    for (var i = 0; i <= order; i++)
      for (var j = 0; j <= order; j++)
        if (!(i === 0 && j === 0)) etas.push([i, j]);
    return etas;
  }

  /**
   * SARSA(lambda) with Fourier linear approximation (lab's algorithm).
   * Returns { rewards, w } — rewards per episode.
   */
  function sarsaLambda(order, gamma, lambda, alpha, epsilon, nEpisodes, maxIters, seed) {
    var etas = fourierEtas(order);
    var nA = 3;
    var w = new Array(etas.length + 1);
    for (var i = 0; i < w.length; i++) w[i] = new Array(nA).fill(0);
    var rewards = [];

    function Qw(s, a) {
      var phi = fourierFeatures(etas, s);
      var acc = 0;
      for (var k = 0; k < phi.length; k++) acc += w[k][a] * phi[k];
      return acc;
    }

    function choose(s, eps) {
      if (Math.random() < eps) return Math.floor(Math.random() * nA);
      var best = 0;
      for (var a = 1; a < nA; a++) if (Qw(s, a) > Qw(s, best)) best = a;
      return best;
    }

    var low = [-1.2, -0.07];
    var high = [0.6, 0.07];

    for (var e = 0; e < nEpisodes; e++) {
      var env = createMountainCar(seed + e);
      var s = [(env.x - low[0]) / (high[0] - low[0]), (env.v - low[1]) / (high[1] - low[1])];
      var a = choose(s, epsilon);
      var z = new Array(etas.length + 1);
      for (var zi = 0; zi < z.length; zi++) z[zi] = new Array(nA).fill(0);
      var episodeReward = 0;
      var steps = 0;
      var done = false;
      while (!done && steps < maxIters) {
        steps++;
        var res = mcStep(env, a);
        episodeReward += res.reward;
        var ns = [
          (env.x - low[0]) / (high[0] - low[0]),
          (env.v - low[1]) / (high[1] - low[1]),
        ];
        var na = res.done ? 0 : choose(ns, epsilon);
        var phi = fourierFeatures(etas, s);
        var delta = res.reward + (res.done ? 0 : gamma * Qw(ns, na)) - Qw(s, a);
        for (var k = 0; k < phi.length; k++) {
          z[k][a] = gamma * lambda * z[k][a] + phi[k];
          w[k][a] += alpha * delta * z[k][a];
        }
        s = ns;
        a = na;
        done = res.done;
      }
      rewards.push(episodeReward);
    }
    return { rewards: rewards, w: w };
  }

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  return {
    MAZE: MAZE,
    ROWS: ROWS,
    COLS: COLS,
    THOMAS_ST: THOMAS_ST,
    MINOTAUR_ST: MINOTAUR_ST,
    ACTIONS: ACTIONS,
    ACT_NAMES: ACT_NAMES,
    REW: REW,
    NEG_INF: NEG_INF,
    validMoves: validMoves,
    buildMazeMdp: buildMazeMdp,
    valueIteration: valueIteration,
    mazeStep: mazeStep,
    mountainCarTrack: mountainCarTrack,
    createMountainCar: createMountainCar,
    mcStep: mcStep,
    fourierFeatures: fourierFeatures,
    fourierEtas: fourierEtas,
    sarsaLambda: sarsaLambda,
  };
});
