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

  /** Minotaur movement rule — the lab lets you pick how the minotaur moves:
   *  "random" (uniform over the valid moves, the lab default), "chase"
   *  (greedy — always takes a step toward Thomas), "static" (never moves).
   *  The MDP transitions, the rewards and the play dynamics ALL use it, so
   *  the VI's win probability reflects the rule. */
  var minoRule = "random";
  var minoCanWait = false; // lab's minotaur_can_wait — tested in the notebook

  function minoMoveProbs(mi, mj, ti, tj, V, map) {
    var acts = validMoves(MAZE, [mi, mj], false);
    if (minoCanWait && acts.indexOf(0) === -1) acts = acts.concat([0]);
    if (minoRule === "static" || acts.length === 0) {
      return { actions: [0], probs: [1] }; // stays put (action 0 = no move)
    }
    if (minoRule === "smart") {
      // adversarial: may also HOLD POSITION (a real adversary guards the
      // exit); with values it picks the LOWEST-V option deterministically,
      // without values (MDP construction) it spreads uniformly over them
      var opts = acts.concat([0]); // 0 = stay
      if (V && map) {
        var worst = opts[0],
          wv = Infinity;
        for (var ai = 0; ai < opts.length; ai++) {
          var nmi = mi + ACTIONS[opts[ai]][0],
            nmj = mj + ACTIONS[opts[ai]][1];
          var sid = map.get(ti + "," + tj + "," + nmi + "," + nmj + ",1");
          var v = sid === undefined ? 0 : V[sid];
          if (v < wv) {
            wv = v;
            worst = opts[ai];
          }
        }
        return { actions: [worst], probs: [1] };
      }
      var pu = 1 / opts.length;
      return {
        actions: opts,
        probs: opts.map(function () {
          return pu;
        }),
      };
    }
    if (minoRule === "chase") {
      var ds = acts.map(function (a) {
        return (
          Math.abs(mi + ACTIONS[a][0] - ti) + Math.abs(mj + ACTIONS[a][1] - tj)
        );
      });
      var dmin = Math.min.apply(null, ds);
      var p =
        1 /
        ds.filter(function (d) {
          return d === dmin;
        }).length;
      return {
        actions: acts,
        probs: acts.map(function (_, k) {
          return ds[k] === dmin ? p : 0;
        }),
      };
    }
    var pu = 1 / acts.length;
    return {
      actions: acts,
      probs: acts.map(function () {
        return pu;
      }),
    };
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
        if (MAZE[tc[0]][tc[1]] === 2 && !(tc[0] === mc[0] && tc[1] === mc[1]))
          sub = 1;
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
      var ti = st[0],
        tj = st[1],
        mi = st[2],
        mj = st[3];
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
        R[si][a] = REW.step;
        var mm = minoMoveProbs(mi, mj, nti, ntj);
        for (var ma = 0; ma < mm.actions.length; ma++) {
          if (mm.probs[ma] <= 0) continue;
          var nmi = mi + ACTIONS[mm.actions[ma]][0];
          var nmj = mj + ACTIONS[mm.actions[ma]][1];
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
        var mm = minoMoveProbs(mi, mj, nti2, ntj2);
        for (var ma2 = 0; ma2 < mm.actions.length; ma2++) {
          if (mm.probs[ma2] <= 0) continue;
          var nmi2 = mi + ACTIONS[mm.actions[ma2]][0];
          var nmj2 = mj + ACTIONS[mm.actions[ma2]][1];
          var ns2 = map.get(nti2 + "," + ntj2 + "," + nmi2 + "," + nmj2 + ",1");
          P[ns2][si][a2] += mm.probs[ma2];
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

    var smart = minoRule === "smart";
    function bellman() {
      for (var s = 0; s < nStates; s++) {
        for (var a = 0; a < nActions; a++) {
          var acc = mdp.R[s][a];
          if (acc === NEG_INF) {
            Q[s][a] = acc;
            continue;
          }
          if (smart) {
            // adversarial minotaur: it picks the WORST reachable next state
            var worst = Infinity;
            for (var ns = 0; ns < nStates; ns++) {
              var p = mdp.P[ns][s][a];
              if (p !== 0 && V[ns] < worst) worst = V[ns];
            }
            Q[s][a] = acc + gamma * (worst === Infinity ? 0 : worst);
          } else {
            var dot = 0;
            for (var ns2 = 0; ns2 < nStates; ns2++) {
              var p2 = mdp.P[ns2][s][a];
              if (p2 !== 0) dot += p2 * V[ns2];
            }
            Q[s][a] = acc + gamma * dot;
          }
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
  function mazeStep(state, a, V, map) {
    var ti = state[0],
      tj = state[1],
      mi = state[2],
      mj = state[3];
    var sub = 0;
    if (MAZE[ti][tj] === 2 && !(ti === mi && tj === mj)) sub = 1;
    else if (ti === mi && tj === mj) sub = -1;
    if (sub !== 0) return { state: state, done: true, won: sub === 1 };

    var thomasActs = validMoves(MAZE, [ti, tj], true);
    var nti = ti,
      ntj = tj;
    if (thomasActs.indexOf(a) !== -1) {
      nti = ti + ACTIONS[a][0];
      ntj = tj + ACTIONS[a][1];
    }
    var mm = minoMoveProbs(mi, mj, nti, ntj);
    var roll = Math.random(),
      acc = 0,
      ma = mm.actions[mm.actions.length - 1];
    for (var k = 0; k < mm.actions.length; k++) {
      acc += mm.probs[k];
      if (roll <= acc) {
        ma = mm.actions[k];
        break;
      }
    }
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
  function sarsaLambda(
    order,
    gamma,
    lambda,
    alpha,
    epsilon,
    nEpisodes,
    maxIters,
    seed,
  ) {
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
      var s = [
        (env.x - low[0]) / (high[0] - low[0]),
        (env.v - low[1]) / (high[1] - low[1]),
      ];
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
    setMinoRule: function (r) {
      minoRule = r;
    },
    setMinoCanWait: function (v) {
      minoCanWait = !!v;
    },
    getMinoCanWait: function () {
      return minoCanWait;
    },
    getMinoRule: function () {
      return minoRule;
    },
    mountainCarTrack: mountainCarTrack,
    createMountainCar: createMountainCar,
    mcStep: mcStep,
    fourierFeatures: fourierFeatures,
    fourierEtas: fourierEtas,
    sarsaLambda: sarsaLambda,
  };
});
