/**
 * RL Lab 1 — interactive cockpit: Minotaur Maze (play + value iteration)
 * and Mountain Car (live SARSA(lambda) learning + manual driving).
 */
(function () {
  "use strict";

  var S = window.RL1;
  var canvas = document.getElementById("view");
  var ctx = canvas.getContext("2d");
  var $id = function (id) {
    return document.getElementById(id);
  };

  // ---------------------------------------------------------------- theme
  var PAL = {
    dark: {
      bg: "#0b0f19",
      panel: "#111c30",
      grid: "rgba(148,163,184,0.14)",
      wall: "#0b0f19",
      wallEdge: "#1e293b",
      empty: "#1e293b",
      exit: "#14532d",
      exitEdge: "#4ade80",
      thomas: "#78350f",
      thomasEdge: "#f59e0b",
      mino: "#7f1d1d",
      minoEdge: "#ef4444",
      text: "#e2e8f0",
      muted: "#94a3b8",
      faint: "#64748b",
      heat: ["#0f172a", "#164e63", "#0e7490", "#06b6d4", "#22d3ee", "#fbbf24"],
      track: "#1e293b",
      trackEdge: "#334155",
      car: "#38bdf8",
      carEdge: "#0284c7",
      chart: "#22d3ee",
      chartGrid: "rgba(148,163,184,0.15)",
    },
    light: {
      bg: "#f1f5f9",
      panel: "#ffffff",
      grid: "rgba(51,65,85,0.12)",
      wall: "#0f172a",
      wallEdge: "#0f172a",
      empty: "#ffffff",
      exit: "#95fd99",
      exitEdge: "#22c55e",
      thomas: "#fae0c3",
      thomasEdge: "#d97706",
      mino: "#ffc4cc",
      minoEdge: "#f87171",
      text: "#0f172a",
      muted: "#475569",
      faint: "#64748b",
      heat: ["#f8fafc", "#cffafe", "#67e8f9", "#06b6d4", "#0e7490", "#f59e0b"],
      track: "#e2e8f0",
      trackEdge: "#94a3b8",
      car: "#0284c7",
      carEdge: "#0c4a6e",
      chart: "#0891b2",
      chartGrid: "rgba(51,65,85,0.12)",
    },
  };

  function pal() {
    var t = document.documentElement.getAttribute("data-theme");
    return PAL[t === "dark" ? "dark" : "light"];
  }

  function applyTheme() {
    var t = document.documentElement.getAttribute("data-theme");
  }

  // ---------------------------------------------------------------- state
  var tab = "maze";
  var maze = {
    mode: "play", // play | vi
    state: [0, 0, 6, 5, 1],
    steps: 0,
    wins: 0,
    losses: 0,
    last: "",
    over: false,
    won: false,
    mdp: null,
    vi: null,
    viSweep: 0,
    viTimer: null,
    viGamma: 0.99,
    viSpeed: 12,
  };
  var car = {
    mode: "learn", // learn | drive
    env: S.createMountainCar(1),
    w: null,
    rewards: [],
    episode: 0,
    nEpisodes: 60,
    maxIters: 300,
    learnTimer: null,
    stepTimer: null,
    boosting: false,
    keys: {},
  };

  function log(msg) {
    var box = $id("log");
    var div = document.createElement("div");
    div.textContent = "› " + msg;
    box.appendChild(div);
    while (box.children.length > 60) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
  }

  function setResult(title, sub, cls) {
    var r = $id("hud-result");
    r.className = "hud-result " + cls;
    r.innerHTML =
      '<div class="hud-result-title">' +
      title +
      "</div>" +
      '<div class="hud-result-sub">' +
      sub +
      "</div>";
  }
  function clearResult() {
    $id("hud-result").className = "hud-result hidden";
  }

  // ---------------------------------------------------------------- maze
  function mazeStart() {
    maze.state = [0, 0, 6, 5, 1];
    maze.steps = 0;
    maze.over = false;
    maze.won = false;
    mazeHist.length = 0;
    mazePathCache = {};
    clearResult();
    $id("hud-status").textContent = "Playing";
    $id("hud-steps").textContent = "0";
  }

  function mazeAct(a) {
    if (maze.mode !== "play" || maze.over) return;
    maze.last = S.ACT_NAMES[a];
    var res = S.mazeStep(
      maze.state,
      a,
      maze.vi ? maze.vi.V : null,
      maze.mdp ? maze.mdp.map : null,
    );
    maze.state = res.state;
    maze.steps++;
    $id("hud-steps").textContent = String(maze.steps);
    if (res.done) {
      maze.over = true;
      maze.won = res.won;
      if (res.won) {
        maze.wins++;
        $id("hud-wins").textContent = String(maze.wins);
        setResult("🏆 Exit reached!", "Thomas escaped the minotaur.", "win");
        log("🏆 Victory in " + maze.steps + " steps!");
      } else {
        maze.losses++;
        $id("hud-losses").textContent = String(maze.losses);
        setResult("💥 Caught!", "The minotaur got Thomas.", "fail");
        log("💥 Caught by the minotaur after " + maze.steps + " steps.");
      }
    }
    render();
  }

  function mazeBestAction() {
    if (!maze.mdp || !maze.vi) return null;
    var key = maze.state.join(",");
    var sid = maze.mdp.map.get(key);
    if (sid === undefined) return null;
    return maze.vi.policy[sid];
  }

  function runVI() {
    stopVI();
    clearResult();
    log("🧮 Value iteration on the full joint MDP…");
    maze.mdp = S.buildMazeMdp();
    maze.vi = S.valueIteration(maze.mdp, maze.viGamma, 1e-3);
    maze.viSweep = maze.vi.sweeps.length - 1;
    log(
      "✅ VI converged in " +
        maze.vi.iterations +
        " sweeps · max V=" +
        Math.round(Math.max.apply(null, maze.vi.V)),
    );
    $id("hud-status").textContent = "VI done";
    var wp = mazeWinProb();
    if (wp !== null)
      $id("hud-winprob").textContent = Math.round(wp * 100) + "%";
    render();
  }

  function animateVI() {
    if (maze.mode !== "vi" || !maze.vi) return;
    stopVI();
    $id("btn-vi-run").textContent = "⏸ Pause VI";
    maze.viSweep = 0;
    log("▶ Animating " + maze.vi.sweeps.length + " sweeps…");
    var speed = maze.viSpeed;
    maze.viTimer = setInterval(
      function () {
        if (!maze.vi) {
          stopVI();
          return;
        }
        maze.viSweep++;
        if (maze.viSweep >= maze.vi.sweeps.length) {
          stopVI();
          log("🏁 VI converged (final policy shown).");
        }
        render();
      },
      Math.max(16, 1000 / speed),
    );
  }

  function stopVI() {
    if (maze.viTimer) {
      clearInterval(maze.viTimer);
      maze.viTimer = null;
    }
    var b = $id("btn-vi-run");
    if (b) b.textContent = "▶ Run VI";
  }

  // ---------------------------------------------------------------- car
  function carEta(order) {
    return S.fourierEtas(order);
  }

  function carQ(w, etas, s, a) {
    var phi = S.fourierFeatures(etas, s);
    var acc = 0;
    for (var k = 0; k < phi.length; k++) acc += w[k][a] * phi[k];
    return acc;
  }

  function carChoose(w, etas, s, eps) {
    if (Math.random() < eps) return Math.floor(Math.random() * 3);
    var best = 0;
    for (var a = 1; a < 3; a++)
      if (carQ(w, etas, s, a) > carQ(w, etas, s, best)) best = a;
    return best;
  }

  function carState(env) {
    return [(env.x + 1.2) / 1.8, (env.v + 0.07) / 0.14];
  }

  function carReset() {
    car.env = S.createMountainCar(Math.floor(Math.random() * 1e9));
    car.rewards = [];
    car.episode = 0;
  }

  /** Runs one learning episode synchronously (fast), animates the last one. */
  function carLearnEpisode() {
    var order = Number($id("car-order").value);
    var gamma = 1;
    var lambda = Number($id("car-lambda").value);
    var alpha = Number($id("car-alpha").value);
    var eps = Number($id("car-eps").value);
    var etas = carEta(order);
    var w = car.w;
    var env = car.env;
    var s = carState(env);
    var a = carChoose(w, etas, s, eps);
    var z = [];
    for (var k = 0; k < etas.length + 1; k++) z.push([0, 0, 0]);
    var reward = 0;
    var steps = 0;
    var done = false;
    var trace = [];
    while (!done && steps < car.maxIters) {
      steps++;
      var res = S.mcStep(env, a);
      reward += res.reward;
      trace.push({ x: env.x, v: env.v });
      var ns = carState(env);
      var na = res.done ? 0 : carChoose(w, etas, ns, eps);
      var phi = S.fourierFeatures(etas, s);
      var delta =
        res.reward +
        (res.done ? 0 : gamma * carQ(w, etas, ns, na)) -
        carQ(w, etas, s, a);
      for (var k2 = 0; k2 < phi.length; k2++) {
        z[k2][a] = gamma * lambda * z[k2][a] + phi[k2];
        w[k2][a] += alpha * delta * z[k2][a];
      }
      s = ns;
      a = na;
      done = res.done;
    }
    car.rewards.push(reward);
    car.episode++;
    car.lastTrace = trace;
    car.lastReward = reward;
    return done;
  }

  function carLearnStart() {
    car.mode = "learn";
    carReset();
    var order = Number($id("car-order").value);
    car.w = [];
    for (var k = 0; k < carEta(order).length + 1; k++) car.w.push([0, 0, 0]);
    log(
      "🧠 SARSA(λ) · Fourier order " +
        order +
        " · λ=" +
        Number($id("car-lambda").value) +
        " · α=" +
        Number($id("car-alpha").value) +
        " · ε=" +
        Number($id("car-eps").value),
    );
    if (car.learnTimer) clearInterval(car.learnTimer);
    car.learnTimer = setInterval(function () {
      if (car.mode !== "learn") return;
      for (var i = 0; i < 3; i++) {
        var done = carLearnEpisode();
        if (done) {
          log("🎉 Car reached the flag on episode " + car.episode + "!");
          stopCarLearn();
          break;
        }
        if (car.episode >= car.nEpisodes) {
          log(
            "🏁 " +
              car.nEpisodes +
              " episodes done — best avg reward: " +
              Math.round(bestAvg() * 10) / 10,
          );
          stopCarLearn();
          break;
        }
      }
      updateCarHud();
      render();
    }, 30);
  }

  function stopCarLearn() {
    if (car.learnTimer) {
      clearInterval(car.learnTimer);
      car.learnTimer = null;
    }
  }

  function bestAvg() {
    if (car.rewards.length === 0) return 0;
    var tail = car.rewards.slice(-10);
    return (
      tail.reduce(function (a, b) {
        return a + b;
      }, 0) / tail.length
    );
  }

  function updateCarHud() {
    $id("hud-episode").textContent = car.episode + " / " + car.nEpisodes;
    $id("hud-reward").textContent = car.rewards.length
      ? String(Math.round(bestAvg() * 10) / 10)
      : "—";
    $id("hud-best").textContent = car.rewards.length
      ? String(Math.max.apply(null, car.rewards))
      : "—";
    $id("hud-pos").textContent = car.env.x.toFixed(2);
  }

  function carDrive(a) {
    car.mode = "drive";
    stopCarLearn();
    var res = S.mcStep(car.env, a);
    if (res.done) {
      log("🏁 Goal reached from x=" + car.env.x.toFixed(2) + "!");
      carReset();
    }
    updateCarHud();
    render();
  }

  // ---------------------------------------------------------------- render
  function sizeCanvas() {
    var panel = $id("stage-panel");
    var w = panel.clientWidth;
    var h = panel.clientHeight;
    var chartEl = document.getElementById("car-chart");
    if (chartEl && !chartEl.classList.contains("hidden")) {
      h -= chartEl.offsetHeight + 14; // chart box + its margin
    }
    var dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function render() {
    if (tab === "maze") renderMaze();
    else renderCar();
    if (tab === "car" && carChart && Date.now() - lastChartT > 250) {
      carChart.update();
      lastChartT = Date.now();
    }
  }

  function renderMaze() {
    var p = pal();
    var w = canvas.width / Math.max(1, window.devicePixelRatio || 1);
    var h = canvas.height / Math.max(1, window.devicePixelRatio || 1);
    ctx.fillStyle = p.bg;
    ctx.fillRect(0, 0, w, h);

    var cols = S.COLS;
    var rows = S.ROWS;
    var pad = Math.min(w, h) * 0.05;
    var cw = Math.min((w - pad * 2) / cols, (h - pad * 2) / rows);
    var ox = (w - cw * cols) / 2;
    var oy = (h - cw * rows) / 2;

    // VI value heat layer
    var heat = null;
    if (maze.mode === "vi" && maze.mdp && maze.vi) {
      heat = viSliceHeat();
    }
    // minotaur position forecast (play) — notebook cell_probs style
    var probs = null;
    if (maze.mode === "play" && !maze.over) {
      probs = minoProbs(10);
    }
    // planned policy path (play) — notebook animate_solution style
    var path = null;
    if (maze.mode === "play" && maze.mdp && maze.vi) {
      var _ck = maze.state.join(",");
      if (!(_ck in mazePathCache)) mazePathCache[_ck] = mazePolicyPath();
      path = mazePathCache[_ck];
    }

    var mi = maze.state[2],
      mj = maze.state[3];
    var ti = maze.state[0],
      tj = maze.state[1];

    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var x = ox + j * cw;
        var y = oy + i * cw;
        var cell = S.MAZE[i][j];
        var isT = i === ti && j === tj;
        var isM = i === mi && j === mj;

        if (cell === 1) {
          ctx.fillStyle = p.wall;
          ctx.fillRect(x, y, cw, cw);
          continue;
        }
        if (cell === 2) {
          ctx.fillStyle = p.exit;
          ctx.fillRect(x, y, cw, cw);
          ctx.fillStyle = p.exitEdge;
          ctx.font = "700 " + Math.max(9, cw * 0.26) + "px system-ui";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("EXIT", x + cw / 2, y + cw / 2 + 1);
        } else if (isT) {
          ctx.fillStyle = p.thomas;
          ctx.fillRect(x, y, cw, cw);
        } else if (isM) {
          ctx.fillStyle = p.mino;
          ctx.fillRect(x, y, cw, cw);
        } else {
          ctx.fillStyle = p.empty;
          ctx.fillRect(x, y, cw, cw);
        }
        ctx.strokeStyle = p.grid;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, cw - 1);

        // VI value heat overlay
        if (heat && cell !== 1) {
          var v = heat[i][j];
          if (v !== null) {
            var tt = clamp01((v - heatMin) / (heatMax - heatMin || 1));
            var ci = Math.min(
              p.heat.length - 1,
              Math.floor(tt * p.heat.length),
            );
            ctx.fillStyle = p.heat[ci] + "cc";
            ctx.fillRect(x, y, cw, cw);
            ctx.strokeStyle = p.grid;
            ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, cw - 1);
          }
        }

        // minotaur probability overlay (play) — faint red heat
        if (probs && cell !== 1 && !isM) {
          var pr = probs[i + "," + j] || 0;
          if (pr > 0.001) {
            ctx.fillStyle = "rgba(239,68,68," + (0.1 + 0.5 * pr) + ")";
            ctx.fillRect(x, y, cw, cw);
          }
        }

        // VI policy arrow
        if (maze.mode === "vi" && maze.vi && cell !== 1 && maze.mdp) {
          var key =
            i + "," + j + "," + maze.state[2] + "," + maze.state[3] + ",1";
          var sid = maze.mdp.map.get(key);
          if (sid !== undefined) {
            drawArrow(
              x + cw / 2,
              y + cw / 2,
              cw * 0.32,
              maze.vi.policy[sid],
              p.text,
            );
          }
        }
      }
    }

    // notebook-style arrows (minoutaur_maze.py compute_arrow): from the cell
    // CENTRE toward the shared edge (0.4 cell), step number at the midpoint
    function nbArrow(cx, cy, dj, di, num, color) {
      var dx = dj * cw * 0.4,
        dy = di * cw * 0.4;
      var ex = cx + dx,
        ey = cy + dy;
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, cw * 0.05);
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      var ang = Math.atan2(dy, dx);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(
        ex - cw * 0.16 * Math.cos(ang - 0.45),
        ey - cw * 0.16 * Math.sin(ang - 0.45),
      );
      ctx.lineTo(
        ex - cw * 0.16 * Math.cos(ang + 0.45),
        ey - cw * 0.16 * Math.sin(ang + 0.45),
      );
      ctx.closePath();
      ctx.fill();
      var nx = cx + dx / 2;
      var ny = cy + dy / 2;
      // perpendicular offset so the number sits BESIDE the arrow line
      // (like the notebook), never on top of it
      var L = Math.hypot(dx, dy) || 1;
      nx += (-dy / L) * cw * 0.17;
      ny += (dx / L) * cw * 0.17;
      ctx.font = "700 " + Math.max(10, cw * 0.26) + "px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.lineWidth = Math.max(3, cw * 0.1);
      ctx.strokeStyle = "rgba(255,255,255,0.92)";
      ctx.strokeText(String(num), nx, ny);
      ctx.fillStyle = "#0f172a";
      ctx.fillText(String(num), nx, ny);
    }

    // planned joint route — the VI policy's FUTURE, notebook-style: orange
    // Thomas moves, red minotaur moves, numbers continuing from the executed
    // steps; re-simulated per state (cached), so it updates as we move
    if (path && path.length > 1) {
      for (var pk = 0; pk < path.length - 1; pk++) {
        var a0 = path[pk],
          a1 = path[pk + 1];
        if (a0.i !== a1.i || a0.j !== a1.j) {
          nbArrow(
            ox + (a0.j + 0.5) * cw,
            oy + (a0.i + 0.5) * cw,
            a1.j - a0.j,
            a1.i - a0.i,
            maze.steps + pk,
            "#f97316",
          );
        }
        if (a0.mi !== a1.mi || a0.mj !== a1.mj) {
          nbArrow(
            ox + (a0.mj + 0.5) * cw,
            oy + (a0.mi + 0.5) * cw,
            a1.mj - a0.mj,
            a1.mi - a0.mi,
            maze.steps + pk,
            "#ef4444",
          );
        }
      }
    }

    // character sprites
    Sprites.drawMinotaur(
      ctx,
      ox + (mj + 0.5) * cw,
      oy + (mi + 0.5) * cw,
      cw * 0.72,
    );
    Sprites.drawThomas(
      ctx,
      ox + (tj + 0.5) * cw,
      oy + (ti + 0.5) * cw,
      cw * 0.7,
    );

    // AI suggestion arrow (play mode, manual) — notebook geometry
    if (maze.mode === "play" && !maze.over && !mazeAutoTimer) {
      var best = mazeBestAction();
      if (best !== null && best !== 0) {
        nbArrow(
          ox + (tj + 0.5) * cw,
          oy + (ti + 0.5) * cw,
          ACT_D[best][1],
          ACT_D[best][0],
          maze.steps,
          "#f97316",
        );
      }
    }

    // legend (bottom-left) — rounded swatch before label, on a soft card
    ctx.font = "600 " + Math.max(11, cw * 0.24) + "px system-ui";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    var ly = oy + rows * cw + 26;
    var lx = ox + 2;
    var sw = Math.max(10, cw * 0.22);
    var items = [
      ["Thomas", p.thomasEdge],
      ["Minotaur", p.minoEdge],
      ["Exit", p.exitEdge],
    ];
    if (probs) items.push(["Minotaur forecast", "rgba(239,68,68,0.55)"]);
    var widths = items.map(function (it) {
      return ctx.measureText(it[0]).width + sw + 10;
    });
    var totalW = widths.reduce(function (a, b) {
      return a + b;
    }, 0);
    // soft card behind the legend
    ctx.fillStyle = p.panel;
    ctx.beginPath();
    ctx.roundRect(lx - 8, ly - sw - 10, totalW + 16, sw + 18, 8);
    ctx.fill();
    ctx.strokeStyle = p.faint;
    ctx.lineWidth = 1;
    ctx.stroke();
    var x = lx;
    for (var li = 0; li < items.length; li++) {
      ctx.beginPath();
      ctx.roundRect(x, ly - sw - 1, sw, sw, sw / 2);
      ctx.fillStyle = items[li][1];
      ctx.fill();
      x += sw + 6;
      ctx.fillStyle = p.muted;
      ctx.fillText(items[li][0], x, ly);
      x += widths[li] - sw - 6;
    }
  }

  // ------------------------------------------------------------- maze extras
  // action deltas (notebook convention: 0 wait,1 left,2 right,3 up,4 down)
  var ACT_D = [
    [0, 0],
    [0, -1],
    [0, 1],
    [-1, 0],
    [1, 0],
  ];
  var mazeAutoTimer = null;
  var mazeHist = []; // executed (Thomas, minotaur) states per step
  var heatMin = 0;
  var heatMax = 1;

  /** Planned joint route from the current state, following the VI policy:
   *  Thomas acts from the policy, the minotaur responds (random valid move —
   *  the lab's dynamics, exactly the notebook's recorded simulation).
   *  Cached per state so the plan is stable between moves. */
  var mazePathCache = {};
  function mazePolicyPath() {
    if (!maze.mdp || !maze.vi) return null;
    var key0 =
      maze.state[0] +
      "," +
      maze.state[1] +
      "," +
      maze.state[2] +
      "," +
      maze.state[3] +
      ",1";
    if (mazePathCache[key0]) return mazePathCache[key0];
    var path = [];
    var st = maze.state.slice();
    var seen = {};
    for (var k = 0; k < 48; k++) {
      var key = st[0] + "," + st[1] + "," + st[2] + "," + st[3] + ",1";
      if (seen[key]) break;
      seen[key] = true;
      var sid = maze.mdp.map.get(key);
      if (sid === undefined) break;
      var a = maze.vi.policy[sid];
      path.push({ i: st[0], j: st[1], mi: st[2], mj: st[3], a: a });
      if (S.MAZE[st[0]][st[1]] === 2) break;
      var res = S.mazeStep(
        st,
        a,
        maze.vi ? maze.vi.V : null,
        maze.mdp ? maze.mdp.map : null,
      );
      st = res.state;
      if (res.done) break;
    }
    mazePathCache[key0] = path;
    return path;
  }

  /**
   * Minotaur position forecast: marginal random walk (uniform valid moves)
   * from its current cell for `horizon` steps — the notebook's cell_probs.
   */
  function minoProbs(horizon) {
    var cur = {};
    cur[maze.state[2] + "," + maze.state[3]] = 1;
    for (var h = 0; h < horizon; h++) {
      var nxt = {};
      for (var k in cur) {
        var parts = k.split(",");
        var mi = +parts[0],
          mj = +parts[1];
        if (!isFinite(mi) || !isFinite(mj)) continue;
        var moves = S.validMoves(S.MAZE, [mi, mj], false);
        var share = cur[k] / moves.length;
        for (var m = 0; m < moves.length; m++) {
          var act = moves[m];
          var nk = mi + ACT_D[act][0] + "," + (mj + ACT_D[act][1]);
          nxt[nk] = (nxt[nk] || 0) + share;
        }
      }
      cur = nxt;
    }
    var max = 0;
    for (var k2 in cur) if (cur[k2] > max) max = cur[k2];
    if (max === 0) return null;
    var norm = {};
    for (var k3 in cur) norm[k3] = cur[k3] / max;
    return norm;
  }

  /** Monte-Carlo win probability of the current policy from the current state. */
  function mazeWinProb() {
    if (!maze.mdp || !maze.vi) return null;
    var wins = 0,
      n = 300;
    for (var s = 0; s < n; s++) {
      var st = maze.state.slice();
      for (var k = 0; k < 40; k++) {
        var sid = maze.mdp.map.get(st.join(","));
        if (sid === undefined) break;
        var a = maze.vi.policy[sid];
        var res = S.mazeStep(
          st,
          a,
          maze.vi ? maze.vi.V : null,
          maze.mdp ? maze.mdp.map : null,
        );
        st = res.state;
        if (res.done) {
          if (res.won) wins++;
          break;
        }
      }
    }
    return wins / n;
  }

  function mazeAutoStart() {
    if (mazeAutoTimer) return;
    $id("btn-auto").textContent = "⏸ Stop auto";
    mazeAutoTimer = setInterval(function () {
      if (maze.over || tab !== "maze" || maze.mode !== "play") {
        mazeAutoStop();
        return;
      }
      var a = mazeBestAction();
      if (a === null) {
        mazeAutoStop();
        return;
      }
      mazeAct(a);
    }, 650);
  }

  function mazeAutoStop() {
    if (mazeAutoTimer) {
      clearInterval(mazeAutoTimer);
      mazeAutoTimer = null;
    }
    var b = $id("btn-auto");
    if (b) b.textContent = "🤖 Auto";
  }

  function viSliceHeat() {
    var mdp = maze.mdp;
    var sweepV =
      maze.vi.sweeps[Math.min(maze.viSweep, maze.vi.sweeps.length - 1)];
    var mi = maze.state[2],
      mj = maze.state[3];
    var grid = [];
    var vals = [];
    for (var i = 0; i < S.ROWS; i++) {
      grid.push(new Array(S.COLS).fill(null));
      for (var j = 0; j < S.COLS; j++) {
        if (S.MAZE[i][j] === 1) continue;
        var sid = mdp.map.get(i + "," + j + "," + mi + "," + mj + ",1");
        if (sid !== undefined && mdp.subset[sid] === 0) {
          grid[i][j] = sweepV[sid];
          vals.push(sweepV[sid]);
        }
      }
    }
    heatMin = vals.length ? Math.min.apply(null, vals) : 0;
    heatMax = vals.length ? Math.max.apply(null, vals) : 1;
    return grid;
  }

  function drawArrow(cx, cy, len, action, color, noHead) {
    var dx = 0,
      dy = 0;
    if (action === 1) {
      dx = -1;
    } else if (action === 2) {
      dx = 1;
    } else if (action === 3) {
      dy = -1;
    } else if (action === 4) {
      dy = 1;
    } else return; // wait
    var x2 = cx + dx * len;
    var y2 = cy + dy * len;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    if (!noHead) {
      var ang = Math.atan2(dy, dx);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 7 * Math.cos(ang - 0.5), y2 - 7 * Math.sin(ang - 0.5));
      ctx.lineTo(x2 - 7 * Math.cos(ang + 0.5), y2 - 7 * Math.sin(ang + 0.5));
      ctx.closePath();
      ctx.fill();
    }
  }

  function drawHeatLegend(w, h, p) {
    var lw = Math.min(220, w * 0.35);
    var lx = w - lw - 14;
    var ly = 14;
    ctx.fillStyle = p.panel + "dd";
    ctx.strokeStyle = p.grid;
    ctx.beginPath();
    ctx.roundRect(lx, ly, lw, 44, 8);
    ctx.fill();
    ctx.stroke();
    var grad = ctx.createLinearGradient(lx + 10, 0, lx + lw - 10, 0);
    for (var i = 0; i < p.heat.length; i++) {
      grad.addColorStop(i / (p.heat.length - 1), p.heat[i]);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(lx + 10, ly + 12, lw - 20, 10);
    ctx.fillStyle = p.muted;
    ctx.font = "10px system-ui";
    ctx.textAlign = "left";
    ctx.fillText("V  " + Math.round(heatMin), lx + 10, ly + 34);
    ctx.textAlign = "right";
    ctx.fillText(Math.round(heatMax), lx + lw - 10, ly + 34);
  }

  function renderCar() {
    var p = pal();
    var w = canvas.width / Math.max(1, window.devicePixelRatio || 1);
    var h = canvas.height / Math.max(1, window.devicePixelRatio || 1);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    ctx.clip(); // the track can never paint outside the canvas
    ctx.fillStyle = p.bg;
    ctx.fillRect(0, 0, w, h);

    // the reward chart lives in its own component (#car-chart) above this canvas
    var trackTop = 14;
    var th = h - trackTop - 20;

    // --- track
    var xMin = -1.2,
      xMax = 0.6;
    var margin = 40;
    var tw = w - margin * 2;
    var baseY = trackTop + th - 14;

    function toX(x) {
      return margin + ((x - xMin) / (xMax - xMin)) * tw;
    }
    // sin(3x) over the episode spans [-1, 1]: peak (y=1) at the top of the
    // track box, valley (y=-1) at the bottom — fully inside the canvas
    var yMin = -1,
      yMax = 1;
    var trackTopY = 26,
      trackBottomY = h - 24;
    function toY(y) {
      return (
        trackTopY + ((yMax - y) / (yMax - yMin)) * (trackBottomY - trackTopY)
      );
    }

    // hill polygon
    ctx.beginPath();
    ctx.moveTo(margin, baseY + 14);
    for (var xx = 0; xx <= tw; xx += 3) {
      var xv = xMin + (xx / tw) * (xMax - xMin);
      ctx.lineTo(margin + xx, toY(S.mountainCarTrack(xv)));
    }
    ctx.lineTo(margin + tw, baseY + 14);
    ctx.closePath();
    ctx.fillStyle = p.track;
    ctx.fill();
    ctx.strokeStyle = p.trackEdge;
    ctx.stroke();

    // goal flag
    var gx = toX(0.5);
    var gy = toY(S.mountainCarTrack(0.5));
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(gx, gy - 26);
    ctx.stroke();
    ctx.fillStyle = p.exit;
    ctx.beginPath();
    ctx.moveTo(gx, gy - 26);
    ctx.lineTo(gx + 16, gy - 19);
    ctx.lineTo(gx, gy - 12);
    ctx.closePath();
    ctx.fill();

    // car
    var cx = toX(car.env.x);
    var cy = toY(S.mountainCarTrack(car.env.x));
    ctx.save();
    var rot = Math.atan2(
      toY(S.mountainCarTrack(car.env.x + 0.01)) -
        toY(S.mountainCarTrack(car.env.x - 0.01)),
      toX(car.env.x + 0.01) - toX(car.env.x - 0.01),
    );
    ctx.translate(cx, cy);
    ctx.rotate(rot);
    ctx.fillStyle = p.car;
    ctx.beginPath();
    ctx.roundRect(-18, -9, 36, 16, 5);
    ctx.fill();
    ctx.strokeStyle = p.carEdge;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(0, 0, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    // wheels
    ctx.fillStyle = p.carEdge;
    ctx.beginPath();
    ctx.arc(cx - 10, cy + 2, 4, 0, Math.PI * 2);
    ctx.arc(cx + 10, cy + 2, 4, 0, Math.PI * 2);
    ctx.fill();

    // velocity arrow
    if (Math.abs(car.env.v) > 0.001) {
      ctx.strokeStyle = p.warn;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy - 16);
      ctx.lineTo(cx + car.env.v * 400, cy - 16);
      ctx.stroke();
      ctx.fillStyle = p.warn;
      ctx.beginPath();
      ctx.moveTo(cx + car.env.v * 400, cy - 16);
      ctx.lineTo(cx + car.env.v * 400 - 6 * Math.sign(car.env.v || 1), cy - 20);
      ctx.lineTo(cx + car.env.v * 400 - 6 * Math.sign(car.env.v || 1), cy - 12);
      ctx.closePath();
      ctx.fill();
    }

    // mode label
    ctx.fillStyle = p.muted;
    ctx.font = "600 12px system-ui";
    ctx.textAlign = "left";
    ctx.fillText(
      car.mode === "learn" ? "learning (SARSA λ)" : "manual drive — ← / →",
      margin,
      trackTop - 6,
    );
    ctx.restore();
  }

  function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
  }

  // ---------------------------------------------------------------- input
  var keyMap = {
    ArrowUp: 3,
    KeyW: 3,
    ArrowDown: 4,
    KeyS: 4,
    ArrowLeft: 1,
    KeyA: 1,
    ArrowRight: 2,
    KeyD: 2,
  };

  document.addEventListener("keydown", function (e) {
    if (guideOpen) return;
    if (e.code === "Space") {
      e.preventDefault();
      if (tab === "maze" && maze.mode === "play" && !maze.over) {
        mazeAutoStop();
        mazeAct(0);
        markKey("Space — wait");
      }
      return;
    }
    if (tab === "maze" && keyMap[e.code]) {
      e.preventDefault();
      if (maze.mode === "play" && !maze.over) {
        mazeAutoStop();
        mazeAct(keyMap[e.code]);
        markKey("wasd/arrows — move");
      }
      return;
    }
    if (tab === "car") {
      if (e.code === "ArrowLeft" || e.code === "KeyA") {
        e.preventDefault();
        carDrive(0);
        markKey("← — push left");
      } else if (e.code === "ArrowRight" || e.code === "KeyD") {
        e.preventDefault();
        carDrive(2);
        markKey("→ — push right");
      } else if (e.code === "Space") {
        e.preventDefault();
        carDrive(1);
        markKey("Space — no-op");
      }
    }
    if (e.code === "KeyR") {
      if (tab === "maze") {
        mazeStart();
        log("↻ New round.");
      } else {
        carReset();
        log("↻ Car reset.");
      }
      markKey("R — restart");
    }
  });

  var lastKeyEl = null;
  function markKey(label) {
    var el = document.getElementById("last-key");
    if (!el) return;
    el.textContent = label;
    if (lastKeyEl) lastKeyEl.style.opacity = "";
  }

  // ---------------------------------------------------------------- guide
  var guideOpen = false;
  function wireGuide() {
    var guide = $id("guide");
    function open() {
      guideOpen = true;
      guide.classList.remove("hidden");
    }
    function close() {
      guideOpen = false;
      guide.classList.add("hidden");
    }
    $id("btn-guide").addEventListener("click", open);
    guide.querySelectorAll("[data-close-guide]").forEach(function (el) {
      el.addEventListener("click", close);
    });
    document.addEventListener("keydown", function (e) {
      if (e.code === "Escape" && guideOpen) close();
    });
  }

  // ---------------------------------------------------------------- wiring
  function wire() {
    // tabs
    document.querySelectorAll(".tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll(".tab").forEach(function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
        tab = btn.dataset.tab;
        $id("panel-title").textContent =
          tab === "maze" ? "🐂 Minotaur Maze" : "⛰️ Mountain Car";
        $id("maze-hud").classList.toggle("hidden", tab !== "maze");
        $id("car-hud").classList.toggle("hidden", tab !== "car");
        $id("car-chart").classList.toggle("hidden", tab !== "car");
        $id("maze-controls").classList.toggle("hidden", tab !== "maze");
        $id("car-controls").classList.toggle("hidden", tab !== "car");
        stopVI();
        stopCarLearn();
        if (tab === "car" && !car.w) carLearnStart();
        render();
      });
    });

    // maze modes
    $id("mode-play").addEventListener("click", function () {
      maze.mode = "play";
      stopVI();
      $id("mode-play").classList.add("active");
      $id("mode-vi").classList.remove("active");
      $id("vi-options").classList.add("hidden");
      $id("play-options").classList.remove("hidden");
      $id("hud-status").textContent = "Playing";
      render();
    });
    $id("mode-play").addEventListener("click", function () {
      maze.mode = "play";
      $id("mode-play").classList.add("active");
      $id("mode-vi").classList.remove("active");
      $id("vi-options").classList.add("hidden");
      $id("play-options").classList.remove("hidden");
      if (!maze.vi) {
        runVI();
        var wp1 = mazeWinProb();
        if (wp1 !== null)
          $id("hud-winprob").textContent = Math.round(wp1 * 100) + "%";
      }
      render();
    });
    $id("mode-vi").addEventListener("click", function () {
      maze.mode = "vi";
      $id("mode-vi").classList.add("active");
      $id("mode-play").classList.remove("active");
      $id("vi-options").classList.remove("hidden");
      $id("play-options").classList.add("hidden");
      $id("hud-status").textContent = "VI ready";
      if (!maze.vi) runVI();
      render();
    });
    $id("btn-vi-run").addEventListener("click", function () {
      if (maze.viTimer) {
        stopVI();
        log("⏸ VI paused.");
      } else {
        if (!maze.vi) runVI(); // a rule change invalidated the solution
        animateVI();
      }
    });
    var autoBtn = $id("btn-auto");
    if (autoBtn) {
      autoBtn.addEventListener("click", function () {
        if (mazeAutoTimer) mazeAutoStop();
        else mazeAutoStart();
      });
    }
    $id("vi-gamma").addEventListener("input", function () {
      maze.viGamma = Number(this.value);
      $id("vi-gamma-v").textContent = this.value;
    });
    $id("vi-speed").addEventListener("input", function () {
      maze.viSpeed = Number(this.value);
      $id("vi-speed-v").textContent = this.value + " sweeps/s";
    });
    $id("mino-wait").addEventListener("change", function () {
      stopVI();
      S.setMinoCanWait(this.checked);
      maze.mdp = S.buildMazeMdp();
      maze.vi = null;
      mazePathCache = {};
      log(
        "🚶 Minotaur may wait: " +
          (this.checked ? "ON — it can hold position" : "OFF") +
          " — MDP rebuilt. Run VI to solve.",
      );
      render();
    });
    $id("mino-rule").addEventListener("change", function () {
      stopVI();
      S.setMinoRule(this.value);
      maze.mdp = S.buildMazeMdp();
      maze.vi = null;
      mazePathCache = {};
      $id("hud-status").textContent = "MDP rebuilt — run VI";
      log(
        "🐂 Minotaur rule: " +
          this.options[this.selectedIndex].text +
          " — MDP rebuilt, run VI to re-solve.",
      );
      render();
    });

    // car modes
    $id("car-learn").addEventListener("click", function () {
      car.mode = "learn";
      $id("car-learn").classList.add("active");
      $id("car-play").classList.remove("active");
      if (!car.w) carLearnStart();
      else {
        stopCarLearn();
        log("🧠 Learning resumed.");
        carLearnStart();
      }
      render();
    });
    $id("car-play").addEventListener("click", function () {
      car.mode = "drive";
      $id("car-play").classList.add("active");
      $id("car-learn").classList.remove("active");
      stopCarLearn();
      log("🎮 Manual drive — ← / → throttle.");
      render();
    });

    // theme
    $id("btn-theme").addEventListener("click", function () {
      var t =
        document.documentElement.getAttribute("data-theme") === "dark"
          ? "light"
          : "dark";
      document.documentElement.setAttribute("data-theme", t);
      try {
        localStorage.setItem("theme", t);
      } catch (e) {}
      applyTheme();
      render();
    });
    window
      .matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", function (ev) {
        if (localStorage.getItem("theme")) return;
        document.documentElement.setAttribute(
          "data-theme",
          ev.matches ? "dark" : "light",
        );
        applyTheme();
        render();
      });

    $id("btn-restart").addEventListener("click", function () {
      if (tab === "maze") {
        mazeStart();
        log("↻ New round.");
      } else {
        carReset();
        log("↻ Car reset.");
      }
      render();
    });

    window.addEventListener("resize", function () {
      sizeCanvas();
      render();
    });
  }

  // ---------------------------------------------------------------- init
  // D3 reward chart (own component, SVG — not canvas/Pixi)
  function carAvgData() {
    var n = car.rewards.length;
    var data = [];
    for (var e = 0; e < n; e++) {
      var from = Math.max(0, e - 9);
      var slice = car.rewards.slice(from, e + 1);
      data.push(
        slice.reduce(function (a, b) {
          return a + b;
        }, 0) / slice.length,
      );
    }
    return data;
  }
  var carChart = null;
  var lastChartT = 0;
  function initChart() {
    var el = document.getElementById("car-chart");
    if (!el || !window.MiniChart) return;
    carChart = window.MiniChart(el, {
      height: 148,
      title: "episode reward (10-ep running avg)",
      emptyText: "training…",
      pad: 6,
      getData: carAvgData,
      color: function () {
        return pal().chart;
      },
    });
  }

  function init() {
    initChart();
    setInterval(function () {
      if (tab === "car" && carChart) carChart.update();
    }, 300);
    applyTheme();
    wireGuide();
    wire();
    sizeCanvas();
    mazeStart();
    maze.mdp = S.buildMazeMdp();
    runVI();
    var wpInit = mazeWinProb();
    if (wpInit !== null)
      $id("hud-winprob").textContent = Math.round(wpInit * 100) + "%";
    log("🐂 Minotaur Maze loaded — arrows to move, reach the gold exit.");
    log("🧮 Value Iteration mode solves the full joint MDP (2240 states).");
    log("⛰️ Mountain Car tab: SARSA(λ) with Fourier features trains live.");
    render();
  }

  init();
})();
