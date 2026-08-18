/**
 * RL Lab 1 — interactive cockpit: Minotaur Maze (play + value iteration)
 * and Mountain Car (live SARSA(lambda) learning + manual driving).
 */
(function () {
  "use strict";

  var S = window.RL1;
  var canvas = document.getElementById("view");
  var ctx = canvas.getContext("2d");
  var $id = function (id) { return document.getElementById(id); };

  // ---------------------------------------------------------------- theme
  var PAL = {
    dark: {
      bg: "#0b0f19", panel: "#111c30", grid: "rgba(148,163,184,0.14)",
      wall: "#1e293b", wallEdge: "#334155", empty: "#101a2c",
      exit: "#fbbf24", exitEdge: "#f59e0b",
      thomas: "#22c55e", thomasEdge: "#15803d", mino: "#ef4444", minoEdge: "#b91c1c",
      text: "#e2e8f0", muted: "#94a3b8", faint: "#64748b",
      heat: ["#0f172a", "#164e63", "#0e7490", "#06b6d4", "#22d3ee", "#fbbf24"],
      track: "#1e293b", trackEdge: "#334155", car: "#38bdf8", carEdge: "#0284c7",
      chart: "#22d3ee", chartGrid: "rgba(148,163,184,0.15)",
    },
    light: {
      bg: "#f1f5f9", panel: "#ffffff", grid: "rgba(51,65,85,0.12)",
      wall: "#cbd5e1", wallEdge: "#94a3b8", empty: "#f8fafc",
      exit: "#f59e0b", exitEdge: "#b45309",
      thomas: "#16a34a", thomasEdge: "#15803d", mino: "#ef4444", minoEdge: "#b91c1c",
      text: "#0f172a", muted: "#475569", faint: "#64748b",
      heat: ["#f8fafc", "#cffafe", "#67e8f9", "#06b6d4", "#0e7490", "#f59e0b"],
      track: "#e2e8f0", trackEdge: "#94a3b8", car: "#0284c7", carEdge: "#0c4a6e",
      chart: "#0891b2", chartGrid: "rgba(51,65,85,0.12)",
    },
  };

  function pal() {
    var t = document.documentElement.getAttribute("data-theme");
    return PAL[t === "dark" ? "dark" : "light"];
  }

  function applyTheme() {
    var t = document.documentElement.getAttribute("data-theme");
    $id("btn-theme").textContent = t === "dark" ? "☀️" : "🌙";
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
      '<div class="hud-result-title">' + title + "</div>" +
      '<div class="hud-result-sub">' + sub + "</div>";
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
    clearResult();
    $id("hud-status").textContent = "Playing";
    $id("hud-steps").textContent = "0";
  }

  function mazeAct(a) {
    if (maze.mode !== "play" || maze.over) return;
    maze.last = S.ACT_NAMES[a];
    var res = S.mazeStep(maze.state, a);
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
      "✅ VI converged in " + maze.vi.iterations + " sweeps · max V=" +
        Math.round(Math.max.apply(null, maze.vi.V)),
    );
    $id("hud-status").textContent = "VI done";
    render();
  }

  function animateVI() {
    if (maze.mode !== "vi" || !maze.vi) return;
    stopVI();
    maze.viSweep = 0;
    log("▶ Animating " + maze.vi.sweeps.length + " sweeps…");
    var speed = maze.viSpeed;
    maze.viTimer = setInterval(function () {
      maze.viSweep++;
      if (maze.viSweep >= maze.vi.sweeps.length) {
        stopVI();
        log("🏁 VI converged (final policy shown).");
      }
      render();
    }, Math.max(16, 1000 / speed));
  }

  function stopVI() {
    if (maze.viTimer) {
      clearInterval(maze.viTimer);
      maze.viTimer = null;
    }
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
    for (var a = 1; a < 3; a++) if (carQ(w, etas, s, a) > carQ(w, etas, s, best)) best = a;
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
      var delta = res.reward + (res.done ? 0 : gamma * carQ(w, etas, ns, na)) - carQ(w, etas, s, a);
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
      "🧠 SARSA(λ) · Fourier order " + order + " · λ=" +
        Number($id("car-lambda").value) + " · α=" + Number($id("car-alpha").value) +
        " · ε=" + Number($id("car-eps").value),
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
          log("🏁 " + car.nEpisodes + " episodes done — best avg reward: " +
            Math.round(bestAvg() * 10) / 10);
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
    return tail.reduce(function (a, b) { return a + b; }, 0) / tail.length;
  }

  function updateCarHud() {
    $id("hud-episode").textContent = car.episode + " / " + car.nEpisodes;
    $id("hud-reward").textContent = car.rewards.length ? String(Math.round(bestAvg() * 10) / 10) : "—";
    $id("hud-best").textContent = car.rewards.length ? String(Math.max.apply(null, car.rewards)) : "—";
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
  }

  function renderMaze() {
    var p = pal();
    var w = canvas.width / Math.max(1, window.devicePixelRatio || 1);
    var h = canvas.height / Math.max(1, window.devicePixelRatio || 1);
    ctx.fillStyle = p.bg;
    ctx.fillRect(0, 0, w, h);

    var cols = S.COLS;
    var rows = S.ROWS;
    var pad = Math.min(w, h) * 0.06;
    var cw = Math.min((w - pad * 2) / cols, (h - pad * 2) / rows);
    var ox = (w - cw * cols) / 2;
    var oy = (h - cw * rows) / 2;

    // heatmap layer (VI mode)
    var heat = null;
    if (maze.mode === "vi" && maze.mdp && maze.vi) {
      heat = viSliceHeat();
    }

    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var x = ox + j * cw;
        var y = oy + i * cw;
        var cell = S.MAZE[i][j];
        if (cell === 1) {
          ctx.fillStyle = p.wall;
          ctx.fillRect(x, y, cw, cw);
          ctx.strokeStyle = p.wallEdge;
          ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, cw - 1);
        } else {
          ctx.fillStyle = p.empty;
          ctx.fillRect(x, y, cw, cw);
          ctx.strokeStyle = p.grid;
          ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, cw - 1);
          if (cell === 2) {
            ctx.fillStyle = p.exit;
            ctx.beginPath();
            ctx.arc(x + cw / 2, y + cw / 2, cw * 0.32, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = p.exitEdge;
            ctx.stroke();
            ctx.fillStyle = "#0f172a";
            ctx.font = "700 " + Math.max(10, cw * 0.3) + "px system-ui";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("EXIT", x + cw / 2, y + cw / 2 + 1);
          }
        }
        // heat overlay
        if (heat && cell !== 1) {
          var v = heat[i][j];
          if (v !== null) {
            var t = clamp01((v - heatMin) / (heatMax - heatMin || 1));
            var ci = Math.min(p.heat.length - 1, Math.floor(t * p.heat.length));
            ctx.fillStyle = p.heat[ci] + "cc";
            ctx.fillRect(x, y, cw, cw);
          }
        }
        // VI policy arrow
        if (maze.mode === "vi" && maze.vi && cell !== 1 && maze.mdp) {
          var key = i + "," + j + "," + maze.state[2] + "," + maze.state[3] + ",1";
          var sid = maze.mdp.map.get(key);
          if (sid !== undefined) {
            drawArrow(x + cw / 2, y + cw / 2, cw * 0.32, maze.vi.policy[sid], p.text);
          }
        }
      }
    }

    // entities
    var mi = maze.state[2], mj = maze.state[3];
    var ti = maze.state[0], tj = maze.state[1];
    // minotaur
    ctx.fillStyle = p.mino;
    ctx.beginPath();
    ctx.arc(ox + (mj + 0.5) * cw, oy + (mi + 0.5) * cw, cw * 0.38, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = p.minoEdge;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "700 " + Math.max(10, cw * 0.32) + "px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("M", ox + (mj + 0.5) * cw, oy + (mi + 0.5) * cw + 1);

    // thomas
    ctx.fillStyle = p.thomas;
    ctx.beginPath();
    ctx.arc(ox + (tj + 0.5) * cw, oy + (ti + 0.5) * cw, cw * 0.34, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = p.thomasEdge;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.fillText("T", ox + (tj + 0.5) * cw, oy + (ti + 0.5) * cw + 1);

    // AI suggestion (play mode)
    if (maze.mode === "play" && !maze.over) {
      var best = mazeBestAction();
      if (best !== null && best !== 0) {
        var bx = ox + (tj + 0.5) * cw;
        var by = oy + (ti + 0.5) * cw;
        ctx.strokeStyle = p.warn;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([4, 3]);
        drawArrow(bx, by, cw * 0.6, best, p.warn, true);
        ctx.setLineDash([]);
        ctx.fillStyle = p.warn;
        ctx.font = "600 " + Math.max(9, cw * 0.22) + "px system-ui";
        ctx.textAlign = "center";
        ctx.fillText("AI", bx, by - cw * 0.5);
      }
    }

    // VI legend
    if (maze.mode === "vi" && heat) {
      drawHeatLegend(w, h, p);
    }
  }

  var heatMin = 0;
  var heatMax = 1;

  function viSliceHeat() {
    var mdp = maze.mdp;
    var sweepV = maze.vi.sweeps[Math.min(maze.viSweep, maze.vi.sweeps.length - 1)];
    var mi = maze.state[2], mj = maze.state[3];
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
    var dx = 0, dy = 0;
    if (action === 1) { dx = -1; }
    else if (action === 2) { dx = 1; }
    else if (action === 3) { dy = -1; }
    else if (action === 4) { dy = 1; }
    else return; // wait
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
    ctx.fillStyle = p.bg;
    ctx.fillRect(0, 0, w, h);

    var chartH = Math.max(90, h * 0.22);
    var trackTop = chartH + 26;
    var th = h - trackTop - 16;

    // --- chart
    drawChart(w, chartH, p);

    // --- track
    var xMin = -1.2, xMax = 0.6;
    var margin = 40;
    var tw = w - margin * 2;
    var baseY = trackTop + th - 20;

    function toX(x) { return margin + ((x - xMin) / (xMax - xMin)) * tw; }
    function toY(y) {
      // hill height in [-0.55, 0.9], scaled to sit below the chart
      var hh = 0.9 - y;
      return baseY - (hh / 1.15) * (th - 40);
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
      toY(S.mountainCarTrack(car.env.x + 0.01)) - toY(S.mountainCarTrack(car.env.x - 0.01)),
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
  }

  function drawChart(w, chartH, p) {
    var margin = 40;
    ctx.fillStyle = p.panel + "88";
    ctx.fillRect(margin, 8, w - margin * 2, chartH - 16);
    ctx.strokeStyle = p.chartGrid;
    ctx.beginPath();
    for (var i = 0; i <= 4; i++) {
      var y = 8 + ((chartH - 16) * i) / 4;
      ctx.moveTo(margin, y);
      ctx.lineTo(w - margin, y);
    }
    ctx.stroke();
    ctx.fillStyle = p.muted;
    ctx.font = "10px system-ui";
    ctx.textAlign = "left";
    ctx.fillText("episode reward (10-ep running avg)", margin + 6, 22);

    if (car.rewards.length < 2) {
      ctx.fillStyle = p.faint;
      ctx.textAlign = "center";
      ctx.font = "600 12px system-ui";
      ctx.fillText("training…", w / 2, chartH / 2 + 4);
      return;
    }
    var n = car.rewards.length;
    var data = [];
    for (var e = 0; e < n; e++) {
      var from = Math.max(0, e - 9);
      var slice = car.rewards.slice(from, e + 1);
      data.push(slice.reduce(function (a, b) { return a + b; }, 0) / slice.length);
    }
    var minR = Math.min.apply(null, data) - 5;
    var maxR = Math.max.apply(null, data) + 5;
    ctx.strokeStyle = p.chart;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var k = 0; k < data.length; k++) {
      var x = margin + (k / Math.max(1, n - 1)) * (w - margin * 2);
      var y = 8 + (chartH - 16) * (1 - (data[k] - minR) / (maxR - minR || 1));
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
  }

  // ---------------------------------------------------------------- input
  var keyMap = {
    ArrowUp: 3, KeyW: 3,
    ArrowDown: 4, KeyS: 4,
    ArrowLeft: 1, KeyA: 1,
    ArrowRight: 2, KeyD: 2,
  };

  document.addEventListener("keydown", function (e) {
    if (guideOpen) return;
    if (e.code === "Space") {
      e.preventDefault();
      if (tab === "maze" && maze.mode === "play" && !maze.over) {
        mazeAct(0);
        markKey("Space — wait");
      }
      return;
    }
    if (tab === "maze" && keyMap[e.code]) {
      e.preventDefault();
      mazeAct(keyMap[e.code]);
      markKey(e.code.replace("Key", "").replace("Arrow", "") + " — move");
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
        document.querySelectorAll(".tab").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        tab = btn.dataset.tab;
        $id("panel-title").textContent = tab === "maze" ? "🐂 Minotaur Maze" : "⛰️ Mountain Car";
        $id("maze-hud").classList.toggle("hidden", tab !== "maze");
        $id("car-hud").classList.toggle("hidden", tab !== "car");
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
        animateVI();
      }
    });
    $id("vi-gamma").addEventListener("input", function () {
      maze.viGamma = Number(this.value);
      $id("vi-gamma-v").textContent = this.value;
    });
    $id("vi-speed").addEventListener("input", function () {
      maze.viSpeed = Number(this.value);
      $id("vi-speed-v").textContent = this.value + " sweeps/s";
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
      var t = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", t);
      try { localStorage.setItem("theme", t); } catch (e) {}
      applyTheme();
      render();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (ev) {
      if (localStorage.getItem("theme")) return;
      document.documentElement.setAttribute("data-theme", ev.matches ? "dark" : "light");
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
  function init() {
    applyTheme();
    wireGuide();
    wire();
    sizeCanvas();
    mazeStart();
    maze.mdp = S.buildMazeMdp();
    log("🐂 Minotaur Maze loaded — arrows to move, reach the gold exit.");
    log("🧮 Value Iteration mode solves the full joint MDP (2240 states).");
    log("⛰️ Mountain Car tab: SARSA(λ) with Fourier features trains live.");
    render();
  }

  init();
})();
