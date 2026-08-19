/**
 * sprites.js — canvas-drawn character sprites for the maze (Thomas & Minotaur).
 * Pure drawing helpers: drawThomas(ctx, cx, cy, size), drawMinotaur(ctx, cx, cy, size).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Sprites = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var INK = "#0F172A";

  /** Thomas: little adventurer with a fedora — head, hat, body. */
  function drawThomas(ctx, cx, cy, s) {
    ctx.save();
    ctx.translate(cx, cy);
    // body (torso)
    ctx.fillStyle = INK;
    roundRect(ctx, -s * 0.3, s * 0.12, s * 0.6, s * 0.42, s * 0.14);
    ctx.fill();
    // legs
    ctx.fillRect(-s * 0.24, s * 0.5, s * 0.16, s * 0.22);
    ctx.fillRect(s * 0.08, s * 0.5, s * 0.16, s * 0.22);
    // head
    ctx.beginPath();
    ctx.arc(0, -s * 0.1, s * 0.24, 0, Math.PI * 2);
    ctx.fill();
    // fedora brim
    ctx.beginPath();
    ctx.ellipse(0, -s * 0.28, s * 0.34, s * 0.1, 0, 0, Math.PI * 2);
    ctx.fill();
    // fedora crown
    ctx.beginPath();
    ctx.moveTo(-s * 0.16, -s * 0.28);
    ctx.lineTo(-s * 0.13, -s * 0.5);
    ctx.lineTo(s * 0.13, -s * 0.5);
    ctx.lineTo(s * 0.16, -s * 0.28);
    ctx.closePath();
    ctx.fill();
    // eyes (2 white dots)
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(-s * 0.09, -s * 0.1, s * 0.045, 0, Math.PI * 2);
    ctx.arc(s * 0.09, -s * 0.1, s * 0.045, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  /** Minotaur: bull head with horns. */
  function drawMinotaur(ctx, cx, cy, s) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.fillStyle = INK;
    // horns (two curved)
    ctx.lineWidth = s * 0.12;
    ctx.lineCap = "round";
    ctx.strokeStyle = INK;
    ctx.beginPath();
    ctx.moveTo(-s * 0.3, -s * 0.22);
    ctx.quadraticCurveTo(-s * 0.55, -s * 0.38, -s * 0.46, -s * 0.62);
    ctx.moveTo(s * 0.3, -s * 0.22);
    ctx.quadraticCurveTo(s * 0.55, -s * 0.38, s * 0.46, -s * 0.62);
    ctx.stroke();
    // head (rounded)
    ctx.fillStyle = INK;
    roundRect(ctx, -s * 0.36, -s * 0.3, s * 0.72, s * 0.52, s * 0.24);
    ctx.fill();
    // ears
    ctx.beginPath();
    ctx.moveTo(-s * 0.36, -s * 0.18);
    ctx.lineTo(-s * 0.5, -s * 0.3);
    ctx.lineTo(-s * 0.32, -s * 0.34);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(s * 0.36, -s * 0.18);
    ctx.lineTo(s * 0.5, -s * 0.3);
    ctx.lineTo(s * 0.32, -s * 0.34);
    ctx.closePath();
    ctx.fill();
    // muzzle (lighter)
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.ellipse(0, s * 0.06, s * 0.2, s * 0.14, 0, 0, Math.PI * 2);
    ctx.fill();
    // nostrils
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(-s * 0.07, s * 0.08, s * 0.035, 0, Math.PI * 2);
    ctx.arc(s * 0.07, s * 0.08, s * 0.035, 0, Math.PI * 2);
    ctx.fill();
    // eyes
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(-s * 0.14, -s * 0.1, s * 0.06, 0, Math.PI * 2);
    ctx.arc(s * 0.14, -s * 0.1, s * 0.06, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  return { drawThomas: drawThomas, drawMinotaur: drawMinotaur };
});
