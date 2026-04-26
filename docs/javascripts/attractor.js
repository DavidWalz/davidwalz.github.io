/* ---------------------------------------------------------------------------
 * Inference Landscape — hero canvas renderer.
 *
 * Draws a 2D probability surface (a sum of slowly drifting Gaussians)
 * sampled on a grid and rendered as a stack of horizontal "contour ribbons".
 * Each ribbon is a polyline whose vertical offset is modulated by the local
 * field value, giving a luminous topographic look.
 *
 * The animation respects `prefers-reduced-motion`, pauses while the tab is
 * hidden, and tears itself down cleanly when the page is swapped via
 * Zensical's instant navigation.
 * ------------------------------------------------------------------------- */

(function () {
  "use strict";

  function setBodyPalette(color) {
    if (!color) return;
    for (var key in color) {
      if (Object.prototype.hasOwnProperty.call(color, key) && color[key] != null) {
        document.body.setAttribute("data-md-color-" + key, color[key]);
      }
    }
  }

  function getStoredPalette() {
    try {
      if (typeof window.__md_get === "function") {
        return window.__md_get("__palette");
      }
      // Fallback when helper is unavailable.
      var scope = new URL(".", window.location).pathname;
      var raw = localStorage.getItem(scope + ".__palette");
      return raw ? JSON.parse(raw) : null;
    } catch (_err) {
      return null;
    }
  }

  function restoreNativePalette() {
    var palette = getStoredPalette();
    if (palette && palette.color) {
      // Mirror theme startup behavior for media-driven palettes.
      if (palette.color.media === "(prefers-color-scheme)") {
        var media = window.matchMedia("(prefers-color-scheme: light)");
        var selector = media.matches
          ? "[data-md-color-media='(prefers-color-scheme: light)']"
          : "[data-md-color-media='(prefers-color-scheme: dark)']";
        var input = document.querySelector(selector);
        if (input) {
          palette.color.media = input.getAttribute("data-md-color-media");
          palette.color.scheme = input.getAttribute("data-md-color-scheme");
          palette.color.primary = input.getAttribute("data-md-color-primary");
          palette.color.accent = input.getAttribute("data-md-color-accent");
        }
      }
      setBodyPalette(palette.color);
      return;
    }

    // No stored palette: fall back to declared default option.
    var defaultInput = document.querySelector("[data-md-color-scheme='default']");
    if (defaultInput) {
      setBodyPalette({
        media: defaultInput.getAttribute("data-md-color-media") || "none",
        scheme: defaultInput.getAttribute("data-md-color-scheme") || "default",
        primary: defaultInput.getAttribute("data-md-color-primary"),
        accent: defaultInput.getAttribute("data-md-color-accent")
      });
    } else {
      document.body.setAttribute("data-md-color-scheme", "default");
    }
  }

  function applyHomeColorScheme() {
    var path = window.location.pathname || "/";
    var isHome = path === "/" || path.endsWith("/index.html");
    if (isHome) {
      document.body.setAttribute("data-md-color-scheme", "slate");
      return;
    }
    restoreNativePalette();
  }

  var SYSTEMS = {
    "inference-landscape": {
      kind: "custom",
      trailFade: 0.10,
      cols: 96,
      rows: 32,
      ribbons: 32,
      _t: 0,
      _field: null,
      _modes: [
        { bx: -0.55, by:  0.10, ax: 0.30, ay: 0.20, fx: 0.17, fy: 0.23, sigma: 0.34, w: 1.0 },
        { bx:  0.50, by: -0.20, ax: 0.28, ay: 0.18, fx: 0.21, fy: 0.13, sigma: 0.30, w: 0.9 },
        { bx:  0.10, by:  0.45, ax: 0.26, ay: 0.22, fx: 0.13, fy: 0.27, sigma: 0.28, w: 0.7 },
        { bx: -0.25, by: -0.55, ax: 0.30, ay: 0.18, fx: 0.19, fy: 0.17, sigma: 0.32, w: 0.6 }
      ],
      setup: function () {
        this._field = new Float32Array(this.cols * this.rows);
        this._t = 0;
      },
      update: function (dt) { this._t += dt; },
      draw: function (ctx, w, h) {
        var t = this._t;
        var cols = this.cols, rows = this.rows;
        var modes = this._modes;
        var nM = modes.length;
        var field = this._field;

        var mxs = this._mxs || (this._mxs = new Float64Array(nM));
        var mys = this._mys || (this._mys = new Float64Array(nM));
        var s2s = this._s2s || (this._s2s = new Float64Array(nM));
        var ws  = this._ws  || (this._ws  = new Float64Array(nM));
        for (var i = 0; i < nM; i++) {
          var m = modes[i];
          mxs[i] = m.bx + m.ax * Math.sin(m.fx * t);
          mys[i] = m.by + m.ay * Math.cos(m.fy * t);
          s2s[i] = m.sigma * m.sigma;
          ws[i]  = m.w;
        }

        var maxV = 0.0001;
        var idx = 0;
        for (var ry = 0; ry < rows; ry++) {
          var fy = -0.95 + (1.9 * ry) / (rows - 1);
          for (var cx = 0; cx < cols; cx++) {
            var fx = -1.4 + (2.8 * cx) / (cols - 1);
            var v = 0;
            for (var k = 0; k < nM; k++) {
              var ddx = fx - mxs[k];
              var ddy = fy - mys[k];
              v += ws[k] * Math.exp(
                -(ddx * ddx + ddy * ddy) / (2 * s2s[k]));
            }
            field[idx++] = v;
            if (v > maxV) maxV = v;
          }
        }
        var invMax = 1 / maxV;

        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = "rgba(7, 8, 12, " + this.trailFade + ")";
        ctx.fillRect(0, 0, w, h);

        ctx.globalCompositeOperation = "lighter";
        ctx.lineWidth = 1.25;

        var nR = this.ribbons;
        var lift = h * 0.18;
        var marginY = h * 0.08;
        var usableH = h - marginY * 2;
        var stepX = w / (cols - 1);

        for (var r = 0; r < nR; r++) {
          var gy = (r / (nR - 1)) * (rows - 1);
          var gy0 = Math.floor(gy);
          var gy1 = Math.min(rows - 1, gy0 + 1);
          var gyt = gy - gy0;
          var baseY = marginY + (r / (nR - 1)) * usableH;

          var avg = 0;
          ctx.beginPath();
          for (var c = 0; c < cols; c++) {
            var v0 = field[gy0 * cols + c];
            var v1 = field[gy1 * cols + c];
            var v  = (v0 + (v1 - v0) * gyt) * invMax;
            avg += v;
            var x = c * stepX;
            var y = baseY - v * lift;
            if (c === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          avg /= cols;

          var hue = 200 - avg * 110;
          var light = 55 + avg * 25;
          var alpha = 0.18 + avg * 0.55;
          ctx.strokeStyle = "hsla(" + hue.toFixed(0) + ", 85%, "
            + light.toFixed(0) + "%, " + alpha.toFixed(2) + ")";
          ctx.stroke();
        }
      }
    }
  };

  function start(canvas, systemName) {
    if (!canvas || canvas.dataset.attractorBound === "1") return null;
    canvas.dataset.attractorBound = "1";

    var system = SYSTEMS[systemName];
    if (!system) return null;

    var ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return null;

    var dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    var width = 0, height = 0;

    function resize() {
      var rect = canvas.getBoundingClientRect();
      width = Math.max(1, Math.floor(rect.width));
      height = Math.max(1, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#07080c";
      ctx.fillRect(0, 0, width, height);
      if (typeof system.setup === "function") system.setup(ctx, width, height);
    }
    resize();

    var ro = ("ResizeObserver" in window) ? new ResizeObserver(resize) : null;
    if (ro) ro.observe(canvas);
    else window.addEventListener("resize", resize);

    var reducedMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    var rafId = 0;
    var running = true;

    function frame(ts) {
      rafId = 0;
      if (!running) return;
      if (!reducedMotion) system.update(1 / 60);
      system.draw(ctx, width, height, ts / 1000);
      rafId = window.requestAnimationFrame(frame);
    }

    function play() {
      if (rafId || !running) return;
      rafId = window.requestAnimationFrame(frame);
    }
    function pause() {
      if (rafId) { window.cancelAnimationFrame(rafId); rafId = 0; }
    }
    function onVisibility() { if (document.hidden) pause(); else play(); }
    document.addEventListener("visibilitychange", onVisibility);

    play();

    return function destroy() {
      running = false;
      pause();
      document.removeEventListener("visibilitychange", onVisibility);
      if (ro) ro.disconnect();
      else window.removeEventListener("resize", resize);
      delete canvas.dataset.attractorBound;
    };
  }

  var teardowns = [];
  function boot() {
    applyHomeColorScheme();
    while (teardowns.length) teardowns.pop()();
    var canvases = document.querySelectorAll("[data-visual-system]");
    for (var index = 0; index < canvases.length; index++) {
      var canvas = canvases[index];
      var teardown = start(canvas, canvas.dataset.visualSystem);
      if (teardown) teardowns.push(teardown);
    }
  }

  if (typeof window !== "undefined" && typeof window.document$ !== "undefined"
      && window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(boot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
