/* ---------------------------------------------------------------------------
 * MathJax bootstrap for Zensical pages using pymdownx.arithmatex.
 *
 * arithmatex emits inline/display math as \(...\) and \[...\]. This script
 * loads MathJax once, then re-typesets after instant-navigation page swaps.
 * ------------------------------------------------------------------------- */

(function () {
  "use strict";

  var SCRIPT_ID = "mathjax-script";

  window.MathJax = window.MathJax || {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };

  function ensureScriptLoaded() {
    if (document.getElementById(SCRIPT_ID)) return;
    var script = document.createElement("script");
    script.id = SCRIPT_ID;
    script.src = "https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js";
    script.async = true;
    document.head.appendChild(script);
  }

  function typesetIfReady() {
    if (!window.MathJax || !window.MathJax.startup) return;
    if (!window.MathJax.typesetPromise) return;
    window.MathJax.startup.output.clearCache();
    window.MathJax.typesetClear();
    window.MathJax.texReset();
    window.MathJax.typesetPromise();
  }

  ensureScriptLoaded();

  if (typeof window.document$ !== "undefined" && window.document$
      && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(function () {
      typesetIfReady();
    });
  } else {
    document.addEventListener("DOMContentLoaded", function () {
      typesetIfReady();
    });
  }
})();
