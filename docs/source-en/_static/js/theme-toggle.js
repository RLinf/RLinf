// Two-state (light / dark) color-mode toggle for the sidebar, replacing the
// pydata switcher's three-way light → dark → auto cycle. It writes the same
// localStorage keys / html data-attributes pydata reads, so the choice persists
// and the theme's early inline script applies it on the next load.
(function () {
  "use strict";

  function setMode(mode) {
    var el = document.documentElement;
    el.dataset.mode = mode;
    el.dataset.theme = mode;
    try {
      localStorage.setItem("mode", mode);
      localStorage.setItem("theme", mode);
    } catch (e) {
      /* storage unavailable; ignore */
    }
  }

  function init() {
    var btn = document.querySelector(".rlinf-theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var dark = document.documentElement.dataset.theme === "dark";
      setMode(dark ? "light" : "dark");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
