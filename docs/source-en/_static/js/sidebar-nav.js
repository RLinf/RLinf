// Scroll the active sidebar entry into view on page load. Top-level section
// captions are always visible via Sphinx :caption: groups (LeRobot style),
// while collapsible sub-trees stay closed unless they contain the active page.
(function () {
  "use strict";

  function scrollableAncestor(el) {
    for (var p = el.parentElement; p; p = p.parentElement) {
      var oy = getComputedStyle(p).overflowY;
      if ((oy === "auto" || oy === "scroll") && p.scrollHeight > p.clientHeight) {
        return p;
      }
    }
    return null;
  }

  function revealCurrent(nav) {
    var current = nav.querySelector("a.current");
    if (!current) return;
    var box = scrollableAncestor(current);
    if (!box) return;
    // Center the active entry within the sidebar without scrolling the page.
    var boxRect = box.getBoundingClientRect();
    var curRect = current.getBoundingClientRect();
    box.scrollTop +=
      curRect.top - boxRect.top - box.clientHeight / 2 + curRect.height / 2;
  }

  function init() {
    var nav = document.querySelector(".bd-docs-nav");
    if (!nav) return;
    revealCurrent(nav);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
