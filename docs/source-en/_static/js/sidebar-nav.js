// Expand every top-level navigation section by default, like the LeRobot docs
// where the section groups land open. The pydata theme only opens the branch
// containing the active page, so here we open all first-level sections (their
// immediate children show; deeper sub-trees stay collapsed). Because the
// expanded sidebar is tall, also scroll the current page's entry into view so a
// fresh load / refresh does not leave the sidebar parked at the top.
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
    nav.querySelectorAll("li.toctree-l1.has-children > details").forEach(
      function (section) {
        section.setAttribute("open", "");
      }
    );
    revealCurrent(nav);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
