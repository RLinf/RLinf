// Persist manually expanded sidebar branches in sessionStorage so multiple
// sections can stay open simultaneously across page navigations, and only
// adjust sidebar scroll if the active link falls outside the viewport.
(function () {
  "use strict";

  var STORAGE_KEY = "rlinf-sidebar-open-v1";

  function detailKey(details) {
    var li = details.parentElement;
    if (!li) return null;
    var link = li.querySelector(":scope > a[href]");
    if (!link) return null;
    try {
      var url = new URL(link.getAttribute("href"), window.location.href);
      return url.pathname.replace(/\/index\.html$/, "/");
    } catch (_err) {
      return link.getAttribute("href");
    }
  }

  function loadOpenSet() {
    try {
      var raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return new Set();
      var parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? new Set(parsed) : new Set();
    } catch (_err) {
      return new Set();
    }
  }

  function saveOpenSet(openSet) {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(openSet)));
    } catch (_err) {
      // Ignore storage quota or private-mode restrictions.
    }
  }

  function scrollableAncestor(el) {
    for (var p = el.parentElement; p; p = p.parentElement) {
      var oy = getComputedStyle(p).overflowY;
      if ((oy === "auto" || oy === "scroll") && p.scrollHeight > p.clientHeight) {
        return p;
      }
    }
    return null;
  }

  function ensureCurrentVisible(nav) {
    var current = nav.querySelector("a.current");
    if (!current) return;
    var box = scrollableAncestor(current);
    if (!box) return;
    var boxRect = box.getBoundingClientRect();
    var curRect = current.getBoundingClientRect();
    if (curRect.top < boxRect.top + 16 || curRect.bottom > boxRect.bottom - 16) {
      box.scrollTop +=
        curRect.top - boxRect.top - box.clientHeight / 2 + curRect.height / 2;
    }
  }

  function init() {
    var nav = document.querySelector(".bd-sidebar-primary .bd-docs-nav");
    if (!nav) return;

    var openSet = loadOpenSet();
    var allDetails = nav.querySelectorAll("li.has-children > details");
    for (var i = 0; i < allDetails.length; i++) {
      var d = allDetails[i];
      var key = detailKey(d);
      if (!key) continue;
      if (openSet.has(key)) {
        d.open = true;
      } else if (d.open) {
        openSet.add(key);
      }
    }
    saveOpenSet(openSet);

    // Use capture phase because `toggle` does not bubble.
    nav.addEventListener(
      "toggle",
      function (event) {
        var target = event.target;
        if (!target || target.tagName !== "DETAILS") return;
        var key = detailKey(target);
        if (!key) return;
        if (target.open) {
          openSet.add(key);
        } else {
          openSet.delete(key);
        }
        saveOpenSet(openSet);
      },
      true
    );

    requestAnimationFrame(function () {
      ensureCurrentVisible(nav);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
