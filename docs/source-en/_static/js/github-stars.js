// Show the live GitHub star count next to the repository icon in the sidebar,
// mirroring the LeRobot docs. The count is cached in localStorage so we don't
// hit the GitHub API (60 req/h unauthenticated) on every page load.
(function () {
  "use strict";

  var REPO = "RLinf/RLinf";
  var CACHE_KEY = "rlinf-github-stars";
  var CACHE_TTL = 6 * 60 * 60 * 1000; // 6 hours

  function render(count) {
    if (typeof count !== "number") return;
    var links = document.querySelectorAll(
      '.rlinf-sidebar-tools a[href*="github.com/' + REPO + '"]'
    );
    links.forEach(function (link) {
      if (link.querySelector(".rlinf-gh-stars")) return;
      var badge = document.createElement("span");
      badge.className = "rlinf-gh-stars";
      badge.textContent = count.toLocaleString("en-US");
      link.appendChild(badge);
      link.classList.add("rlinf-gh-link");
    });
  }

  function fetchStars() {
    fetch("https://api.github.com/repos/" + REPO)
      .then(function (resp) {
        return resp.ok ? resp.json() : null;
      })
      .then(function (data) {
        if (!data || typeof data.stargazers_count !== "number") return;
        render(data.stargazers_count);
        try {
          localStorage.setItem(
            CACHE_KEY,
            JSON.stringify({ count: data.stargazers_count, time: Date.now() })
          );
        } catch (e) {
          /* storage unavailable; ignore */
        }
      })
      .catch(function () {
        /* offline or rate-limited; leave the icon as-is */
      });
  }

  function init() {
    try {
      var cached = JSON.parse(localStorage.getItem(CACHE_KEY) || "null");
      if (cached && typeof cached.count === "number") {
        render(cached.count); // show last-known count immediately
        if (Date.now() - cached.time < CACHE_TTL) return; // still fresh
      }
    } catch (e) {
      /* malformed cache; fall through to fetch */
    }
    fetchStars();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
