// Point the sidebar language switcher at the *same* page in the other language.
// ReadTheDocs serves translations under /en/<version>/... and /zh-cn/<version>/...
// with identical relative paths (EN/ZH parity), so we just swap the language
// segment. Off ReadTheDocs (no language segment), the template's root links stay.
(function () {
  "use strict";

  function init() {
    var m = window.location.pathname.match(/^\/(en|zh-cn)\/([^/]+)\/(.*)$/);
    if (!m) return;
    var ver = m[2];
    var rest = m[3];
    var en = document.querySelector(".rlinf-lang-en");
    var zh = document.querySelector(".rlinf-lang-zh");
    if (en) en.setAttribute("href", "/en/" + ver + "/" + rest);
    if (zh) zh.setAttribute("href", "/zh-cn/" + ver + "/" + rest);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
