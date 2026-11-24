// Sidebar persistence + instant PJAX navigation (prefetch on hover, swap only
// <main id="main-content"> so the sidebar, header, scroll position, and open
// branches stay rock-solid with zero full-page reload).
(function () {
  "use strict";

  var STORAGE_KEY = "rlinf-sidebar-open-v1";
  var COLLAPSED_KEY = "rlinf-sidebar-collapsed-v1";
  var TAB_STORAGE_PREFIX = "sphinx-design-tab-id-";
  var MAX_CACHE_ENTRIES = 30;
  var PREFETCH_DELAY_MS = 50;
  var PROGRESS_DELAY_MS = 120;

  var NON_HTML_EXT_RE =
    /\.(?:pdf|png|jpe?g|gif|webp|svg|ico|mp4|webm|mov|zip|tar|gz|tgz|bz2|xz|ya?ml|json|py|sh|txt|bib|csv|rst|md)$/i;

  var COPY_ICON_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">' +
    '<path stroke="none" d="M0 0h24v24H0z" fill="none"/>' +
    '<rect x="8" y="8" width="12" height="12" rx="2" />' +
    '<path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2" />' +
    "</svg>";

  var pageCache = new Map();
  var inflight = new Map();
  var navController = null;
  var progressTimer = null;
  var progressBar = null;
  var codeCellCounter = 0;
  var docRootPath = null;
  var openSet = null;
  var collapsedSet = null;
  var lastLoadedKey = null;

  function canonicalPath(href, base) {
    try {
      var u = new URL(href, base || window.location.href);
      return u.pathname.replace(/\/index\.html$/, "/");
    } catch (_err) {
      return href;
    }
  }

  function computeDocRootPath() {
    var rootRel = document.documentElement.getAttribute("data-content_root");
    if (typeof rootRel !== "string" || !rootRel) {
      var docOpts =
        typeof DOCUMENTATION_OPTIONS !== "undefined"
          ? DOCUMENTATION_OPTIONS
          : window.DOCUMENTATION_OPTIONS;
      rootRel =
        docOpts && typeof docOpts.URL_ROOT === "string" ? docOpts.URL_ROOT : "./";
    }
    if (rootRel === "") rootRel = "./";
    try {
      var u = new URL(rootRel, window.location.href);
      var p = u.pathname;
      if (!p.endsWith("/")) p += "/";
      return p;
    } catch (_err) {
      return "/";
    }
  }

  function detailKey(details) {
    var li = details.parentElement;
    if (!li) return null;
    var link = li.querySelector(":scope > a[href]");
    if (!link) return null;
    return canonicalPath(link.getAttribute("href"));
  }

  function loadStoredSet(key) {
    try {
      var raw = sessionStorage.getItem(key);
      if (!raw) return new Set();
      var parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? new Set(parsed) : new Set();
    } catch (_err) {
      return new Set();
    }
  }

  function saveStoredSet(key, set) {
    try {
      sessionStorage.setItem(key, JSON.stringify(Array.from(set)));
    } catch (_err) {
      // Ignore storage quota or private-mode restrictions.
    }
  }

  function loadOpenSet() {
    return loadStoredSet(STORAGE_KEY);
  }

  function saveOpenSet(set) {
    saveStoredSet(STORAGE_KEY, set);
  }

  function setSectionCollapsed(cap, ul, collapsed, persist) {
    if (!cap || !ul) return;
    var winY = window.scrollY;
    var box = scrollableAncestor(cap);
    var boxY = box ? box.scrollTop : 0;

    cap.classList.toggle("is-collapsed", collapsed);
    ul.classList.toggle("rlinf-section-collapsed", collapsed);

    var ctrl = cap.querySelector(".rlinf-caption-toggle, .rlinf-caption-btn");
    if (ctrl) {
      ctrl.setAttribute("aria-expanded", collapsed ? "false" : "true");
    }

    if (persist && collapsedSet) {
      var key = cap.getAttribute("data-rlinf-section-key");
      if (key) {
        if (collapsed) {
          collapsedSet.add(key);
        } else {
          collapsedSet.delete(key);
        }
        saveStoredSet(COLLAPSED_KEY, collapsedSet);
      }
    }

    if (window.scrollY !== winY) {
      window.scrollTo(window.scrollX, winY);
    }
    if (box && box.scrollTop !== boxY) {
      box.scrollTop = boxY;
    }
  }

  // Turn Level-1 sidebar captions (<p class="caption">) into LeRobot-style
  // collapsible chapter headers (default expanded). Chapters whose first child
  // is "Overview" / "总览" merge that link directly into the clickable heading
  // and hide the redundant child row.
  function enhanceSectionCaptions(container) {
    if (!container) return;
    var currentKey = canonicalPath(window.location.href);
    var captions = container.querySelectorAll(".bd-docs-nav p.caption");
    for (var i = 0; i < captions.length; i++) {
      var cap = captions[i];
      if (cap.classList.contains("rlinf-caption-collapsible")) continue;
      var ul = cap.nextElementSibling;
      if (!ul || ul.tagName !== "UL") continue;

      var titleText = (cap.textContent || "").trim();
      var firstLi = ul.querySelector(":scope > li:first-child");
      var firstLink = firstLi ? firstLi.querySelector(":scope > a[href]") : null;
      var firstText = firstLink ? (firstLink.textContent || "").trim() : "";
      var isOverview = Boolean(
        firstLink &&
          !firstLi.classList.contains("has-children") &&
          (firstText === "Overview" || firstText === "总览")
      );

      var secKey = firstLink
        ? canonicalPath(firstLink.getAttribute("href"))
        : "cap:" + titleText;
      cap.classList.add("rlinf-caption-collapsible");
      cap.setAttribute("data-rlinf-section-key", secKey);

      var hasActiveInside = Boolean(ul.querySelector("a.current"));

      if (isOverview) {
        var overviewHref = firstLink.getAttribute("href");
        var isCurrentOverview = canonicalPath(overviewHref) === currentKey;
        firstLi.classList.add("rlinf-overview-merged");
        firstLi.classList.remove("current", "active");
        firstLink.classList.remove("current");

        var a = document.createElement("a");
        a.className = "rlinf-caption-link";
        a.setAttribute("href", overviewHref);
        if (isCurrentOverview) {
          a.classList.add("current");
          hasActiveInside = true;
        }
        while (cap.firstChild) {
          a.appendChild(cap.firstChild);
        }

        var toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "rlinf-caption-toggle";
        toggle.setAttribute("aria-expanded", "true");
        toggle.setAttribute("aria-label", "Toggle " + titleText);
        toggle.innerHTML = '<i class="fa-solid fa-chevron-down" aria-hidden="true"></i>';

        cap.appendChild(a);
        cap.appendChild(toggle);
      } else {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "rlinf-caption-btn";
        btn.setAttribute("aria-expanded", "true");
        while (cap.firstChild) {
          btn.appendChild(cap.firstChild);
        }
        var chev = document.createElement("span");
        chev.className = "rlinf-caption-chevron";
        chev.setAttribute("aria-hidden", "true");
        chev.innerHTML = '<i class="fa-solid fa-chevron-down"></i>';
        btn.appendChild(chev);
        cap.appendChild(btn);
      }

      if (collapsedSet && collapsedSet.has(secKey) && !hasActiveInside) {
        setSectionCollapsed(cap, ul, true, false);
      }
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

  // Convert relative href/src attributes to root-relative paths so persistent
  // chrome (sidebar, header) and swapped <main> work across directory depths.
  // When resolveSelfHash is true, Sphinx's current-page `href="#"` link in the
  // sidebar is rewritten to the initial page's pathname so pushState does not
  // later make it falsely match every navigated URL.
  function rebaseUrls(container, baseUrl, resolveSelfHash) {
    if (!container) return;

    var links = container.querySelectorAll("a[href]");
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var raw = a.getAttribute("href");
      if (!raw || /^(?:mailto:|tel:|javascript:|data:)/i.test(raw)) {
        continue;
      }
      if (raw === "#") {
        if (!resolveSelfHash) continue;
        try {
          var selfUrl = new URL(baseUrl);
          a.setAttribute("href", selfUrl.pathname);
        } catch (_err) {
          // Ignore invalid baseUrl.
        }
        continue;
      }
      if (raw.charAt(0) === "#") {
        continue;
      }
      try {
        var u = new URL(raw, baseUrl);
        if (u.origin === window.location.origin) {
          a.setAttribute("href", u.pathname + u.search + u.hash);
        }
      } catch (_err) {
        // Leave malformed href untouched.
      }
    }

    var srcAttrs = [
      ["img[src]", "src"],
      ["source[src]", "src"],
      ["video[src]", "src"],
      ["video[poster]", "poster"],
      ["audio[src]", "src"],
      ["iframe[src]", "src"],
      ["object[data]", "data"],
    ];
    for (var s = 0; s < srcAttrs.length; s++) {
      var els = container.querySelectorAll(srcAttrs[s][0]);
      var attr = srcAttrs[s][1];
      for (var j = 0; j < els.length; j++) {
        var el = els[j];
        var val = el.getAttribute(attr);
        if (!val || /^(?:data:|blob:|https?:|\/\/)/i.test(val)) continue;
        try {
          var su = new URL(val, baseUrl);
          el.setAttribute(
            attr,
            su.origin === window.location.origin
              ? su.pathname + su.search + su.hash
              : su.href
          );
        } catch (_err) {
          // Ignore invalid URL.
        }
      }
    }

    var srcsets = container.querySelectorAll("img[srcset], source[srcset]");
    for (var k = 0; k < srcsets.length; k++) {
      var ss = srcsets[k].getAttribute("srcset");
      if (!ss) continue;
      var rewritten = ss
        .split(",")
        .map(function (part) {
          var trimmed = part.trim();
          if (!trimmed) return "";
          var tokens = trimmed.split(/\s+/);
          var srcUrl = tokens[0];
          if (!/^(?:data:|blob:|https?:|\/\/)/i.test(srcUrl)) {
            try {
              var u = new URL(srcUrl, baseUrl);
              tokens[0] =
                u.origin === window.location.origin
                  ? u.pathname + u.search + u.hash
                  : u.href;
            } catch (_err) {
              // Keep original token.
            }
          }
          return tokens.join(" ");
        })
        .join(", ");
      srcsets[k].setAttribute("srcset", rewritten);
    }
  }

  function isPjaxEligibleLink(link) {
    if (window.location.protocol === "file:") return false;
    if (!link || link.tagName !== "A") return false;
    if (link.hasAttribute("download")) return false;
    var target = link.getAttribute("target");
    if (target && target !== "_self") return false;
    if (link.classList.contains("external")) return false;
    var rel = link.getAttribute("rel");
    if (rel && /\bexternal\b/i.test(rel)) return false;

    var rawHref = link.getAttribute("href");
    if (
      !rawHref ||
      rawHref.charAt(0) === "#" ||
      /^(?:mailto:|tel:|javascript:)/i.test(rawHref)
    ) {
      return false;
    }

    var url;
    try {
      url = new URL(rawHref, window.location.href);
    } catch (_err) {
      return false;
    }
    if (url.origin !== window.location.origin) return false;
    if (docRootPath && url.pathname.indexOf(docRootPath) !== 0) return false;

    var relPath = docRootPath ? url.pathname.slice(docRootPath.length) : url.pathname;
    if (
      /^(?:_static|_images|_sources|_downloads)\//.test(relPath) ||
      relPath === "search.html" ||
      relPath === "genindex.html"
    ) {
      return false;
    }
    if (NON_HTML_EXT_RE.test(url.pathname)) return false;

    return true;
  }

  function isSamePage(url) {
    return (
      canonicalPath(url.href) === canonicalPath(window.location.href) &&
      url.search === window.location.search
    );
  }

  function setCache(key, html) {
    if (pageCache.has(key)) {
      pageCache.delete(key);
    } else if (pageCache.size >= MAX_CACHE_ENTRIES) {
      var oldest = pageCache.keys().next().value;
      pageCache.delete(oldest);
    }
    pageCache.set(key, html);
  }

  function fetchPageHtml(urlStr, signal) {
    var cacheKey = canonicalPath(urlStr);
    if (pageCache.has(cacheKey)) {
      return Promise.resolve({ html: pageCache.get(cacheKey), finalUrl: urlStr });
    }
    if (!signal && inflight.has(cacheKey)) {
      return inflight.get(cacheKey);
    }
    var req = fetch(urlStr, {
      credentials: "same-origin",
      headers: { Accept: "text/html,application/xhtml+xml" },
      signal: signal,
    }).then(function (res) {
      if (!res.ok) throw new Error("HTTP " + res.status);
      var ct = res.headers.get("content-type") || "";
      if (ct && ct.indexOf("text/html") === -1) {
        throw new Error("Non-HTML response");
      }
      var finalUrl = res.url || urlStr;
      var parsedFinal = new URL(finalUrl, window.location.href);
      if (
        parsedFinal.origin !== window.location.origin ||
        (docRootPath && parsedFinal.pathname.indexOf(docRootPath) !== 0)
      ) {
        throw new Error("Cross-root redirect");
      }
      return res.text().then(function (html) {
        setCache(canonicalPath(finalUrl), html);
        return { html: html, finalUrl: finalUrl };
      });
    });

    if (!signal) {
      inflight.set(cacheKey, req);
      req.then(
        function () {
          inflight.delete(cacheKey);
        },
        function () {
          inflight.delete(cacheKey);
        }
      );
    }
    return req;
  }

  function showProgress() {
    if (progressTimer) clearTimeout(progressTimer);
    progressTimer = setTimeout(function () {
      if (!progressBar) {
        progressBar = document.createElement("div");
        progressBar.id = "rlinf-pjax-bar";
        document.body.appendChild(progressBar);
      }
      progressBar.classList.remove("is-done");
      progressBar.classList.add("is-loading");
    }, PROGRESS_DELAY_MS);
  }

  function hideProgress() {
    if (progressTimer) {
      clearTimeout(progressTimer);
      progressTimer = null;
    }
    if (progressBar && progressBar.classList.contains("is-loading")) {
      progressBar.classList.remove("is-loading");
      progressBar.classList.add("is-done");
      setTimeout(function () {
        if (progressBar) progressBar.classList.remove("is-done");
      }, 220);
    }
  }

  function updateSidebarActive(nav, targetUrl, opts) {
    if (!nav) return;
    var targetKey = canonicalPath(targetUrl);

    var prevActive = nav.querySelectorAll(".current, .active");
    for (var i = 0; i < prevActive.length; i++) {
      prevActive[i].classList.remove("current", "active");
    }

    var captionLinks = nav.querySelectorAll("p.caption > a.rlinf-caption-link");
    for (var c = 0; c < captionLinks.length; c++) {
      var chref = captionLinks[c].getAttribute("href");
      if (chref && canonicalPath(chref) === targetKey) {
        captionLinks[c].classList.add("current");
        var capEl = captionLinks[c].closest("p.caption");
        var capUl = capEl ? capEl.nextElementSibling : null;
        if (capEl && capUl && capEl.classList.contains("is-collapsed")) {
          setSectionCollapsed(capEl, capUl, false, true);
        }
        if (!opts || !opts.fromSidebar) {
          requestAnimationFrame(function () {
            ensureCurrentVisible(nav);
          });
        }
        return;
      }
    }

    var matched =
      opts &&
      opts.clickedLink &&
      !opts.clickedLink.classList.contains("rlinf-caption-link") &&
      nav.contains(opts.clickedLink)
        ? opts.clickedLink
        : null;
    if (!matched) {
      var links = nav.querySelectorAll("li:not(.rlinf-overview-merged) a[href]");
      for (var j = 0; j < links.length; j++) {
        var href = links[j].getAttribute("href");
        if (!href || href.charAt(0) === "#") continue;
        if (canonicalPath(href) === targetKey) {
          matched = links[j];
          break;
        }
      }
    }
    if (!matched) return;

    matched.classList.add("current");
    var li = matched.closest("li");
    if (li) {
      li.classList.add("current", "active");
      var ownDetails = li.querySelector(":scope > details");
      if (ownDetails && !ownDetails.open) {
        ownDetails.open = true;
        var ownKey = detailKey(ownDetails);
        if (ownKey && openSet) openSet.add(ownKey);
      }
    }

    var parent = li ? li.parentElement : null;
    while (parent && nav.contains(parent)) {
      if (parent.tagName === "DETAILS") {
        parent.open = true;
        var dk = detailKey(parent);
        if (dk && openSet) openSet.add(dk);
      } else if (parent.tagName === "LI") {
        parent.classList.add("current", "active");
      } else if (
        parent.tagName === "UL" &&
        parent.classList.contains("rlinf-section-collapsed")
      ) {
        var secCap = parent.previousElementSibling;
        if (secCap && secCap.tagName === "P") {
          setSectionCollapsed(secCap, parent, false, true);
        }
      }
      parent = parent.parentElement;
    }

    if (openSet) saveOpenSet(openSet);
    if (!opts || !opts.fromSidebar) {
      requestAnimationFrame(function () {
        ensureCurrentVisible(nav);
      });
    }
  }

  function updateLanguageSwitcher() {
    var m = window.location.pathname.match(/^\/(en|zh-cn)\/([^/]+)\/(.*)$/);
    if (!m) return;
    var ver = m[2];
    var rest = m[3];
    var en = document.querySelector(".rlinf-lang-en");
    var zh = document.querySelector(".rlinf-lang-zh");
    if (en) en.setAttribute("href", "/en/" + ver + "/" + rest);
    if (zh) zh.setAttribute("href", "/zh-cn/" + ver + "/" + rest);
  }

  function updateDocumentationOptions(targetUrl) {
    if (!docRootPath) return;
    try {
      var u = new URL(targetUrl, window.location.href);
      var rel = u.pathname.slice(docRootPath.length);
      var pagename = rel.replace(/\.html$/, "").replace(/\/$/, "/index");
      if (!pagename || pagename === "/index") pagename = "index";
      var depth = (rel.match(/\//g) || []).length;
      var relRoot = depth > 0 ? new Array(depth + 1).join("../") : "./";
      document.documentElement.setAttribute("data-content_root", relRoot);
      var docOpts =
        typeof DOCUMENTATION_OPTIONS !== "undefined"
          ? DOCUMENTATION_OPTIONS
          : window.DOCUMENTATION_OPTIONS;
      if (docOpts) {
        docOpts.pagename = pagename;
        docOpts.URL_ROOT = depth > 0 ? relRoot : "";
      }
    } catch (_err) {
      // Non-fatal.
    }
  }

  function initCopyButtons(container) {
    var lang = document.documentElement.lang || "en";
    var tooltip = lang.toLowerCase().indexOf("zh") === 0 ? "复制" : "Copy";
    var cells = container.querySelectorAll("div.highlight pre");
    for (var i = 0; i < cells.length; i++) {
      var pre = cells[i];
      if (pre.parentElement && pre.parentElement.querySelector(":scope > button.copybtn")) {
        continue;
      }
      codeCellCounter += 1;
      var id = "rlinf-codecell-" + codeCellCounter;
      pre.setAttribute("id", id);
      var btn =
        '<button class="copybtn o-tooltip--left" data-tooltip="' +
        tooltip +
        '" data-clipboard-target="#' +
        id +
        '">' +
        COPY_ICON_SVG +
        "</button>";
      pre.insertAdjacentHTML("afterend", btn);
    }
  }

  function initDesignTabs(container) {
    var labels = container.querySelectorAll(".sd-tab-label");
    if (!labels.length) return;
    var keyMap = {};
    var groups = {};

    for (var i = 0; i < labels.length; i++) {
      var label = labels[i];
      var syncId = label.getAttribute("data-sync-id");
      var syncGroup = label.getAttribute("data-sync-group");
      if (!syncId || !syncGroup) continue;
      var key = syncGroup + "--" + syncId;
      if (!keyMap[key]) keyMap[key] = [];
      keyMap[key].push(label);

      if (!groups[syncGroup]) {
        groups[syncGroup] = true;
        var tabParam = new URLSearchParams(window.location.search).get(syncGroup);
        if (tabParam) {
          try {
            sessionStorage.setItem(TAB_STORAGE_PREFIX + syncGroup, tabParam);
          } catch (_err) {
            // Ignore storage errors.
          }
        }
      }

      try {
        var prev = sessionStorage.getItem(TAB_STORAGE_PREFIX + syncGroup);
        if (prev === syncId && label.previousElementSibling) {
          label.previousElementSibling.checked = true;
        }
      } catch (_err) {
        // Ignore storage errors.
      }

      label.onclick = (function (g, id, k) {
        return function () {
          var peers = keyMap[k] || [];
          for (var p = 0; p < peers.length; p++) {
            if (peers[p] !== this && peers[p].previousElementSibling) {
              peers[p].previousElementSibling.checked = true;
            }
          }
          try {
            sessionStorage.setItem(TAB_STORAGE_PREFIX + g, id);
          } catch (_err) {
            // Ignore storage errors.
          }
        };
      })(syncGroup, syncId, key);
    }
  }

  function initMathJax(fetchedDoc, container) {
    var hasMath = container.querySelector(".math");
    if (!hasMath) return;

    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise([container]).catch(function () {});
      return;
    }

    if (document.querySelector('script[src*="mathjax"]')) return;
    var mathScript = fetchedDoc.querySelector('script[src*="mathjax"]');
    if (!mathScript) return;
    var s = document.createElement("script");
    s.async = true;
    s.src = mathScript.getAttribute("src");
    document.head.appendChild(s);
  }

  function refreshScrollSpy() {
    if (window.bootstrap && window.bootstrap.ScrollSpy) {
      try {
        var spy = window.bootstrap.ScrollSpy.getInstance(document.body);
        if (spy) spy.refresh();
      } catch (_err) {
        // Ignore ScrollSpy refresh errors on pages without secondary TOC.
      }
    }
  }

  function closeMobileDialogs() {
    var ids = ["pst-primary-sidebar-modal", "pst-secondary-sidebar-modal"];
    for (var i = 0; i < ids.length; i++) {
      var dlg = document.getElementById(ids[i]);
      if (dlg && dlg.open && typeof dlg.close === "function") {
        dlg.close();
      }
    }
  }

  function scrollToTarget(hash, savedScrollY) {
    if (hash) {
      var targetId = hash.slice(1);
      var targetEl =
        document.getElementById(targetId) ||
        document.getElementById(decodeURIComponent(targetId));
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: "instant", block: "start" });
        return;
      }
    }
    var y = typeof savedScrollY === "number" ? savedScrollY : 0;
    try {
      window.scrollTo({ top: y, left: 0, behavior: "instant" });
    } catch (_err) {
      window.scrollTo(0, y);
    }
  }

  function navigateTo(urlStr, options) {
    var opts = options || {};
    var targetUrl;
    try {
      targetUrl = new URL(urlStr, window.location.href);
    } catch (_err) {
      window.location.href = urlStr;
      return;
    }

    if (navController) {
      navController.abort();
      navController = null;
    }
    var controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    navController = controller;

    if (!opts.fromPopstate) {
      try {
        history.replaceState(
          { scrollY: window.scrollY },
          "",
          window.location.href
        );
      } catch (_err) {
        // Ignore history state errors.
      }
    }

    showProgress();

    fetchPageHtml(targetUrl.href, controller ? controller.signal : undefined)
      .then(function (result) {
        if (controller && controller.signal.aborted) return;

        var parser = new DOMParser();
        var doc = parser.parseFromString(result.html, "text/html");

        // Fall back to full navigation for redirect stubs or pages with inline
        // scripts inside <main>.
        if (doc.querySelector('meta[http-equiv="refresh"]')) {
          window.location.href = targetUrl.href;
          return;
        }
        var newMain = doc.querySelector("main#main-content");
        var currentMain = document.querySelector("main#main-content");
        if (!newMain || !currentMain || newMain.querySelector("script")) {
          window.location.href = targetUrl.href;
          return;
        }

        var finalUrl = new URL(result.finalUrl, window.location.href);
        if (targetUrl.hash && !finalUrl.hash) {
          finalUrl.hash = targetUrl.hash;
        }

        rebaseUrls(newMain, finalUrl.href);

        if (!opts.fromPopstate) {
          history.pushState(
            { scrollY: 0 },
            "",
            finalUrl.pathname + finalUrl.search + finalUrl.hash
          );
        }

        document.title = doc.title || document.title;
        currentMain.replaceWith(newMain);
        lastLoadedKey = canonicalPath(finalUrl.href) + finalUrl.search;

        updateDocumentationOptions(finalUrl.href);
        updateLanguageSwitcher();
        initCopyButtons(newMain);
        initDesignTabs(newMain);
        initMathJax(doc, newMain);
        refreshScrollSpy();
        closeMobileDialogs();

        var nav = document.querySelector(".bd-sidebar-primary .bd-docs-nav");
        updateSidebarActive(nav, finalUrl.href, {
          clickedLink: opts.clickedLink,
          fromSidebar: opts.fromSidebar,
        });
        scrollToTarget(finalUrl.hash, opts.savedScrollY);
        hideProgress();
      })
      .catch(function (err) {
        if (err && err.name === "AbortError") return;
        hideProgress();
        window.location.href = targetUrl.href;
      });
  }

  function setupPjax(nav) {
    if (window.location.protocol === "file:") return;

    docRootPath = computeDocRootPath();
    lastLoadedKey = canonicalPath(window.location.href) + window.location.search;

    // Normalize relative URLs (and Sphinx's `href="#"` self-link) in persistent
    // chrome once before any pushState.
    var primarySidebar = document.getElementById("pst-primary-sidebar");
    var primarySidebarModal = document.getElementById("pst-primary-sidebar-modal");
    rebaseUrls(primarySidebar, window.location.href, true);
    rebaseUrls(primarySidebarModal, window.location.href, true);
    rebaseUrls(document.querySelector(".bd-header"), window.location.href, true);
    enhanceSectionCaptions(primarySidebar);
    enhanceSectionCaptions(primarySidebarModal);

    // Warm the cache with the initial page HTML snapshot on Back navigation.
    try {
      history.replaceState({ scrollY: window.scrollY }, "", window.location.href);
    } catch (_err) {
      // Ignore.
    }

    // Prefetch sidebar / prev-next links on hover, keyboard focus, or touch.
    var hoverTimer = null;
    function schedulePrefetch(target) {
      var link = target && target.closest ? target.closest("a[href]") : null;
      if (!link || !isPjaxEligibleLink(link)) return;
      var url = new URL(link.getAttribute("href"), window.location.href);
      if (isSamePage(url)) return;
      if (pageCache.has(canonicalPath(url.href))) return;
      if (hoverTimer) clearTimeout(hoverTimer);
      hoverTimer = setTimeout(function () {
        fetchPageHtml(url.href).catch(function () {});
      }, PREFETCH_DELAY_MS);
    }

    if (nav) {
      nav.addEventListener("pointerenter", function (e) {
        schedulePrefetch(e.target);
      }, true);
      nav.addEventListener("focusin", function (e) {
        schedulePrefetch(e.target);
      });
      nav.addEventListener(
        "touchstart",
        function (e) {
          schedulePrefetch(e.target);
        },
        { passive: true }
      );
    }

    // Intercept clicks on internal documentation links.
    document.addEventListener("click", function (e) {
      if (
        e.defaultPrevented ||
        e.button !== 0 ||
        e.metaKey ||
        e.ctrlKey ||
        e.shiftKey ||
        e.altKey
      ) {
        return;
      }

      var link = e.target && e.target.closest ? e.target.closest("a[href]") : null;
      if (!link || !isPjaxEligibleLink(link)) return;

      var fromSidebar = Boolean(link.closest && link.closest(".bd-docs-nav"));
      var isCaptionLink = link.classList.contains("rlinf-caption-link");
      var url = new URL(link.getAttribute("href"), window.location.href);

      if (isCaptionLink) {
        var cap = link.closest("p.caption");
        var capUl = cap ? cap.nextElementSibling : null;
        if (isSamePage(url)) {
          e.preventDefault();
          if (cap && capUl) {
            setSectionCollapsed(cap, capUl, !cap.classList.contains("is-collapsed"), true);
          }
          return;
        }
        if (cap && capUl && cap.classList.contains("is-collapsed")) {
          setSectionCollapsed(cap, capUl, false, true);
        }
        e.preventDefault();
        navigateTo(url.href, {
          fromPopstate: false,
          fromSidebar: true,
          clickedLink: null,
        });
        return;
      }

      // Clicking anywhere on a collapsible sidebar folder row toggles its
      // children: expanding also opens its overview page via instant PJAX,
      // while clicking an already-open folder row collapses it in place.
      if (fromSidebar) {
        var li = link.parentElement;
        if (li && li.classList.contains("has-children")) {
          var details = li.querySelector(":scope > details");
          if (details) {
            var k = detailKey(details);
            var winY = window.scrollY;
            var box = scrollableAncestor(link);
            var boxY = box ? box.scrollTop : 0;
            if (details.open) {
              e.preventDefault();
              details.open = false;
              if (k && openSet) {
                openSet.delete(k);
                saveOpenSet(openSet);
              }
              if (window.scrollY !== winY) {
                window.scrollTo(window.scrollX, winY);
              }
              if (box && box.scrollTop !== boxY) {
                box.scrollTop = boxY;
              }
              return;
            }
            details.open = true;
            if (k && openSet) {
              openSet.add(k);
              saveOpenSet(openSet);
            }
            if (window.scrollY !== winY) {
              window.scrollTo(window.scrollX, winY);
            }
            if (box && box.scrollTop !== boxY) {
              box.scrollTop = boxY;
            }
          }
        }
      }

      if (isSamePage(url)) {
        if (!url.hash) {
          e.preventDefault();
          if (!fromSidebar) {
            scrollToTarget("", 0);
          }
        }
        return;
      }

      e.preventDefault();
      navigateTo(url.href, {
        fromPopstate: false,
        fromSidebar: fromSidebar,
        clickedLink: fromSidebar ? link : null,
      });
    });

    window.addEventListener("popstate", function (e) {
      var currentKey =
        canonicalPath(window.location.href) + window.location.search;
      if (currentKey === lastLoadedKey) return;
      var savedY = e.state && typeof e.state.scrollY === "number" ? e.state.scrollY : 0;
      navigateTo(window.location.href, { fromPopstate: true, savedScrollY: savedY });
    });
  }

  function init() {
    openSet = loadOpenSet();
    collapsedSet = loadStoredSet(COLLAPSED_KEY);
    var nav = document.querySelector(".bd-sidebar-primary .bd-docs-nav");
    if (nav) {
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

      // Prevent Safari from focusing 0px-height <summary> elements or sidebar
      // toggle buttons inside the sticky sidebar on mouse click.
      document.addEventListener("mousedown", function (event) {
        if (
          event.target &&
          event.target.closest &&
          event.target.closest(
            ".bd-docs-nav li.has-children > details > summary, .bd-docs-nav .rlinf-caption-toggle, .bd-docs-nav .rlinf-caption-btn"
          )
        ) {
          event.preventDefault();
        }
      });

      document.addEventListener("click", function (event) {
        var capCtrl =
          event.target && event.target.closest
            ? event.target.closest(
                ".bd-docs-nav .rlinf-caption-toggle, .bd-docs-nav .rlinf-caption-btn"
              )
            : null;
        if (!capCtrl) return;
        var cap = capCtrl.closest("p.caption");
        var ul = cap ? cap.nextElementSibling : null;
        if (!cap || !ul || ul.tagName !== "UL") return;
        event.preventDefault();
        setSectionCollapsed(cap, ul, !cap.classList.contains("is-collapsed"), true);
      });

      nav.addEventListener("click", function (event) {
        var summary =
          event.target && event.target.closest
            ? event.target.closest("li.has-children > details > summary")
            : null;
        if (!summary || event.detail === 0) return;
        var details = summary.parentElement;
        if (!details || details.tagName !== "DETAILS") return;
        event.preventDefault();
        var winY = window.scrollY;
        var box = scrollableAncestor(summary);
        var boxY = box ? box.scrollTop : 0;
        details.open = !details.open;
        if (window.scrollY !== winY) {
          window.scrollTo(window.scrollX, winY);
        }
        if (box && box.scrollTop !== boxY) {
          box.scrollTop = boxY;
        }
      });

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

    setupPjax(nav);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
