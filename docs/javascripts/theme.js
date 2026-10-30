/* A keyboard-accessible appearance toggle; all navigation comes from MkDocs. */
(() => {
  // RTD normally scrolls the active link to the top, hiding the project header.
  // Keep its current-page expansion while preserving the reader's sidebar view.
  const navigation = window.SphinxRtdTheme?.Navigation;
  if (navigation) {
    const reset = navigation.reset;
    navigation.reset = function (...args) {
      const sidebar = document.querySelector(".wy-side-scroll");
      const previousScroll = sidebar?.scrollTop || 0;
      reset.apply(this, args);
      if (sidebar) sidebar.scrollTop = location.hash ? previousScroll : 0;
    };
  }

  const drawer = document.querySelector(".wy-nav-side");
  const menuButton = document.querySelector(".mobile-menu-toggle");
  if (drawer && menuButton) {
    drawer.id = "documentation-navigation";
    menuButton.setAttribute("aria-controls", drawer.id);
    const syncDrawer = () => {
      menuButton.setAttribute("aria-expanded", String(drawer.classList.contains("shift")));
    };
    new MutationObserver(syncDrawer).observe(drawer, {
      attributes: true, attributeFilter: ["class"],
    });
    syncDrawer();
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !drawer.classList.contains("shift")) return;
      document.querySelectorAll("[data-toggle='wy-nav-shift'], .rst-versions")
        .forEach((element) => element.classList.remove("shift"));
      menuButton.focus();
    });
  }

  const header = document.querySelector(".wy-side-nav-search");
  if (!header) return;

  const button = document.createElement("button");
  button.type = "button";
  button.className = "appearance-toggle";
  const isDark = () => document.documentElement.dataset.theme === "dark";

  function updateLabel() {
    button.textContent = isDark() ? "Light appearance" : "Dark appearance";
    button.setAttribute("aria-label", `Switch to ${isDark() ? "light" : "dark"} appearance`);
    button.setAttribute("aria-pressed", String(isDark()));
  }

  button.addEventListener("click", () => {
    document.documentElement.dataset.theme = isDark() ? "light" : "dark";
    try {
      localStorage.setItem("flash-ipa-theme", document.documentElement.dataset.theme);
    } catch (_) {}
    updateLabel();
  });

  header.append(button);
  updateLabel();

  // Keep groups visible initially while allowing readers to fold them away.
  document.querySelectorAll(".wy-menu-vertical > .caption").forEach((caption, index) => {
    const list = caption.nextElementSibling;
    if (!list || list.tagName !== "UL") return;
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "nav-group-toggle";
    toggle.textContent = caption.textContent;
    toggle.setAttribute("aria-expanded", "true");
    list.id = `nav-group-${index}`;
    toggle.setAttribute("aria-controls", list.id);
    toggle.addEventListener("click", () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      list.hidden = expanded;
    });
    caption.replaceChildren(toggle);
  });
})();
