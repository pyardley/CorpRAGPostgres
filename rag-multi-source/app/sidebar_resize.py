"""
Sidebar resize persistence.

Streamlit's sidebar already supports drag-to-resize, but two things about
the native handle fall short of what this app wants:

1. The chosen width lives only in that browser tab's React state — it
   resets to the default on every page reload.
2. Streamlit hard-caps interactive dragging at ``min(600px, 90vw)``
   (baked into its frontend bundle — no CSS override can change it), which
   isn't enough room for wide content like the Audit Log table.

This injects a small script into the same-origin parent document
(Streamlit's ``st.iframe`` renders into a same-origin iframe that can reach
``window.parent``) that:

* restores the last-used width from ``localStorage`` on load,
* watches the sidebar via ``ResizeObserver`` and saves any new width the
  user drags to (up to Streamlit's native ~600px cap), and
* if ``force_width_px`` is given (the "Sidebar width" slider in the
  sidebar), applies and persists that width directly — bypassing the
  native drag cap entirely, since this sets the DOM width outside of
  Streamlit's own resize code path.

The saved width is scoped to the browser (``localStorage``), not the user
account — that matches "resize it once, it stays how I left it on this
machine" without needing any server-side schema changes.
"""

from __future__ import annotations

import streamlit as st

# Public — reused by app.sidebar for the "Sidebar width" slider's bounds.
DEFAULT_WIDTH_PX = 416  # ~26rem — the app's previous fixed default
MIN_WIDTH_PX = 288  # 18rem, matches the CSS min-width in main.py
MAX_WIDTH_PX = 1200  # 75rem, matches the CSS max-width in main.py

_STORAGE_KEY = "corporateRagSidebarWidthPx"


def inject_sidebar_resize_persistence(force_width_px: int | None = None) -> None:
    """
    Render the (invisible) persistence script. Call once per sidebar render.

    ``force_width_px``, when given, is applied immediately and saved to
    localStorage — used when the "Sidebar width" slider changes, so the
    user can size the sidebar beyond Streamlit's native drag-resize cap.
    """
    force_width_literal = "null" if force_width_px is None else str(int(force_width_px))
    st.iframe(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            const STORAGE_KEY = "{_STORAGE_KEY}";
            const DEFAULT_WIDTH = {DEFAULT_WIDTH_PX};
            const MIN_WIDTH = {MIN_WIDTH_PX};
            const MAX_WIDTH = {MAX_WIDTH_PX};
            const FORCE_WIDTH = {force_width_literal};

            function clamp(px) {{
                return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, px));
            }}

            function applyWidth(section, px) {{
                section.style.setProperty("width", px + "px");
                const inner = section.firstElementChild;
                if (inner) {{
                    inner.style.setProperty("width", px + "px");
                }}
            }}

            function init() {{
                const section = doc.querySelector('section[data-testid="stSidebar"]');
                if (!section) {{
                    return false;
                }}

                let startWidth;
                if (FORCE_WIDTH !== null) {{
                    startWidth = clamp(FORCE_WIDTH);
                    window.parent.localStorage.setItem(STORAGE_KEY, String(startWidth));
                }} else {{
                    const saved = parseInt(
                        window.parent.localStorage.getItem(STORAGE_KEY), 10
                    );
                    startWidth = clamp(
                        Number.isFinite(saved) && saved > 0 ? saved : DEFAULT_WIDTH
                    );
                }}
                applyWidth(section, startWidth);

                // Only attach one observer for the lifetime of the tab —
                // this script re-runs on every Streamlit rerun, but the
                // sidebar DOM node is reused, not remounted.
                if (!window.parent.__sidebarResizePersistInit) {{
                    window.parent.__sidebarResizePersistInit = true;
                    let lastWidth = Math.round(
                        section.getBoundingClientRect().width
                    );
                    let debounceTimer = null;
                    const observer = new window.parent.ResizeObserver((entries) => {{
                        for (const entry of entries) {{
                            const width = Math.round(entry.contentRect.width);
                            if (width === lastWidth || width <= 0) continue;
                            lastWidth = width;
                            clearTimeout(debounceTimer);
                            debounceTimer = setTimeout(() => {{
                                window.parent.localStorage.setItem(
                                    STORAGE_KEY, String(width)
                                );
                            }}, 300);
                        }}
                    }});
                    observer.observe(section);
                }}
                return true;
            }}

            if (!init()) {{
                // First paint — sidebar may not be mounted yet. Retry
                // briefly rather than giving up.
                let attempts = 0;
                const retry = setInterval(() => {{
                    attempts += 1;
                    if (init() || attempts > 20) {{
                        clearInterval(retry);
                    }}
                }}, 100);
            }}
        }})();
        </script>
        """,
        height=1,
    )
