---
title: Home
hide:
  - title
  - toc
  - footer
---

<style>
  .md-content__inner > h1#__skip {
    display: none;
  }

  /* Home page should be a single full-bleed hero with no vertical scrolling. */
  html,
  body {
    overflow: hidden;
  }

  .md-main {
    overflow: hidden;
  }

  .md-sidebar,
  .md-sidebar--primary {
    display: none;
  }

  .md-main__inner {
    max-width: none;
    margin: 0;
    grid-template-columns: minmax(0, 1fr) !important;
    gap: 0;
  }

  .md-main__inner.md-grid {
    max-width: none;
    margin: 0;
    padding: 0;
  }

  .md-content,
  .md-content__inner {
    margin: 0;
    padding: 0;
    max-width: none;
    width: 100%;
  }

  .md-footer {
    display: none;
  }
</style>

<div class="hero" markdown>

<canvas
  class="hero__canvas"
  data-visual-system="inference-landscape"
  aria-hidden="true"
></canvas>

<div class="hero__content" markdown>

Notes around statistics, optimization and decision making.

</div>
</div>
