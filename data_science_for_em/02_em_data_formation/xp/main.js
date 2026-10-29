// Entry point: loads the widget groups and mounts each widget when its slide is shown.
// In the deck: every <div class="xp" data-xp="NAME"> is a placeholder for widget NAME.
import { mount, setActive } from "./core.js";

const GROUPS = ["./redline.js", "./w-learn.js", "./w-sampling.js", "./w-noise.js", "./w-loss.js", "./w-trust.js"];

async function loadGroups() {
  const res = await Promise.allSettled(GROUPS.map((g) => import(g)));
  res.forEach((r, i) => { if (r.status === "rejected") console.error(`xp: ${GROUPS[i]} failed`, r.reason); });
}

const slideOf = (el) => el.closest("section");

// Fit a widget into the free height of its slide: scale it up while it fits (at most 1.3x and the
// slide width), then centre it between the elements above and below it.
const SLIDE_H = 1080 - 70, SLIDE_W = 1840;
function fit(el) {
  const sec = slideOf(el);
  if (!sec || !window.Reveal || el.dataset.fit === "off") return;
  const sc = Reveal.getScale() || 1, z0 = parseFloat(el.style.zoom) || 1;
  const r = el.getBoundingClientRect(), top = (r.top - sec.getBoundingClientRect().top) / sc - (parseFloat(el.style.marginTop) || 0);
  const h = r.height / sc / z0, kid = el.firstElementChild, w = kid ? kid.getBoundingClientRect().width / sc / z0 : SLIDE_W;
  let below = 0;
  for (let n = el.nextElementSibling; n; n = n.nextElementSibling) {
    const nr = n.getBoundingClientRect();
    if (nr.height > 0 && getComputedStyle(n).position !== "absolute") below += nr.height / sc + 12;
  }
  const avail = SLIDE_H - top - below;
  if (avail <= 0 || h <= 0) return;
  const z = Math.max(1, Math.min(1.3, avail / h, SLIDE_W / w));
  el.style.zoom = z > 1.02 ? z.toFixed(3) : "";
  const zz = z > 1.02 ? z : 1, spare = avail - h * zz;
  el.style.marginTop = spare > 16 ? `${Math.min(spare / 2, 140).toFixed(0)}px` : "";
}
function watch(el) {
  if (el.__fit) return;
  el.__fit = new ResizeObserver(() => { if (el.__xp && el.__xp.active) requestAnimationFrame(() => fit(el)); });
  el.__fit.observe(el.firstElementChild || el);
}

function show(current) {
  for (const el of document.querySelectorAll(".xp[data-xp]")) {
    const on = slideOf(el) === current;
    if (on) { setActive(mount(el), true); watch(el); requestAnimationFrame(() => fit(el)); setTimeout(() => fit(el), 800); }
    else if (el.__xp) setActive(el.__xp, false);
  }
  // Build the widgets of the next slide in idle time, so that they appear without delay.
  const next = window.Reveal && Reveal.getSlides()[Reveal.getSlidePastCount() + 1];
  if (next) (window.requestIdleCallback || setTimeout)(() => next.querySelectorAll(".xp[data-xp]").forEach(mount));
}

async function init() {
  await loadGroups();
  if (window.XP_DEV) {                                   // dev.html: everything on one page, all active
    document.querySelectorAll(".xp[data-xp]").forEach((el) => setActive(mount(el), true));
    document.body.dataset.ready = "1";
    return;
  }
  if (/print-pdf/i.test(location.search)) {
    document.querySelectorAll(".xp[data-xp]").forEach((el) => mount(el));
    return;
  }
  const start = () => {
    show(Reveal.getCurrentSlide());
    Reveal.on("slidechanged", (e) => show(e.currentSlide));
  };
  if (Reveal.isReady()) start(); else Reveal.on("ready", start);
}

init();
