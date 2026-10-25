/* All Papers browse page.
 *
 * Everything happens in the browser against one JSON file fetched once.
 * 3,717 papers is small enough that plain array scans beat any index: a full
 * filter pass measures about a millisecond, so there is no debounce and no
 * search library. What does cost real time is the DOM, which is why only the
 * current page of rows is ever built.
 */

"use strict";

// Page sizes offered in the "Show" dropdown. 50 is the default: 100 rows made
// the page long enough that the pagination controls were off screen.
const PAGE_SIZES = [10, 50, 100];
const DEFAULT_PAGE_SIZE = 50;
const TYPE_NAMES = ["Poster", "Oral", "Spotlight"];
const TYPE_CLASS = ["pres-poster", "pres-oral", "pres-spotlight"];

let INDEX = null; // parsed papers.json
let SEARCH = []; // per paper: its title, authors, and areas as one
// lowercase, accent-free string to search against
let view = []; // current filtered and sorted papers

const state = {
  q: "",
  year: "",
  type: "",
  area: "",
  code: 0,
  early: 0,
  sort: "year-desc",
  per: DEFAULT_PAGE_SIZE,
  page: 1,
};

// U+1D45B, mathematical italic small n. An <option> cannot carry markup, so
// this is the only way to italicize the n inside a native dropdown. If it ever
// renders as an empty box on some device, change it to a plain "n" here.
const N_ITALIC = "\u{1D45B}";

const $ = (id) => document.getElementById(id);

/* Lowercase the text and take the accent marks off the letters, so that typing
 * "muller" finds "Müller" and "jose" finds "José".
 *
 * One thing this does not do: the micro sign µ and the Greek letter mu μ both
 * end up as μ, so either spelling of "µ2 Tokenizer" finds that paper, but
 * typing the letters "mu2" does not, because μ is one letter and not two. */
function searchable(s) {
  // NFKD separates a letter from its accent mark; \u0300-\u036f is the range
  // those marks land in, so deleting that range leaves the plain letter.
  // Written as escapes because the characters themselves are invisible in an
  // editor and a re-encode of this file would silently break the pattern.
  return s.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

function scaleFor(year) {
  return INDEX.scales[String(year)] || 6;
}

/* One predicate for every filter, with an escape hatch: passing skip="area"
 * evaluates every filter except the area one, which is what the area dropdown
 * needs in order to count how many papers each option would return. */
function matches(p, i, skip) {
  if (skip !== "year" && state.year && p.year !== +state.year) return false;
  if (skip !== "type" && state.type !== "" && p.type !== +state.type) return false;
  if (skip !== "area" && state.area !== "" && !p.areas.includes(+state.area)) return false;
  if (skip !== "code" && state.code && !p.code_url) return false;
  if (skip !== "early" && state.early && !p.early) return false;
  if (skip !== "q" && state.q) {
    const hay = SEARCH[i];
    for (const term of searchable(state.q).split(/\s+/)) {
      if (term && !hay.includes(term)) return false;
    }
  }
  return true;
}

const SORTS = {
  "year-desc": (a, b) => b.year - a.year || a.title.localeCompare(b.title),
  "year-asc": (a, b) => a.year - b.year || a.title.localeCompare(b.title),
  "score-desc": (a, b) => b.score_norm - a.score_norm || b.year - a.year,
  "score-asc": (a, b) => a.score_norm - b.score_norm || a.year - b.year,
  "title-asc": (a, b) => a.title.localeCompare(b.title),
};

/* Score sorting is offered only when a single year is selected.
 *
 * MICCAI scored on 1-9 in 2021, 1-8 in 2022 and 2023, and 1-6 since 2024.
 * Sorting the raw number across years is plainly wrong (the best possible 2025
 * paper averages 6.00, below the 417th-best 2021 paper), but normalizing does
 * not rescue it either: 2021 reviewers used the top of their scale far more
 * freely, so 62 of the top 100 by normalized score are from 2021. Either way
 * the ranking measures which year a paper was reviewed in. Inside one year the
 * scale is fixed and the comparison is real, so that is where it is allowed.
 * (Within one year raw and normalized give the same order, so which one the
 * comparator uses no longer matters.) */
const SCORE_SORTS = ["score-desc", "score-asc"];

function scoreSortAllowed() {
  return state.year !== "";
}

/* Enable or disable the two score options and explain why, so the rule is
 * visible rather than a silently missing feature. */
function syncSortOptions() {
  const allowed = scoreSortAllowed();
  const sel = $("sort-select");
  for (const value of SCORE_SORTS) {
    const opt = sel.querySelector(`option[value="${value}"]`);
    if (opt) opt.disabled = !allowed;
  }
  if (!allowed && SCORE_SORTS.includes(state.sort)) state.sort = "year-desc";
  sel.value = state.sort;

  const hint = $("sort-hint");
  hint.hidden = allowed;
  hint.textContent = allowed
    ? ""
    : "Pick a year to sort by score. Sorting by score is only available for a single year because MICCAI changed its review score scale over the years.";
}

/* ------------------------------------------------------------------ render */

function rowHtml(p) {
  const areaName = p.areas.length ? INDEX.areas[p.areas[0]] : "";
  const allAreas = p.areas.map((a) => INDEX.areas[a]).join(" · ");
  return (
    `<tr class="paper-row" data-id="${esc(p.id)}">` +
    `<td class="year-col">${p.year}</td>` +
    `<td><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></td>` +
    `<td class="subject-col" title="${esc(allAreas)}">${esc(areaName)}</td>` +
    `<td class="score-col"><span class="score-mean">${p.score.toFixed(2)}/${scaleFor(p.year)}</span></td>` +
    `<td class="pres-col"><span class="${TYPE_CLASS[p.type]}">${TYPE_NAMES[p.type]}</span></td>` +
    `</tr>` +
    `<tr class="detail-row" hidden><td colspan="5"></td></tr>`
  );
}

function render() {
  const start = (state.page - 1) * state.per;
  $("papers-tbody").innerHTML = view
    .slice(start, start + state.per)
    .map(rowHtml)
    .join("");

  const n = view.length;
  const total = INDEX.papers.length;
  $("result-count").textContent =
    n === total
      ? `${total.toLocaleString()} papers`
      : `${n.toLocaleString()} of ${total.toLocaleString()} papers`;

  const status = $("papers-status");
  status.textContent = "No papers match these filters.";
  status.hidden = n !== 0;
}

function pageButton(label, page, opts) {
  const cls = opts && opts.current ? ' class="current"' : "";
  const dis = opts && opts.disabled ? " disabled" : "";
  return `<button type="button" data-page="${page}"${cls}${dis}>${label}</button>`;
}

function renderPagination() {
  const pages = Math.max(1, Math.ceil(view.length / state.per));
  const cur = state.page;
  if (pages <= 1) {
    $("pagination").innerHTML = "";
    return;
  }
  const parts = [pageButton("Prev", cur - 1, { disabled: cur === 1 })];
  const shown = new Set([1, pages, cur, cur - 1, cur + 1, cur - 2, cur + 2]);
  let last = 0;
  for (let p = 1; p <= pages; p++) {
    if (!shown.has(p)) continue;
    if (p - last > 1) parts.push('<span class="ellipsis">...</span>');
    parts.push(pageButton(String(p), p, { current: p === cur }));
    last = p;
  }
  parts.push(pageButton("Next", cur + 1, { disabled: cur === pages }));
  $("pagination").innerHTML = parts.join("");
}

function refresh() {
  view = INDEX.papers.filter((p, i) => matches(p, i, null));
  view.sort(SORTS[state.sort] || SORTS["year-desc"]);
  const pages = Math.max(1, Math.ceil(view.length / state.per));
  state.page = Math.min(Math.max(1, state.page), pages);
  render();
  renderPagination();
}

/* ------------------------------------------------------------------ detail */

/* Link chips: label, key on the record. Only the ones a paper actually has are
 * drawn, so there are no dead links and no blank slots. Coverage runs from 100%
 * (DOI, SharedIt) down to 17% (supplementary, 2024 and 2025 only). */
const LINKS = [
  ["Paper page", "url"],
  ["PDF", "pdf_url"],
  ["DOI", "doi"],
  ["SharedIt", "sharedit_url"],
  ["Code", "code_url"],
  ["Suppl.", "supp_url"],
];

function detailHtml(p) {
  const rows = [];

  rows.push([
    "Authors",
    esc(p.authors.join("; ")) +
    ` <span class="score-pct">(${p.authors.length})</span>`,
  ]);

  if (p.areas.length) {
    rows.push([
      "Subject areas",
      p.areas.map((a) => esc(INDEX.areas[a])).join(" &middot; "),
    ]);
  }

  rows.push([
    "Reviews",
    `${p.reviews.join(", ")} <span class="score-pct">(mean ` +
    `${p.score.toFixed(2)} / ${scaleFor(p.year)})</span>`,
  ]);

  rows.push(["Early accept", p.early ? "yes" : "no"]);

  let pres = TYPE_NAMES[p.type];
  if (p.session) pres += ` in the &ldquo;${esc(p.session)}&rdquo; session`;
  rows.push(["Presentation", pres]);

  const chips = LINKS.filter(([, key]) => p[key]).map(
    ([label, key]) =>
      `<a class="link-chip" href="${esc(p[key])}" target="_blank" rel="noopener">${label}</a>`
  );
  (p.dataset_urls || []).forEach((u, i) => {
    const label = p.dataset_urls.length > 1 ? `Data ${i + 1}` : "Data";
    chips.push(
      `<a class="link-chip" href="${esc(u)}" target="_blank" rel="noopener">${label}</a>`
    );
  });
  if (chips.length) rows.push(["Links", chips.join("")]);

  if (p.lncs) rows.push(["Volume", esc(p.lncs)]);

  return (
    '<dl class="paper-detail">' +
    rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") +
    "</dl>"
  );
}

/* --------------------------------------------------------------- area list */

/* The subject-area dropdown is rebuilt from whatever the other filters leave
 * showing, and each option carries its count. MICCAI renames its areas (2025
 * renamed nearly all of them; no name appears in all five years and 102 of 161
 * appear in exactly one), so a fixed list of 161 would silently exclude whole
 * years. Counting live makes that visible instead. */
function populateAreaFilter() {
  const counts = new Map();
  let total = 0;
  INDEX.papers.forEach((p, i) => {
    if (!matches(p, i, "area")) return;
    total += 1;
    for (const a of p.areas) counts.set(a, (counts.get(a) || 0) + 1);
  });

  const opts = [...counts.entries()]
    .sort(
      (x, y) =>
        y[1] - x[1] || INDEX.areas[x[0]].localeCompare(INDEX.areas[y[0]])
    )
    .map(
      ([id, n]) =>
        `<option value="${id}">${esc(INDEX.areas[id])} (${N_ITALIC}=${n.toLocaleString()})</option>`
    );

  const sel = $("filter-area");
  // "All areas (3,717)" read as a count of areas rather than of papers, which
  // is why every option carries an explicit n=.
  sel.innerHTML =
    `<option value="">All areas (${N_ITALIC}=${total.toLocaleString()})</option>` +
    opts.join("");
  sel.value = state.area;
  // A stale area id (from an old URL, or one filtered out) falls back to All.
  if (sel.value !== state.area) state.area = "";
}

/* ---------------------------------------------------------------- URL state */

/* Filter state lives in the query string so a filtered view can be bookmarked,
 * shared, or linked to from elsewhere on the site. Typing uses replaceState so
 * the back button is not flooded with one entry per keystroke; discrete control
 * changes use pushState so Back means what a reader expects. */
const URL_KEYS = [
  ["q", "q", String],
  ["year", "year", String],
  ["type", "type", String],
  ["area", "area", String],
  ["code", "code", Number],
  ["early", "early", Number],
  ["sort", "sort", String],
  ["per", "per", Number],
  ["p", "page", Number],
];

const DEFAULTS = {
  q: "",
  year: "",
  type: "",
  area: "",
  code: 0,
  early: 0,
  sort: "year-desc",
  per: DEFAULT_PAGE_SIZE,
  page: 1,
};

function readUrl() {
  const params = new URLSearchParams(location.search);
  for (const [param, key, cast] of URL_KEYS) {
    if (params.has(param)) state[key] = cast(params.get(param));
  }
  if (!SORTS[state.sort]) state.sort = "year-desc";
  if (!PAGE_SIZES.includes(state.per)) state.per = DEFAULT_PAGE_SIZE;
  if (!(state.page >= 1)) state.page = 1;
}

function writeUrl(push) {
  const params = new URLSearchParams();
  for (const [param, key] of URL_KEYS) {
    if (state[key] === DEFAULTS[key]) continue;
    params.set(param, state[key]);
  }
  const qs = params.toString();
  const url = location.pathname + (qs ? "?" + qs : "");
  if (push) history.pushState(null, "", url);
  else history.replaceState(null, "", url);
}

function applyStateToControls() {
  $("papers-search").value = state.q;
  $("filter-year").value = state.year;
  $("filter-type").value = state.type;
  $("filter-code").checked = !!state.code;
  $("filter-early").checked = !!state.early;
  $("page-size").value = String(state.per);
  // Sets the sort control too, and may reset it if the year filter forbids it.
  syncSortOptions();
  // The area dropdown is rebuilt from the other filters, so it goes last.
  populateAreaFilter();
}

/* --------------------------------------------------------------- CSV export */

/* Exports the whole filtered set, not just the visible page. Written as UTF-8
 * with a byte order mark, because Excel otherwise mangles accented author
 * names. */
const CSV_COLUMNS = [
  ["year", (p) => p.year],
  ["title", (p) => p.title],
  ["authors", (p) => p.authors.join("; ")],
  ["subject_areas", (p) => p.areas.map((a) => INDEX.areas[a]).join("; ")],
  ["mean_score", (p) => p.score.toFixed(2)],
  ["scale_max", (p) => scaleFor(p.year)],
  ["score_pct_of_scale", (p) => Math.round(p.score_norm * 100)],
  ["reviewer_scores", (p) => p.reviews.join("; ")],
  ["presentation_type", (p) => TYPE_NAMES[p.type].toLowerCase()],
  ["session", (p) => p.session || ""],
  ["early_accepted", (p) => (p.early ? "yes" : "no")],
  ["paper_url", (p) => p.url || ""],
  ["pdf_url", (p) => p.pdf_url || ""],
  ["doi", (p) => p.doi || ""],
  ["sharedit_url", (p) => p.sharedit_url || ""],
  ["code_url", (p) => p.code_url || ""],
  ["supp_url", (p) => p.supp_url || ""],
  ["dataset_urls", (p) => (p.dataset_urls || []).join("; ")],
  ["lncs_volume", (p) => p.lncs || ""],
];

function csvCell(value) {
  const s = String(value);
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function toCsv(papers) {
  const head = CSV_COLUMNS.map(([name]) => name).join(",");
  const rows = papers.map((p) =>
    CSV_COLUMNS.map(([, get]) => csvCell(get(p))).join(",")
  );
  // \uFEFF is the UTF-8 byte order mark, written as an escape because the
  // literal character is invisible in an editor and trivially lost on a
  // copy-paste, which would silently break accented names in Excel.
  return "\uFEFF" + [head, ...rows].join("\r\n") + "\r\n";
}

function exportFilename(extension) {
  const bits = ["miccai-papers"];
  if (state.year) bits.push(state.year);
  if (state.type !== "") bits.push(TYPE_NAMES[+state.type].toLowerCase());
  if (state.area !== "") {
    bits.push(
      INDEX.areas[+state.area]
        .replace(/[^A-Za-z0-9]+/g, "-")
        .replace(/^-|-$/g, "")
        .toLowerCase()
        .slice(0, 40)
    );
  }
  if (state.code) bits.push("with-code");
  if (state.early) bits.push("early");
  if (state.q) bits.push(state.q.replace(/[^A-Za-z0-9]+/g, "-").slice(0, 20));
  return bits.join("-") + "." + extension;
}

/* Hands the browser a generated file. Used by both exports. */
function download(text, mime, filename) {
  const blob = new Blob([text], { type: mime + ";charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* -------------------------------------------------------------- JSON export */

/* The JSON export carries the filters that produced it, and the link that
 * reproduces them, so the file explains itself once it has been emailed on.
 * Subject areas are written out by name rather than as indices into
 * papers.json: an export that needs a second file to be readable is not much
 * of an export. */
function toJson(papers) {
  const filters = {};
  if (state.q) filters.search = state.q;
  if (state.year) filters.year = +state.year;
  if (state.type !== "") filters.presentation_type = TYPE_NAMES[+state.type].toLowerCase();
  if (state.area !== "") filters.subject_area = INDEX.areas[+state.area];
  if (state.code) filters.has_code = true;
  if (state.early) filters.early_accepted = true;

  const records = papers.map((p) => {
    const out = {
      paper_id: p.id,
      year: p.year,
      title: p.title,
      authors: p.authors,
      subject_areas: p.areas.map((a) => INDEX.areas[a]),
      mean_score: p.score,
      scale_max: scaleFor(p.year),
      reviewer_scores: p.reviews,
      presentation_type: TYPE_NAMES[p.type].toLowerCase(),
      early_accepted: !!p.early,
      paper_url: p.url,
    };
    if (p.session) out.session = p.session;
    for (const key of ["pdf_url", "doi", "sharedit_url", "code_url", "supp_url"]) {
      if (p[key]) out[key] = p[key];
    }
    if (p.dataset_urls) out.dataset_urls = p.dataset_urls;
    if (p.lncs) out.lncs_volume = p.lncs;
    return out;
  });

  return JSON.stringify(
    {
      source: "MICCAI Explorer, All Papers",
      link: location.href,
      exported: new Date().toISOString().slice(0, 10),
      index_built: INDEX.built,
      filters: filters,
      sort: state.sort,
      count: records.length,
      total_papers: INDEX.papers.length,
      review_scale_max_by_year: INDEX.scales,
      papers: records,
    },
    null,
    2
  );
}

/* --------------------------------------------------------------- copy link */

const COPY_LABEL = "Copy link";

/* An off-screen textarea and the old synchronous copy command. Works over
 * plain http and in the cases below where the clipboard promise is no use. */
function copyViaTextarea(text) {
  const box = document.createElement("textarea");
  box.value = text;
  box.setAttribute("readonly", "");
  box.style.position = "fixed";
  box.style.top = "0";
  box.style.opacity = "0";
  document.body.appendChild(box);
  box.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch (err) {
    ok = false;
  }
  document.body.removeChild(box);
  return ok;
}

/* navigator.clipboard needs a secure context (https, or localhost), and it is
 * raced against a timeout rather than simply awaited: in some browsers its
 * promise never settles at all, which would leave the button sitting on its
 * original label with nothing to tell the reader whether the copy worked.
 * The failure message points at the address bar rather than opening a dialog;
 * a modal to report a failed copy is worse than the failed copy. */
async function copyLink(button) {
  const url = location.href;
  let ok = false;
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await Promise.race([
        navigator.clipboard.writeText(url),
        new Promise((resolve, reject) => setTimeout(reject, 1200)),
      ]);
      ok = true;
    } catch (err) {
      ok = false;
    }
  }
  if (!ok) ok = copyViaTextarea(url);
  button.textContent = ok ? "Link copied" : "Copy from the address bar";
  setTimeout(() => {
    button.textContent = COPY_LABEL;
  }, 2000);
}

/* ------------------------------------------------------------------- wiring */

function onFilterChange(push) {
  state.page = 1;
  syncSortOptions();
  populateAreaFilter();
  refresh();
  writeUrl(push);
}

function bindControls() {
  $("papers-search").addEventListener("input", (e) => {
    state.q = e.target.value;
    onFilterChange(false);
  });

  for (const [id, key] of [
    ["filter-year", "year"],
    ["filter-type", "type"],
    ["filter-area", "area"],
  ]) {
    $(id).addEventListener("change", (e) => {
      state[key] = e.target.value;
      onFilterChange(true);
    });
  }

  for (const [id, key] of [
    ["filter-code", "code"],
    ["filter-early", "early"],
  ]) {
    $(id).addEventListener("change", (e) => {
      state[key] = e.target.checked ? 1 : 0;
      onFilterChange(true);
    });
  }

  $("sort-select").addEventListener("change", (e) => {
    state.sort = e.target.value;
    state.page = 1;
    refresh();
    writeUrl(true);
  });

  $("page-size").addEventListener("change", (e) => {
    const per = +e.target.value;
    // Keep the first row of the current page in view rather than jumping back
    // to page 1: going from 100 to 10 on page 3 should land on the same papers.
    const firstRow = (state.page - 1) * state.per;
    state.per = PAGE_SIZES.includes(per) ? per : DEFAULT_PAGE_SIZE;
    state.page = Math.floor(firstRow / state.per) + 1;
    refresh();
    writeUrl(true);
  });

  $("clear-filters").addEventListener("click", () => {
    Object.assign(state, DEFAULTS);
    applyStateToControls();
    refresh();
    writeUrl(true);
  });

  $("pagination").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-page]");
    if (!btn || btn.disabled) return;
    state.page = +btn.dataset.page;
    refresh();
    writeUrl(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  $("papers-tbody").addEventListener("click", (e) => {
    // The title is a real link out; do not swallow that click.
    if (e.target.closest("a")) return;
    const row = e.target.closest("tr.paper-row");
    if (!row) return;
    const detail = row.nextElementSibling;
    if (!detail || !detail.classList.contains("detail-row")) return;
    if (detail.hidden) {
      const paper = view.find((p) => p.id === row.dataset.id);
      if (!paper) return;
      detail.firstElementChild.innerHTML = detailHtml(paper);
      detail.hidden = false;
    } else {
      detail.hidden = true;
    }
  });

  $("export-csv").addEventListener("click", () => {
    download(toCsv(view), "text/csv", exportFilename("csv"));
  });

  $("export-json").addEventListener("click", () => {
    download(toJson(view), "application/json", exportFilename("json"));
  });

  $("copy-link").addEventListener("click", (e) => copyLink(e.currentTarget));

  window.addEventListener("popstate", () => {
    Object.assign(state, DEFAULTS);
    readUrl();
    applyStateToControls();
    refresh();
  });
}

function buildSearchStrings() {
  SEARCH = INDEX.papers.map((p) =>
    searchable(
      p.title +
      " " +
      p.authors.join(" ") +
      " " +
      p.areas.map((a) => INDEX.areas[a]).join(" ")
    )
  );
}

async function load() {
  const status = $("papers-status");
  try {
    const res = await fetch("papers.json");
    if (!res.ok) throw new Error("HTTP " + res.status);
    INDEX = await res.json();
  } catch (err) {
    status.hidden = false;
    status.textContent = "Could not load the paper list. ";
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "papers-button";
    retry.textContent = "Retry";
    retry.addEventListener("click", () => location.reload());
    status.appendChild(retry);
    return;
  }
  buildSearchStrings();
  readUrl();
  applyStateToControls();
  bindControls();
  refresh();
}

// The script is loaded with defer, so the DOM is already parsed here.
load();
