// Fills the page from data/site.json, written by tools/build_site_data.py.
const f3 = (x) => x.toFixed(3);
const pct = (a, b) => `${((100 * a) / b).toFixed(1)}%`;

export function facts(site) {
  const g = site.gender;
  const continued = site.runs.filter((r) => r.start === "continued");
  const stalled = continued.filter((r) => r.best_epoch <= 2 && r.epochs_run >= 100);
  const final = site.runs.find((r) => r.run === site.final_run);
  const merged = site.runs.filter((r) => r.dataset === final.dataset && r.start === "continued");
  return {
    ...site,
    runs_n: site.runs.length, first_date: site.runs[0].date, last_date: site.runs.at(-1).date,
    people_fmt: g.people.toLocaleString("en"), found_pct: pct(g.found, g.people), right_pct: pct(g.right, g.found),
    unique_train_photos_fmt: site.unique_train_photos.toLocaleString("en"),
    dataset: { ...site.dataset, train_fmt: site.dataset.train.toLocaleString("en") },
    continued_n: continued.length, stalled_n: stalled.length,
    final_val: f3(final.val_map50), final_dataset: final.dataset,
    before_best: f3(Math.max(...merged.filter((r) => r.date < final.date).map((r) => r.val_map50))),
    after: merged.filter((r) => r.date > final.date).map((r) => f3(r.val_map50)).join(" and "),
  };
}

const get = (obj, path) => path.split(".").reduce((o, k) => o?.[k], obj);

export function fill(root, site) {
  const f = facts(site);
  root.querySelectorAll("[data-v]").forEach((el) => { el.textContent = get(f, el.dataset.v) ?? "?"; });
  const t = site.test, pc = site.per_class;
  root.querySelector("#stats").innerHTML = [
    ["mAP50", t.map50], ["mAP50-95", t.map50_95], ["Precision", t.precision], ["Recall", t.recall],
    ["mAP50, women", pc.Female.map50, "female"], ["mAP50, men", pc.Male.map50, "male"],
  ].map(([k, v, cls = ""]) => `<div class="stat ${cls}"><b>${f3(v)}</b><span>${k}</span></div>`).join("");
  root.querySelector("#stalled").textContent =
    `${f.stalled_n} of the ${f.continued_n} runs that started from an earlier model's weights scored best at epoch 1 or 2, ` +
    "then trained about 100 more epochs without improving until early stopping (patience 100) ended them. " +
    "Training on the same data from a model that had already converged only cost time.";
  root.querySelector("#runs-table tbody").innerHTML = site.runs.map((r) => `
    <tr class="${r.run === site.final_run ? "final" : ""}"><th scope="row">${r.run}</th><td>${r.date}</td><td>${r.dataset}</td><td>${r.start}</td>
    <td>${r.best_epoch} of ${r.epochs_run}</td><td>${f3(r.val_map50)}</td><td>${f3(r.val_map50_95)}</td></tr>`).join("");
}

if (typeof document !== "undefined") {
  fill(document, await fetch("data/site.json").then((r) => r.json()));
}
