// SatQuery AI analysis report.
//
// Rendered by services/report.py, which writes `data.json` and the image
// previews next to this file in a scratch directory. All user-facing strings
// arrive already localised in `data.labels`; this template only lays them out.

#let data = json("data.json")
#let L = data.labels
#let res = data.result
#let tr = res.translation
#let localised = data.localised
#let is-rtl = data.rtl
#let doc-dir = if is-rtl { rtl } else { ltr }

#let ink = rgb("#0a0a0a")
#let muted = rgb("#666666")
#let faint = rgb("#8f8f8f")
#let rule = rgb("#e2e2e2")
#let soft = rgb("#f4f4f4")

// Poppins sets Latin text, as in the web app. Every Indic script falls through
// to its Noto face; the ranges below deliberately exclude ZWJ/ZWNJ (U+200C/D)
// so joiners stay in the same font as the conjunct they shape.
#let latin = regex("[\u{0020}-\u{024F}\u{2010}-\u{205F}\u{20A0}-\u{20C0}\u{2190}-\u{21FF}]")
#let fonts = (
  (name: "Poppins", covers: latin),
  "Noto Sans Devanagari",
  "Noto Sans Bengali",
  "Noto Sans Gurmukhi",
  "Noto Sans Gujarati",
  "Noto Sans Oriya",
  "Noto Sans Tamil",
  "Noto Sans Telugu",
  "Noto Sans Kannada",
  "Noto Sans Malayalam",
  "Noto Sans Ol Chiki",
  "Noto Sans Meetei Mayek",
) + (if data.nastaliq { ("Noto Nastaliq Urdu",) } else { () }) + (
  "Noto Naskh Arabic",
)
#let mono(body) = text(font: "DejaVu Sans Mono", size: 0.9em, body)

// Text the models produced stays English. Inline runs only need the language
// set; the bidi algorithm places them inside right-to-left sentences. Whole
// English paragraphs also run left to right.
#let english(body) = text(lang: "en", body)
#let english-par(body) = text(lang: "en", dir: ltr, body)
// Page chrome keeps fixed left/right positions whatever the reading direction.
#let chrome(start-part, end-part) = {
  set text(dir: ltr)
  grid(
    columns: (1fr, auto),
    align: (left + bottom, right + bottom),
    text(dir: doc-dir, start-part),
    text(dir: doc-dir, end-part),
  )
}
// A short Latin value (a duration, a size) kept in one piece, so the bidi
// algorithm cannot reorder "6.8 s" into "s 6.8" inside a right-to-left line.
#let ltr-run(body) = box(text(lang: "en", dir: ltr, body))
#let pick(local, fallback) = if localised and local != none { local } else { fallback }

#let fmt-ms(ms) = if ms >= 1000 {
  str(calc.round(ms / 1000, digits: 1)) + " s"
} else {
  str(int(calc.round(ms))) + " ms"
}
#let pct(x) = str(int(calc.round(x * 100))) + "%"
#let fill-index(template, i) = template.replace("{{index}}", str(i))

#set document(title: L.title, author: "SatQuery AI", keywords: ("remote sensing", "analysis", res.task))
#set text(
  font: fonts,
  size: 9.5pt,
  fill: ink,
  lang: data.lang,
  dir: doc-dir,
)
#set par(leading: 0.72em, spacing: 0.9em)
// Nastaliq climbs and descends far beyond the Latin line box; measure lines by
// the real glyph extents so rules and neighbouring lines never cut through it.
#set text(top-edge: "ascender", bottom-edge: "descender") if data.nastaliq
#set par(leading: 0.35em) if data.nastaliq
#set page(
  paper: "a4",
  margin: (x: 18mm, top: 20mm, bottom: 20mm),
  header: context {
    if counter(page).get().first() > 1 {
      set text(7.5pt, fill: faint)
      block(width: 100%, inset: (bottom: 4pt), stroke: (bottom: 0.4pt + rule), chrome(
        english[SatQuery AI],
        text(bottom-edge: "descender", L.title),
      ))
    }
  },
  footer: context {
    set text(7.5pt, fill: faint)
    chrome(
      english(mono(res.request_id)),
      text(dir: ltr, counter(page).display("1 / 1", both: true)),
    )
  },
)

#show heading.where(level: 1): it => block(
  width: 100%,
  above: 20pt,
  below: 9pt,
  sticky: true,
  inset: (bottom: 5pt),
  stroke: (bottom: 0.5pt + ink),
  text(11pt, weight: "semibold", bottom-edge: "descender", it.body),
)
#show heading.where(level: 2): it => block(above: 12pt, below: 6pt, sticky: true, text(
  9.5pt,
  weight: "semibold",
  it.body,
))

// ─── Masthead ────────────────────────────────────────────────────────────────

#chrome(
  text(10pt, weight: "semibold", english[SatQuery AI]),
  text(8pt, fill: muted, L.generated),
)
#v(18pt)
#text(24pt, weight: "semibold", L.title)
#v(4pt)
#text(8pt, fill: muted)[#L.request #h(4pt) #english(mono(res.request_id))]
#v(14pt)

// Key facts, read left to right (or right to left) before the details.
#let fact(label, value) = block(width: 100%, inset: (y: 7pt), {
  text(7.5pt, fill: muted, label)
  linebreak()
  text(10pt, weight: "medium", value)
})
#let confidence-value = {
  ltr-run(pct(res.confidence))
  h(6pt)
  box(baseline: -1.5pt, width: 42pt, height: 3.5pt, radius: 2pt, fill: rule, place(
    left,
    box(width: 42pt * res.confidence, height: 3.5pt, radius: 2pt, fill: ink),
  ))
}
#let facts = (
  fact(L.task, L.task_name),
  fact(L.input, L.configuration_name),
  fact(L.confidence, confidence-value),
  fact(L.time, ltr-run(fmt-ms(res.execution_time_ms))),
  fact(L.language, L.language_name),
) + (
  if res.usage != none and res.usage.total_tokens != none {
    (fact(L.tokens, english(str(res.usage.total_tokens))),)
  } else { () }
)
#block(
  stroke: (top: 0.5pt + ink, bottom: 0.5pt + rule),
  grid(columns: (1fr,) * 3, column-gutter: 14pt, row-gutter: 0pt, ..facts),
)

// ─── Question and answer ─────────────────────────────────────────────────────

= #L.question

#text(
  10.5pt,
  lang: data.query_lang,
  dir: if data.query_rtl { rtl } else { ltr },
  data.query,
)
#if tr != none and tr.original_query != tr.english_query {
  v(2pt)
  block(text(8.5pt, fill: muted, {
    L.query_translated
    linebreak()
    english-par(text(fill: ink.lighten(20%), "“" + tr.english_query + "”"))
  }))
}

= #L.answer

#text(11pt, pick(if tr != none { tr.answer }, res.answer))

#if localised {
  v(4pt)
  block(width: 100%, fill: soft, radius: 3pt, inset: 10pt, {
    text(7.5pt, fill: muted, L.english_original)
    v(1pt)
    english-par(text(9pt, fill: rgb("#333333"), res.answer))
  })
}

// ─── Images with evidence ────────────────────────────────────────────────────

#let label-of(i) = {
  let local = if tr != none and tr.at("evidence_labels", default: none) != none {
    tr.evidence_labels.at(i)
  }
  let value = pick(local, res.evidence.at(i).label)
  if value == none { fill-index(L.region, i + 1) } else { value }
}
#let description-of(i) = {
  let local = if tr != none and tr.at("evidence_descriptions", default: none) != none {
    tr.evidence_descriptions.at(i)
  }
  pick(local, res.evidence.at(i).description)
}

// One preview with its boxes drawn at their normalised positions. Each box is
// a white stroke over a darker one so it reads on bright and dark imagery.
#let shot(img, w) = {
  let ht = w * img.aspect
  box(width: w, height: ht, clip: true, radius: 2pt, {
    image(img.path, width: w, height: ht, fit: "stretch")
    for (i, ev) in res.evidence.enumerate() {
      if ev.type == "bbox" and ev.data != none and ev.image_index == img.index {
        let b = ev.data
        let x = w * b.x_min
        let y = ht * b.y_min
        let bw = w * (b.x_max - b.x_min)
        let bh = ht * (b.y_max - b.y_min)
        place(top + left, dx: x, dy: y, rect(width: bw, height: bh, stroke: 2.2pt + rgb(0, 0, 0, 140)))
        place(top + left, dx: x, dy: y, rect(width: bw, height: bh, stroke: 1.1pt + white))
        let tag-y = if y > 11pt { y - 10.5pt } else { y + 1.5pt }
        place(top + left, dx: x, dy: tag-y, box(
          fill: white,
          inset: (x: 3pt, y: 2pt),
          radius: 1pt,
          text(6.5pt, fill: black, weight: "medium")[#str(i + 1) #h(2pt) #label-of(i)],
        ))
      }
    }
  })
}

#let caption(img) = {
  let info = res.inputs.find(x => x.index == img.index)
  set text(7.5pt, fill: muted)
  if res.inputs.len() > 1 {
    text(fill: ink, weight: "medium", fill-index(L.image, img.index + 1))
    h(8pt)
  }
  L.modality.at(info.modality)
  h(8pt)
  ltr-run[#info.width × #info.height px]
  if info.is_georeferenced {
    h(8pt)
    L.georeferenced
  }
  linebreak()
  english(mono(info.filename))
}

#if data.images.len() > 0 {
  [= #L.images]
  let full = 174mm
  if data.images.len() == 1 {
    let img = data.images.first()
    // Keep a single square scene from filling the whole page.
    let w = calc.min(full, 115mm / img.aspect)
    align(center, block(width: w, breakable: false, {
      shot(img, w)
      v(4pt)
      align(start, caption(img))
    }))
  } else {
    let w = (full - 8mm) / 2
    grid(
      columns: (w, w),
      column-gutter: 8mm,
      ..data.images.map(img => block(breakable: false, {
        shot(img, w)
        v(4pt)
        caption(img)
      }))
    )
  }
}

// ─── Evidence ────────────────────────────────────────────────────────────────

#if res.evidence.len() > 0 {
  [= #L.evidence]
  for (i, ev) in res.evidence.enumerate() {
    let spatial = ev.type == "bbox" and ev.data != none
    let marker = box(
      width: 14pt,
      height: 14pt,
      radius: 2pt,
      fill: if spatial { ink } else { none },
      stroke: if spatial { none } else { 0.5pt + faint },
      align(center + horizon, english(text(7pt, weight: "semibold", fill: if spatial { white } else { muted }, str(i + 1)))),
    )
    let body = {
      let lbl = if localised and tr != none and tr.at("evidence_labels", default: none) != none {
        tr.evidence_labels.at(i)
      } else { ev.label }
      if lbl != none { text(weight: "semibold", lbl); linebreak() }
      description-of(i)
      if localised {
        linebreak()
        english-par(text(8pt, fill: faint, {
          if ev.label != none [#ev.label: ]
          ev.description
        }))
      }
    }
    let side = {
      set text(8pt, fill: muted)
      let parts = ()
      if ev.confidence != none { parts.push(ltr-run(pct(ev.confidence))) }
      if res.inputs.len() > 1 { parts.push(fill-index(L.image, ev.image_index + 1)) }
      parts.join(linebreak())
    }
    block(width: 100%, breakable: false, inset: (y: 7pt), stroke: (bottom: 0.4pt + rule), grid(
      columns: (14pt, 1fr, auto),
      column-gutter: 9pt,
      align: (top, top, top + end),
      marker, body, side,
    ))
  }
  if L.no_spatial != none {
    v(4pt)
    text(8pt, fill: muted, L.no_spatial)
  }
}

// ─── Controller notes ────────────────────────────────────────────────────────

#let warnings = pick(if tr != none { tr.at("warnings", default: none) }, res.trace.warnings)
#if warnings.len() > 0 {
  [= #L.controller_notes]
  list(marker: text(fill: muted)[–], spacing: 0.6em, ..warnings)
}

// ─── Execution trace ─────────────────────────────────────────────────────────

= #L.trace

#let rationale = pick(if tr != none { tr.at("routing_rationale", default: none) }, res.trace.routing_rationale)
#grid(
  columns: (auto, 1fr),
  column-gutter: 14pt,
  row-gutter: 8pt,
  text(fill: muted, L.routing),
  {
    L.task_name
    if rationale != none and rationale != "" {
      linebreak()
      text(8.5pt, fill: muted, rationale)
    }
  },
  text(fill: muted, L.tools), english(mono(res.trace.selected_tools.join(", "))),
)
#v(10pt)

#for (i, step) in res.trace.steps.enumerate() {
  block(width: 100%, breakable: false, inset: (y: 6pt), stroke: (top: 0.4pt + rule), english-par(grid(
    columns: (16pt, 60pt, 1fr, auto),
    column-gutter: 8pt,
    text(8pt, fill: faint, str(i + 1)),
    text(weight: "semibold", step.stage),
    {
      mono(step.tool)
      if step.detail != "" {
        linebreak()
        text(8pt, fill: muted, dir: auto, step.detail)
      }
    },
    align(end, text(8pt, fill: muted, ltr-run(fmt-ms(step.duration_ms)))),
  )))
}
