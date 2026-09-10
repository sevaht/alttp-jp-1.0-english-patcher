"use strict";

/**
 * Renders text using the real SNES dialogue VWF glyph bitmaps (a sprite
 * sheet decoded server-side by FontSheet, one glyph per WidthTable code)
 * and computes the same per-character pixel widths the in-game engine
 * uses. There is no auto-wrap in the real engine -- MAX_LINE_WIDTH is a
 * hard limit, not a wrap point (see script_reviewer.py's WidthTable).
 */
class GlyphRenderer {
  constructor(fontImage, widths, codeOf, glyphW, glyphH, maxLineWidth,
              nameToken, nameChars, namePlaceholderWord, maxNameCharWidth,
              digitOperands, maxDigitWidth, gbaOpPlaceholderCode,
              linkFaceToken, linkFaceCodes) {
    this.fontImage = fontImage;
    this.widths = widths; // {code: px}
    this.codeOf = codeOf; // {char: code}
    this.glyphW = glyphW;
    this.glyphH = glyphH;
    this.maxLineWidth = maxLineWidth;
    this.nameToken = nameToken; // "[LINK]"
    this.nameChars = nameChars; // 6 -- worst-case name length
    // real word (all real font glyphs -- "[LINK]" itself can't be spelled
    // out, the font has no "[" or "]") drawn in place of [LINK], but each
    // letter advances by maxNameCharWidth instead of its own natural
    // width, so the total always equals the true worst-case name width.
    this.namePlaceholderWord = namePlaceholderWord; // "PLAYER"
    this.maxNameCharWidth = maxNameCharWidth; // 7
    // "0123" -- only these N in [#N] are hardware-valid (ParseText_WriteBCD
    // reads a 2-byte BCD buffer; see script_reviewer.py's WidthTable docstring)
    this.digitOperands = digitOperands;
    this.maxDigitWidth = maxDigitWidth; // 6
    this.gbaOpPlaceholderCode = gbaOpPlaceholderCode; // synthetic hatched-box glyph
    this.linkFaceToken = linkFaceToken; // the grimacing-face emoji
    this.linkFaceCodes = linkFaceCodes; // [0x4A, 0x4B] -- fixed, real, no guessing
  }

  /** text -> [{kind: "char", ch} | {kind: "name"} | {kind: "digit", ch} |
   * {kind: "opTag"} | {kind: "linkFace"} | {kind: "rawCode", code} |
   * {kind: "badBracket", ch}, ...]. An exact `[LINK]` run is one "name"
   * token (the player's actual runtime name, worst-cased); `[#0]`-`[#3]`
   * is one "digit" token -- ch is the slot digit itself ("0"-"3"), drawn
   * as that digit so four adjacent placeholders read as four distinct
   * digit positions of one shared value rather than four identical
   * unknowns, but the WIDTH is always maxDigitWidth regardless of which
   * digit is drawn (worst-cased, same as the name placeholder); `[OP:...]`
   * (an unresolved GBA control code -- see gba_script_diff.py's own tag)
   * is one "opTag" token, drawn as the synthetic hatched-box glyph since
   * it isn't a stand-in for any known real character; the Link-face emoji
   * is one "linkFace" token (expands to linkFaceCodes, both real, no
   * guessing needed); `[C:XX]` is one "rawCode" token (draws/measures
   * exactly ROM code XX -- used where a comment character is ambiguous,
   * see script_reviewer.py's WidthTable docstring on the heart icons);
   * any OTHER `[`/`]` is invalid placeholder syntax. */
  tokenize(text) {
    const tokens = [];
    let i = 0;
    while (i < text.length) {
      if (text.startsWith(this.nameToken, i)) {
        tokens.push({ kind: "name" });
        i += this.nameToken.length;
        continue;
      }
      if (text.startsWith(this.linkFaceToken, i)) {
        tokens.push({ kind: "linkFace" });
        i += this.linkFaceToken.length;
        continue;
      }
      if (text[i] === "[" && text[i + 1] === "#" && this.digitOperands.includes(text[i + 2]) && text[i + 3] === "]") {
        tokens.push({ kind: "digit", ch: text[i + 2] });
        i += 4;
        continue;
      }
      if (text.startsWith("[OP:", i)) {
        const end = text.indexOf("]", i);
        tokens.push({ kind: "opTag" });
        i = end === -1 ? text.length : end + 1;
        continue;
      }
      if (text[i] === "[" && text[i + 1] === "C" && text[i + 2] === ":" && text[i + 5] === "]") {
        const code = parseInt(text.slice(i + 3, i + 5), 16);
        if (!Number.isNaN(code)) {
          tokens.push({ kind: "rawCode", code });
          i += 6;
          continue;
        }
      }
      if (text[i] === "[" || text[i] === "]") {
        tokens.push({ kind: "badBracket", ch: text[i] });
        i += 1;
        continue;
      }
      tokens.push({ kind: "char", ch: text[i] });
      i += 1;
    }
    return tokens;
  }

  /** Pixel-width contribution of one token in isolation (0 for a
   * badBracket or an unmeasurable/unknown glyph -- lineWidth() and
   * firstWordWidth() both flag those separately rather than fold them into
   * a width number). */
  _tokenWidth(tok) {
    if (tok.kind === "name") return this.nameChars * this.maxNameCharWidth;
    if (tok.kind === "digit") return this.maxDigitWidth;
    if (tok.kind === "opTag") return this.widths[this.gbaOpPlaceholderCode];
    if (tok.kind === "linkFace") {
      return this.linkFaceCodes.reduce((sum, c) => sum + this.widths[c], 0);
    }
    if (tok.kind === "rawCode") return this.widths[tok.code];
    if (tok.kind === "badBracket") return 0;
    const code = this.codeOf[tok.ch];
    return code === undefined ? 0 : this.widths[code];
  }

  /** [pixel width, sorted unmeasurable chars, bad-bracket chars found]. */
  lineWidth(text) {
    let width = 0;
    const unknown = new Set();
    const badBrackets = [];
    for (const tok of this.tokenize(text)) {
      if (tok.kind === "badBracket") {
        badBrackets.push(tok.ch);
      } else if (tok.kind === "char" && this.codeOf[tok.ch] === undefined) {
        unknown.add(tok.ch);
      } else {
        width += this._tokenWidth(tok);
      }
    }
    return [width, [...unknown].sort(), badBrackets];
  }

  /** Pixel width of `text`'s first space-delimited "word" -- tokenize()
   * first so a bracketed placeholder token (e.g. a multi-argument
   * `[OP:...]` tag) is never split on a literal space inside its own
   * brackets. 0 for an empty line or one starting with a space. Used for
   * the reflow marker: "would the next line's first word fit on this
   * line?" is a real per-word question, not a per-character one. */
  firstWordWidth(text) {
    let width = 0;
    for (const tok of this.tokenize(text)) {
      if (tok.kind === "char" && tok.ch === " ") break;
      width += this._tokenWidth(tok);
    }
    return width;
  }

  /** Renders `lines` (array of strings, one per in-game line) onto a
   * fresh canvas at the given integer pixel scale and returns it. A dashed
   * separator marks every 3rd line -- the real textbox only ever shows 3
   * lines at once (RenderText_SetLine cycles through exactly 3 row slots;
   * see script_reviewer.py's WidthTable docstring on MAX_LINE_WIDTH for
   * the buffer-geometry source), so a 4th+ line means the player has to
   * press a button to scroll/advance before seeing it. */
  static LINES_PER_PAGE = 3;

  renderLines(lines, scale = 2) {
    const lineHeight = this.glyphH * scale;
    const gapHeight = 6 * scale;
    const widths = lines.map((l) => this.lineWidth(l)[0]);
    // +1: room for the reflow marker's own 1px column, drawn flush at
    // maxLineWidth -- without this it's clipped off the canvas whenever no
    // line actually overflows past maxLineWidth (the common case).
    const maxWidth = Math.max(this.maxLineWidth + 1, ...widths, 1);
    const pageBreaks = Math.max(0, Math.ceil(lines.length / GlyphRenderer.LINES_PER_PAGE) - 1);
    const canvas = document.createElement("canvas");
    canvas.width = maxWidth * scale;
    canvas.height = Math.max(1, lines.length) * lineHeight + pageBreaks * gapHeight;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false; // this is pixel art -- bilinear
    // scaling would sample across glyph boundaries in the sprite sheet,
    // producing exactly the grey bleeding/blur between characters seen
    // without this.
    const spaceWidth = this.widths[this.codeOf[" "]] || 0;
    let y = 0;
    lines.forEach((line, i) => {
      this._drawLine(ctx, line, 0, y, scale);
      const isPageEnd = (i + 1) % GlyphRenderer.LINES_PER_PAGE === 0;
      // Reflow marker: a real, same-page next line whose first word would
      // still fit here (plus the space that joining them needs) if you
      // pulled it up -- a manual reflow aid, not an error/overflow signal.
      // Drawn at the row's own render-area edge (maxLineWidth), not this
      // line's own text-end -- a fixed reference column across every row,
      // not tied to how short this particular line happens to be. Never
      // overlaps this line's own text: the fit check already guarantees
      // widths[i] <= maxLineWidth whenever the marker is drawn at all.
      const nextLine = i < lines.length - 1 && !isPageEnd ? lines[i + 1] : null;
      if (nextLine !== null && nextLine.trim() !== "") {
        const nextWordWidth = this.firstWordWidth(nextLine);
        if (widths[i] + spaceWidth + nextWordWidth <= this.maxLineWidth) {
          this._drawReflowMarker(ctx, this.maxLineWidth, y, lineHeight, scale);
        }
      }
      y += lineHeight;
      if (isPageEnd && i !== lines.length - 1) {
        this._drawPageBreak(ctx, canvas.width, y, gapHeight, scale);
        y += gapHeight;
      }
    });
    return canvas;
  }

  /** A 1-in-game-pixel-wide yellow bar at column `x` (unscaled game
   * pixels, expected to be maxLineWidth -- the row's own render-area
   * edge, the same for every row regardless of that row's own text
   * length). */
  _drawReflowMarker(ctx, x, y, lineHeight, scale) {
    ctx.save();
    ctx.fillStyle = "#f4d03f";
    ctx.fillRect(x * scale, y, scale, lineHeight);
    ctx.restore();
  }

  _drawPageBreak(ctx, width, y, gapHeight, scale) {
    const midY = y + gapHeight / 2;
    ctx.save();
    ctx.strokeStyle = "#6ba3d6";
    ctx.lineWidth = Math.max(1, scale);
    ctx.setLineDash([4 * scale, 3 * scale]);
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();
    ctx.restore();
  }

  _drawLine(ctx, text, x0, y0, scale) {
    let x = x0;
    for (const tok of this.tokenize(text)) {
      if (tok.kind === "name") {
        for (const ch of this.namePlaceholderWord) {
          const code = this.codeOf[ch];
          x = this._drawGlyph(ctx, code, x, y0, scale, this.maxNameCharWidth);
        }
      } else if (tok.kind === "digit") {
        const code = this.codeOf[tok.ch];
        x = this._drawGlyph(ctx, code, x, y0, scale, this.maxDigitWidth);
      } else if (tok.kind === "opTag") {
        x = this._drawGlyph(ctx, this.gbaOpPlaceholderCode, x, y0, scale);
      } else if (tok.kind === "linkFace") {
        for (const code of this.linkFaceCodes) {
          x = this._drawGlyph(ctx, code, x, y0, scale);
        }
      } else if (tok.kind === "rawCode") {
        x = this._drawGlyph(ctx, tok.code, x, y0, scale);
      } else if (tok.kind === "badBracket") {
        x += 6 * scale; // can't render it -- the badge list says why
      } else {
        const code = this.codeOf[tok.ch];
        if (code === undefined) {
          x += 6 * scale; // unknown-glyph placeholder gap
        } else {
          x = this._drawGlyph(ctx, code, x, y0, scale);
        }
      }
    }
  }

  /** `advanceOverride`, if given, replaces the glyph's own table width for
   * cursor advancement (used only for the [LINK] worst-case placeholder --
   * see the constructor comment) while still drawing that glyph's real
   * bitmap. */
  _drawGlyph(ctx, code, x, y0, scale, advanceOverride) {
    ctx.drawImage(
      this.fontImage,
      code * this.glyphW, 0, this.glyphW, this.glyphH,
      x, y0, this.glyphW * scale, this.glyphH * scale,
    );
    const advance = advanceOverride !== undefined ? advanceOverride : this.widths[code];
    return x + advance * scale;
  }
}

/**
 * Word-level diff producing {left, right} HTML strings with <ins>/<del>
 * markup, mirroring gba_script_diff.py's _word_diff_html so the visual
 * language matches the static wording.html report.
 */
class DiffHighlighter {
  /** aLines/bLines, if given, are that side's real per-line array (e.g.
   * row.snes_lines, or the Proposed textarea's current lines) -- the
   * output reproduces line breaks at exactly those same word positions
   * (via a literal "\n", relying on the caller's white-space: pre-wrap),
   * so the highlighted text underneath wraps identically to the rendered
   * image above it instead of re-wrapping to the container's width. */
  static diff(aText, bText, aLines = null, bLines = null) {
    const a = aText.split(" ");
    const b = bText.split(" ");
    const ops = DiffHighlighter._lcsOps(a, b);
    const aBoundaries = aLines ? DiffHighlighter._lineBoundaries(aLines) : new Set();
    const bBoundaries = bLines ? DiffHighlighter._lineBoundaries(bLines) : new Set();

    let left = "";
    let right = "";
    let aIdx = 0;
    let bIdx = 0;
    let aStarted = false;
    let bStarted = false;
    for (const [tag, aSeg, bSeg] of ops) {
      if (aSeg.length) {
        let inner = "";
        for (const w of aSeg) {
          if (aStarted) inner += aBoundaries.has(aIdx) ? "\n" : " ";
          inner += DiffHighlighter._escape(w);
          aIdx += 1;
          aStarted = true;
        }
        left += tag === "equal" ? inner : `<del>${inner}</del>`;
      }
      if (bSeg.length) {
        let inner = "";
        for (const w of bSeg) {
          if (bStarted) inner += bBoundaries.has(bIdx) ? "\n" : " ";
          inner += DiffHighlighter._escape(w);
          bIdx += 1;
          bStarted = true;
        }
        right += tag === "equal" ? inner : `<ins>${inner}</ins>`;
      }
    }
    return { left, right };
  }

  /** {word-index, ...} where a real line break falls in `lines` once
   * flattened the same way the diff itself flattens text (join(" ") then
   * split(" ") -- so a blank line, like everywhere else in this file,
   * counts as one empty-string "word", keeping the indices in step). */
  static _lineBoundaries(lines) {
    const boundaries = new Set();
    let idx = 0;
    for (let i = 0; i < lines.length - 1; i++) {
      idx += lines[i].split(" ").length;
      boundaries.add(idx);
    }
    return boundaries;
  }

  static _escape(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  /** Plain O(n*m) LCS-based diff (word counts here are small: dozens, not
   * thousands) producing merged equal/replace runs. */
  static _lcsOps(a, b) {
    const n = a.length;
    const m = b.length;
    const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    const ops = [];
    let i = 0;
    let j = 0;
    let aBuf = [];
    let bBuf = [];
    const flush = () => {
      if (aBuf.length || bBuf.length) {
        ops.push(["replace", aBuf, bBuf]);
        aBuf = [];
        bBuf = [];
      }
    };
    while (i < n && j < m) {
      if (a[i] === b[j]) {
        flush();
        ops.push(["equal", [a[i]], [b[j]]]);
        i++;
        j++;
      } else if (dp[i + 1][j] >= dp[i][j + 1]) {
        aBuf.push(a[i]);
        i++;
      } else {
        bBuf.push(b[j]);
        j++;
      }
    }
    while (i < n) aBuf.push(a[i++]);
    while (j < m) bBuf.push(b[j++]);
    flush();
    // merge consecutive same-tag ops (mostly runs of "equal" single words)
    const merged = [];
    for (const op of ops) {
      const last = merged[merged.length - 1];
      if (last && last[0] === op[0]) {
        last[1] = last[1].concat(op[1]);
        last[2] = last[2].concat(op[2]);
      } else {
        merged.push([op[0], op[1].slice(), op[2].slice()]);
      }
    }
    return merged;
  }
}

/** One message's DOM row: renders SNES/Proposed/GBA panels, wires the
 * editable Proposed textarea to live re-render + width-check + diff +
 * autosave. */
class MessageRow {
  constructor(data, renderer, state, onDirty) {
    this.data = data; // {key, snes_id, gba_id, kind, snes_lines, gba_lines, proposed_default}
    this.renderer = renderer;
    this.onDirty = onDirty; // called with (key, {proposed, completed}) to persist

    const saved = state[data.key] || {};
    this.proposedText = saved.proposed ?? data.proposed_default;
    this.completed = !!saved.completed;

    this.el = this._buildDom();
    if (data.kind === "changed") this._wireEditing();
    this._renderAll();
  }

  matchesFilter(query) {
    if (!query) return true;
    const q = query.toLowerCase();
    if (`$${(this.data.snes_id ?? "").toString(16).padStart(4, "0")}`.toLowerCase().includes(q)) return true;
    return this.el.textContent.toLowerCase().includes(q);
  }

  setHideCompleted(hide) {
    this.el.classList.toggle("hidden", hide && this.completed);
  }

  /** Sets completed state + keeps the checkbox and the row's green/red
   * outline in sync. Doesn't persist -- callers do that explicitly. */
  _setCompleted(value) {
    this.completed = value;
    this.el.classList.toggle("completed", value);
    const box = this.el.querySelector(".completed-box");
    if (box) box.checked = value;
  }

  _buildDom() {
    const row = document.createElement("div");
    row.className = `row kind-${this.data.kind}${this.completed ? " completed" : ""}`;
    row.dataset.key = this.data.key;

    const idLabel = this.data.snes_id !== null
      ? `$${this.data.snes_id.toString(16).toUpperCase().padStart(4, "0")}`
      : "GBA-only";
    const gbaLabel = this.data.gba_id !== null ? `gba ${this.data.gba_id}` : "";

    row.innerHTML = `
      <div class="row-head">
        <span>${idLabel}</span>
        <span>${gbaLabel}</span>
        <label class="completed"><input type="checkbox" class="completed-box"> done</label>
      </div>
      <div class="panels"></div>
    `;
    row.querySelector(".completed-box").checked = this.completed;
    row.querySelector(".completed-box").addEventListener("change", (e) => {
      this._setCompleted(e.target.checked);
      this.onDirty(this.data.key, { proposed: this._savedProposedValue(), completed: this.completed });
    });

    const panels = row.querySelector(".panels");
    if (this.data.kind === "changed") {
      panels.appendChild(this._buildSnesPanel());
      panels.appendChild(this._buildProposedPanel());
      panels.appendChild(this._buildGbaPanel());
    } else {
      panels.appendChild(this._buildGbaPanel(true));
    }
    return row;
  }

  _buildSnesPanel() {
    const panel = document.createElement("div");
    panel.className = "panel snes";
    panel.innerHTML = `
      <div class="label">US SNES</div>
      <canvas></canvas>
      <div class="text plain"></div>
    `;
    panel.querySelector(".plain").textContent = this.data.snes_lines.join("\n");
    return panel;
  }

  _buildProposedPanel() {
    const panel = document.createElement("div");
    panel.className = "panel proposed";
    panel.innerHTML = `
      <div class="label">Proposed<span class="tag">editable</span>
        <span class="proposed-actions">
          <span class="matches-snes-badge hidden">matches SNES</span>
          <button type="button" class="restore-snes-btn"
                  title="Replace this Proposed text with the exact SNES original">Restore SNES</button>
        </span>
      </div>
      <canvas></canvas>
      <textarea spellcheck="false"></textarea>
      <div class="error-banner hidden"></div>
      <div class="diff-preview"></div>
      <div class="badges"></div>
    `;
    panel.querySelector("textarea").value = this.proposedText;
    return panel;
  }

  _buildGbaPanel(standalone = false) {
    const panel = document.createElement("div");
    panel.className = "panel gba";
    panel.innerHTML = `
      <div class="label">GBA<span class="tag">rendered w/ SNES font</span></div>
      <canvas></canvas>
      <div class="text diff"></div>
      ${standalone ? '<div class="note-inline">no SNES counterpart -- new in GBA</div>' : ""}
    `;
    return panel;
  }

  _wireEditing() {
    const textarea = this.el.querySelector(".proposed textarea");
    let debounce = null;
    const persistSoon = () => {
      clearTimeout(debounce);
      debounce = setTimeout(() => {
        this.onDirty(this.data.key, { proposed: this._savedProposedValue(), completed: this.completed });
      }, 400);
    };
    textarea.addEventListener("input", () => {
      this.proposedText = textarea.value;
      this._renderProposed();
      persistSoon();
    });

    const restoreBtn = this.el.querySelector(".restore-snes-btn");
    restoreBtn.addEventListener("click", () => {
      this.proposedText = this.data.snes_lines.join("\n");
      textarea.value = this.proposedText;
      this._renderProposed();
      // a discrete action, not a keystroke -- persist right away rather
      // than waiting out the typing debounce.
      clearTimeout(debounce);
      this.onDirty(this.data.key, { proposed: this._savedProposedValue(), completed: this.completed });
    });
  }

  /** null (= "use current curated default") when the field matches the
   * default verbatim, matching ReviewState's documented semantics. */
  _savedProposedValue() {
    return this.proposedText === this.data.proposed_default ? null : this.proposedText;
  }

  _renderAll() {
    const snesJoined = this.data.snes_lines.join(" ");
    const gbaJoined = this.data.gba_lines.join(" ");

    const snesCanvas = this.el.querySelector(".panel.snes canvas");
    if (snesCanvas) snesCanvas.replaceWith(this.renderer.renderLines(this.data.snes_lines));

    const gbaCanvas = this.el.querySelector(".panel.gba canvas");
    if (gbaCanvas) gbaCanvas.replaceWith(this.renderer.renderLines(this.data.gba_lines));

    const gbaDiffEl = this.el.querySelector(".panel.gba .diff");
    if (gbaDiffEl) {
      gbaDiffEl.innerHTML =
        DiffHighlighter.diff(snesJoined, gbaJoined, this.data.snes_lines, this.data.gba_lines).right;
    }

    if (this.data.kind === "changed") this._renderProposed();
  }

  /** "".split("\n") is ["": length 1] in JS, which would falsely count an
   * entirely empty field as "1 line" -- a genuinely empty field has 0
   * lines, matching a message with no text at all (e.g. $0000). */
  static _splitLines(text) {
    return text === "" ? [] : text.split("\n");
  }

  _renderProposed() {
    const lines = MessageRow._splitLines(this.proposedText);
    const canvas = this.el.querySelector(".panel.proposed canvas");
    canvas.replaceWith(this.renderer.renderLines(lines));

    const snesJoined = this.data.snes_lines.join(" ");
    const proposedJoined = lines.join(" ");
    this.el.querySelector(".panel.proposed .diff-preview").innerHTML =
      DiffHighlighter.diff(snesJoined, proposedJoined, this.data.snes_lines, lines).right;

    // Exact match, including line breaks -- not just the same wording
    // reflowed differently -- since that's what "identical to SNES" means.
    const matchesSnes = this.proposedText === this.data.snes_lines.join("\n");
    this.el.querySelector(".matches-snes-badge").classList.toggle("hidden", !matchesSnes);

    const hasError = this._renderBadges(lines);
    this._applyErrorState(hasError);
  }

  /** Renders the width/unmeasured/bracket badges for `lines` and returns
   * whether any BLOCKING error was found (overflow, an unmeasurable
   * character, or invalid bracket syntax -- NOT the extra-page-count
   * badge, which is advisory only and doesn't block "done"). */
  _renderBadges(lines) {
    const box = this.el.querySelector(".panel.proposed .badges");
    box.innerHTML = "";
    let hasError = false;
    lines.forEach((line, i) => {
      const [width, unknown, badBrackets] = this.renderer.lineWidth(line);
      const over = width > this.renderer.maxLineWidth;
      if (over || unknown.length || badBrackets.length) hasError = true;
      const badge = document.createElement("span");
      badge.className = `badge${over ? " bad" : ""}`;
      badge.textContent = `line ${i + 1}: ${width}/${this.renderer.maxLineWidth}px`;
      box.appendChild(badge);
      if (unknown.length) {
        const warn = document.createElement("span");
        warn.className = "badge bad";
        warn.textContent = `unmeasured: ${unknown.join(" ")}`;
        box.appendChild(warn);
      }
      if (badBrackets.length) {
        const warn = document.createElement("span");
        warn.className = "badge bad";
        warn.textContent = `invalid bracket (only ${this.renderer.nameToken} is supported): ${badBrackets.join(" ")}`;
        box.appendChild(warn);
      }
    });
    // Pages, not raw line count: the box holds LINES_PER_PAGE (3) lines at
    // once, so e.g. 2 lines -> 3 lines is still one page, not "an extra
    // page" -- only crossing a multiple of 3 actually adds one.
    const perPage = GlyphRenderer.LINES_PER_PAGE;
    const originalLines = this.data.snes_lines.length;
    const originalPages = Math.ceil(originalLines / perPage);
    const editedPages = Math.ceil(lines.length / perPage);
    if (editedPages > originalPages) {
      const pageBadge = document.createElement("span");
      pageBadge.className = "badge bad";
      const noun = (n) => (n === 1 ? "line" : "lines");
      pageBadge.textContent =
        `${lines.length} ${noun(lines.length)} (original ${originalLines} ${noun(originalLines)}) `
        + `-- ${editedPages} pages vs ${originalPages}, may need an extra page`;
      box.appendChild(pageBadge);
      // advisory only -- deliberately doesn't set hasError/block "done"
    }
    return hasError;
  }

  /** Makes a blocking error hard to miss (banner + red textarea border)
   * and disables "done" -- forcibly unchecking + persisting if a row that
   * was previously marked done now has one (e.g. GLOBAL_FIXES or the
   * width table changed since it was checked off). */
  _applyErrorState(hasError) {
    this.hasError = hasError;
    const textarea = this.el.querySelector(".panel.proposed textarea");
    textarea.classList.toggle("has-error", hasError);
    const banner = this.el.querySelector(".panel.proposed .error-banner");
    banner.classList.toggle("hidden", !hasError);
    banner.textContent = hasError
      ? "fix the errors below before marking this done" : "";

    const box = this.el.querySelector(".completed-box");
    box.disabled = hasError;
    box.closest(".completed").classList.toggle("disabled", hasError);
    box.title = hasError ? "fix the Proposed text's errors first" : "";
    if (hasError && this.completed) {
      this._setCompleted(false);
      this.onDirty(this.data.key, { proposed: this._savedProposedValue(), completed: this.completed });
    }
  }
}

class ReviewApp {
  async start() {
    this._pinColumnHeadersBelowMainHeader();

    const resp = await fetch("/api/data");
    const data = await resp.json();

    const fontImage = new Image();
    const fontLoaded = new Promise((resolve) => { fontImage.onload = resolve; });
    fontImage.src = "/api/font.png";
    await fontLoaded;

    this.renderer = new GlyphRenderer(
      fontImage, data.widths, data.char_codes,
      data.glyph_width, data.glyph_height, data.max_line_width,
      data.name_token, data.name_chars, data.name_placeholder_word, data.max_name_char_width,
      data.digit_operands, data.max_digit_width, data.gba_op_placeholder_code,
      data.link_face_token, data.link_face_codes,
    );

    this.saveIndicator = document.getElementById("save-indicator");
    this.rowsEl = document.getElementById("rows");
    this.rows = data.messages.map(
      (m) => new MessageRow(m, this.renderer, data.state, this._save.bind(this)),
    );
    this.rows.forEach((r) => this.rowsEl.appendChild(r.el));

    this._wireFilter();
    this._updateCount();
  }

  /** Docks .column-headers directly under the real <header>, using its
   * MEASURED height rather than a hardcoded guess (the header's own
   * height isn't fixed -- e.g. it can wrap on a narrow window). */
  _pinColumnHeadersBelowMainHeader() {
    const header = document.querySelector("header");
    const set = () => {
      document.documentElement.style.setProperty("--header-height", `${header.offsetHeight}px`);
    };
    set();
    window.addEventListener("resize", set);
  }

  _wireFilter() {
    const filter = document.getElementById("filter");
    const hideCompleted = document.getElementById("hide-completed");
    const apply = () => {
      const q = filter.value.trim();
      for (const row of this.rows) {
        const matches = row.matchesFilter(q);
        row.el.classList.toggle("hidden", !matches);
        if (matches) row.setHideCompleted(hideCompleted.checked);
      }
      this._updateCount();
    };
    filter.addEventListener("input", apply);
    hideCompleted.addEventListener("change", apply);
    apply();
  }

  _updateCount() {
    const shown = this.rows.filter((r) => !r.el.classList.contains("hidden")).length;
    const done = this.rows.filter((r) => r.completed).length;
    document.getElementById("count").textContent =
      `${shown} / ${this.rows.length} shown, ${done} / ${this.rows.length} completed`;
  }

  async _save(key, payload) {
    this.saveIndicator.textContent = "saving…";
    try {
      await fetch("/api/state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key, ...payload }),
      });
      this.saveIndicator.textContent = "saved";
    } catch (e) {
      this.saveIndicator.textContent = "save failed";
    }
    this._updateCount();
  }
}

new ReviewApp().start();
