import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath, pathToFileURL } from 'node:url';

async function loadMarked() {
  const candidates = [
    process.env.MARKED_MODULE,
    new URL('../vendor/marked/lib/marked.esm.js', import.meta.url).href,
    'marked',
    path.join(process.cwd(), 'node_modules/marked/lib/marked.esm.js'),
  ].filter(Boolean);
  const failures = [];
  for (const candidate of candidates) {
    try {
      const specifier = candidate.startsWith('file:')
        ? candidate
        : candidate.includes('/') ? pathToFileURL(candidate).href : candidate;
      const module = await import(specifier);
      if (module.marked?.parse) return module.marked;
      failures.push(`${candidate}: missing named export marked`);
    } catch (error) {
      failures.push(`${candidate}: ${error.code || error.message}`);
    }
  }
  throw new Error(`无法加载 marked。安装 marked 或设置 MARKED_MODULE。\n${failures.join('\n')}`);
}

const marked = await loadMarked();

const renderMarkdown = (source) => String(marked.parse(source, { gfm: true, breaks: false }));

const htmlEscape = (value) => String(value)
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;')
  .replaceAll("'", '&#39;');

const jsonForScript = (value) => JSON.stringify(value)
  .replaceAll('<', '\\u003c')
  .replaceAll('\u2028', '\\u2028')
  .replaceAll('\u2029', '\\u2029');

function normalizeTopology(raw, field) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new TypeError(`${field} 必须是对象`);
  }
  const caption = requireNonEmptyString(raw.caption, `${field}.caption`);
  if (!Array.isArray(raw.nodes) || raw.nodes.length < 2 || raw.nodes.length > 8) {
    throw new TypeError(`${field}.nodes 必须包含 2–8 个节点`);
  }
  const nodeIds = new Set();
  const positions = new Set();
  const nodes = raw.nodes.map((node, index) => {
    if (!node || typeof node !== 'object' || Array.isArray(node)) {
      throw new TypeError(`${field}.nodes[${index}] 必须是对象`);
    }
    const id = requireNonEmptyString(node.id, `${field}.nodes[${index}].id`);
    if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(id)) {
      throw new TypeError(`${field}.nodes[${index}].id 必须以字母开头且只含字母、数字、_、-`);
    }
    if (nodeIds.has(id)) throw new Error(`${field} 中节点 id ${id} 重复`);
    if (!Number.isInteger(node.row) || node.row < 0 || !Number.isInteger(node.column) || node.column < 0) {
      throw new TypeError(`${field}.nodes[${index}] 的 row/column 必须是非负整数`);
    }
    const position = `${node.row}:${node.column}`;
    if (positions.has(position)) throw new Error(`${field} 中位置 ${position} 被多个节点占用`);
    nodeIds.add(id);
    positions.add(position);
    return {
      id,
      label: requireNonEmptyString(node.label, `${field}.nodes[${index}].label`),
      row: node.row,
      column: node.column,
    };
  });
  if (!Array.isArray(raw.edges) || raw.edges.length < 1) {
    throw new TypeError(`${field}.edges 必须至少包含一条边`);
  }
  const edges = raw.edges.map((edge, index) => {
    if (!edge || typeof edge !== 'object' || Array.isArray(edge)) {
      throw new TypeError(`${field}.edges[${index}] 必须是对象`);
    }
    const from = requireNonEmptyString(edge.from, `${field}.edges[${index}].from`);
    const to = requireNonEmptyString(edge.to, `${field}.edges[${index}].to`);
    if (!nodeIds.has(from) || !nodeIds.has(to)) {
      throw new Error(`${field}.edges[${index}] 引用了不存在的节点`);
    }
    if (from === to) throw new Error(`${field}.edges[${index}] 不能连接节点自身`);
    return { from, to, label: requireNonEmptyString(edge.label, `${field}.edges[${index}].label`) };
  });
  if (!Array.isArray(raw.groups) || raw.groups.length < 1) {
    throw new TypeError(`${field}.groups 必须至少包含一个当前术语边界`);
  }
  const groups = raw.groups.map((group, index) => {
    if (!group || typeof group !== 'object' || Array.isArray(group)) {
      throw new TypeError(`${field}.groups[${index}] 必须是对象`);
    }
    if (!Array.isArray(group.members) || group.members.length < 2) {
      throw new TypeError(`${field}.groups[${index}].members 必须至少包含两个节点`);
    }
    const members = [...new Set(group.members.map((member, memberIndex) => {
      const id = requireNonEmptyString(member, `${field}.groups[${index}].members[${memberIndex}]`);
      if (!nodeIds.has(id)) throw new Error(`${field}.groups[${index}] 引用了不存在的节点 ${id}`);
      return id;
    }))];
    return { label: requireNonEmptyString(group.label, `${field}.groups[${index}].label`), members };
  });
  return { caption, nodes, edges, groups };
}

function wrapTopologyLabel(value, maximumUnits) {
  const units = (character) => /[\u0000-\u00ff]/.test(character) ? 0.58 : 1;
  const lines = [];
  let line = '';
  let width = 0;
  for (const character of value) {
    const next = units(character);
    if (line && width + next > maximumUnits) {
      lines.push(line);
      line = character;
      width = next;
    } else {
      line += character;
      width += next;
    }
  }
  if (line) lines.push(line);
  return lines.slice(0, 3);
}

function renderTopologySvg(topology, cardKey) {
  const canvasWidth = 520;
  const marginX = 28;
  const top = 42;
  const nodeHeight = 54;
  const rowGap = 64;
  const columnGap = 24;
  const columnCount = Math.max(...topology.nodes.map((node) => node.column)) + 1;
  const rowCount = Math.max(...topology.nodes.map((node) => node.row)) + 1;
  const nodeWidth = (canvasWidth - (2 * marginX) - ((columnCount - 1) * columnGap)) / columnCount;
  if (nodeWidth < 92) throw new Error(`topology ${cardKey} 列数过多，节点宽度不足`);
  const canvasHeight = top + (rowCount * nodeHeight) + ((rowCount - 1) * rowGap) + 34;
  const positions = new Map(topology.nodes.map((node) => [node.id, {
    ...node,
    x: marginX + node.column * (nodeWidth + columnGap),
    y: top + node.row * (nodeHeight + rowGap),
  }]));
  const markerId = `arrow-${cardKey.replace(/[^A-Za-z0-9_-]/g, '-')}`;

  const groups = topology.groups.map((group) => {
    const members = group.members.map((id) => positions.get(id));
    const minX = Math.max(4, Math.min(...members.map((node) => node.x)) - 12);
    const minY = Math.max(22, Math.min(...members.map((node) => node.y)) - 22);
    const maxX = Math.min(canvasWidth - 4, Math.max(...members.map((node) => node.x + nodeWidth)) + 12);
    const maxY = Math.min(canvasHeight - 4, Math.max(...members.map((node) => node.y + nodeHeight)) + 12);
    return `<g class="topology-group"><rect x="${minX}" y="${minY}" width="${maxX - minX}" height="${maxY - minY}" rx="12"></rect><text x="${minX + 9}" y="${minY + 14}">${htmlEscape(group.label)}</text></g>`;
  }).join('');

  const pairCounts = new Map();
  const edges = topology.edges.map((edge) => {
    const from = positions.get(edge.from);
    const to = positions.get(edge.to);
    const pair = [edge.from, edge.to].sort().join('|');
    const pairIndex = pairCounts.get(pair) || 0;
    pairCounts.set(pair, pairIndex + 1);
    let x1;
    let y1;
    let x2;
    let y2;
    if (from.row === to.row) {
      const rightward = from.column < to.column;
      const offset = pairIndex % 2 === 0 ? -7 : 7;
      x1 = rightward ? from.x + nodeWidth : from.x;
      x2 = rightward ? to.x : to.x + nodeWidth;
      y1 = from.y + nodeHeight / 2 + offset;
      y2 = to.y + nodeHeight / 2 + offset;
    } else {
      const downward = from.row < to.row;
      x1 = from.x + nodeWidth / 2;
      x2 = to.x + nodeWidth / 2;
      y1 = downward ? from.y + nodeHeight : from.y;
      y2 = downward ? to.y : to.y + nodeHeight;
    }
    const labelX = (x1 + x2) / 2;
    const labelY = (y1 + y2) / 2 - 5;
    return `<g class="topology-edge"><path d="M ${x1} ${y1} L ${x2} ${y2}" marker-end="url(#${markerId})"></path><text x="${labelX}" y="${labelY}">${htmlEscape(edge.label)}</text></g>`;
  }).join('');

  const nodes = topology.nodes.map((node) => {
    const position = positions.get(node.id);
    const lines = wrapTopologyLabel(node.label, Math.max(7, Math.floor(nodeWidth / 13)));
    const firstY = position.y + (nodeHeight / 2) - ((lines.length - 1) * 8) + 4;
    const tspans = lines.map((line, index) => `<tspan x="${position.x + nodeWidth / 2}" y="${firstY + index * 16}">${htmlEscape(line)}</tspan>`).join('');
    return `<g class="topology-node"><rect x="${position.x}" y="${position.y}" width="${nodeWidth}" height="${nodeHeight}" rx="9"></rect><text text-anchor="middle">${tspans}</text></g>`;
  }).join('');

  const accessible = topology.nodes.map((node) => node.label).join('，');
  return `<figure class="term-topology"><figcaption>${htmlEscape(topology.caption)}</figcaption><div class="term-topology-scroll"><svg viewBox="0 0 ${canvasWidth} ${canvasHeight}" role="img" aria-label="${htmlEscape(topology.caption)}。包含：${htmlEscape(accessible)}"><defs><marker id="${markerId}" class="topology-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>${groups}${edges}${nodes}</svg></div></figure>`;
}

function requireNonEmptyString(value, field) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError(`${field} 必须是非空字符串`);
  }
  return value.trim();
}

export function normalizeTermCards(raw, sourceName = 'term cards') {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new TypeError(`${sourceName} 顶层必须是对象`);
  }
  if (!raw.cards || typeof raw.cards !== 'object' || Array.isArray(raw.cards)) {
    throw new TypeError(`${sourceName}.cards 必须是对象`);
  }
  if (raw.meta !== undefined && (!raw.meta || typeof raw.meta !== 'object' || Array.isArray(raw.meta))) {
    throw new TypeError(`${sourceName}.meta 必须是对象`);
  }

  const meta = {};
  if (raw.meta?.title !== undefined) meta.title = requireNonEmptyString(raw.meta.title, `${sourceName}.meta.title`);
  if (raw.meta?.audience !== undefined) meta.audience = requireNonEmptyString(raw.meta.audience, `${sourceName}.meta.audience`);

  const cards = Object.create(null);
  const aliasOwners = new Map();
  for (const [key, card] of Object.entries(raw.cards)) {
    requireNonEmptyString(key, `${sourceName}.cards 的 key`);
    if (!card || typeof card !== 'object' || Array.isArray(card)) {
      throw new TypeError(`${sourceName}.cards.${key} 必须是对象`);
    }
    const label = requireNonEmptyString(card.label, `${sourceName}.cards.${key}.label`);
    if (!Array.isArray(card.aliases) || card.aliases.length !== 1) {
      throw new TypeError(`${sourceName}.cards.${key}.aliases 必须是只含一个正文触发词的数组`);
    }
    if (typeof card.body !== 'string') {
      throw new TypeError(`${sourceName}.cards.${key}.body 必须是 Markdown 字符串`);
    }
    const structure = requireNonEmptyString(card.structure, `${sourceName}.cards.${key}.structure`);
    if (!['atomic', 'composite'].includes(structure)) {
      throw new TypeError(`${sourceName}.cards.${key}.structure 必须是 atomic 或 composite`);
    }
    if (structure === 'composite' && card.topology === undefined) {
      throw new TypeError(`${sourceName}.cards.${key} 是 composite，必须提供 topology`);
    }
    if (structure === 'atomic' && card.topology !== undefined) {
      throw new TypeError(`${sourceName}.cards.${key} 是 atomic，不应提供 topology`);
    }
    const topology = structure === 'composite'
      ? normalizeTopology(card.topology, `${sourceName}.cards.${key}.topology`)
      : null;

    const alias = requireNonEmptyString(card.aliases[0], `${sourceName}.cards.${key}.aliases[0]`);
    const normalizedAlias = alias.toLowerCase();
    const existingOwner = aliasOwners.get(normalizedAlias);
    if (existingOwner && existingOwner !== key) {
      throw new Error(`${sourceName} 中 alias ${alias} 同时属于 ${existingOwner} 与 ${key}`);
    }
    aliasOwners.set(normalizedAlias, key);

    const mentions = [];
    const seenMentions = new Set();
    if (card.mentions !== undefined && !Array.isArray(card.mentions)) {
      throw new TypeError(`${sourceName}.cards.${key}.mentions 必须是数组`);
    }
    for (let index = 0; index < (card.mentions || []).length; index += 1) {
      const mention = requireNonEmptyString(card.mentions[index], `${sourceName}.cards.${key}.mentions[${index}]`);
      const normalizedMention = mention.toLowerCase();
      if (normalizedMention === normalizedAlias || seenMentions.has(normalizedMention)) continue;
      seenMentions.add(normalizedMention);
      mentions.push(mention);
    }

    cards[key] = {
      label,
      aliases: [alias],
      mentions,
      structure,
      topology,
      topologyHtml: topology ? renderTopologySvg(topology, key) : '',
      body: card.body,
      bodyHtml: renderMarkdown(card.body),
    };
  }

  return { meta, cards };
}

export function hasAsciiWordBoundary(text, index, matchLength) {
  const first = text[index] || '';
  const last = text[index + matchLength - 1] || '';
  const previous = index > 0 ? text[index - 1] : '';
  const next = index + matchLength < text.length ? text[index + matchLength] : '';
  const asciiWord = /[A-Za-z0-9_]/;
  return !(
    (asciiWord.test(first) && asciiWord.test(previous))
    || (asciiWord.test(last) && asciiWord.test(next))
  );
}

const STYLES = String.raw`
    :root {
      color-scheme: light;
      --page: #f7f2e8;
      --surface: #fffdf8;
      --surface-raised: #fffefa;
      --surface-soft: #efe9de;
      --sidebar: #f1ece2;
      --ink: #302f2c;
      --ink-soft: #47443f;
      --muted: #5f5b54;
      --faint: #6b665e;
      --rule: #ded7cb;
      --rule-strong: #c9c0b2;
      --accent: #6979c9;
      --accent-strong: #4f5fa9;
      --accent-soft: #e8e9f7;
      --code-bg: #292a31;
      --code-ink: #f2efe8;
      --overlay: rgba(48, 47, 44, .3);
      --shadow: 0 20px 60px rgba(74, 67, 56, .13), 0 3px 12px rgba(74, 67, 56, .06);
      --body-font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      --mono-font: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      --page-gutter: clamp(24px, 2.55vw, 52px);
      --toc-width: clamp(230px, 15vw, 300px);
      --layout-gap: clamp(56px, 5.25vw, 112px);
      --article-max: 1180px;
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --page: #0d0f14;
      --surface: #12151d;
      --surface-raised: #161922;
      --surface-soft: #161922;
      --sidebar: #10131a;
      --ink: #e7e9ef;
      --ink-soft: #c9ccd6;
      --muted: #a1a6b3;
      --faint: #747a8a;
      --rule: #242936;
      --rule-strong: #363d50;
      --accent: #8298ff;
      --accent-strong: #a9b6ff;
      --accent-soft: #20294b;
      --code-bg: #161922;
      --code-ink: #e7e9ef;
      --overlay: rgba(3, 5, 9, .68);
      --shadow: 0 24px 72px rgba(0, 0, 0, .5);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      min-width: 0;
      margin: 0;
      overflow-x: hidden;
      background: var(--page);
      color: var(--ink);
      font-family: var(--body-font);
      font-size: 17px;
      line-height: 1.82;
      text-rendering: optimizeLegibility;
    }
    button, input { font: inherit; }
    button { color: inherit; }
    .reading-progress {
      position: fixed;
      inset: 0 0 auto;
      z-index: 120;
      height: 3px;
      background: transparent;
    }
    .reading-progress span {
      display: block;
      width: 100%;
      height: 100%;
      transform: scaleX(0);
      transform-origin: left;
      background: linear-gradient(90deg, var(--accent), var(--accent-strong));
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 80;
      border-bottom: 1px solid var(--rule);
      background: color-mix(in srgb, var(--surface) 88%, transparent);
      box-shadow: 0 1px 12px rgba(75, 105, 116, .04);
      backdrop-filter: blur(18px) saturate(1.15);
    }
    .topbar-inner {
      display: flex;
      align-items: center;
      gap: 10px;
      width: 100%;
      min-height: 66px;
      margin: 0;
      padding: 10px var(--page-gutter);
    }
    .brand { min-width: 0; margin-right: auto; line-height: 1.35; }
    .brand strong { display: block; overflow: hidden; color: var(--ink); font-size: .86rem; font-weight: 760; text-overflow: ellipsis; white-space: nowrap; }
    .brand span { display: block; overflow: hidden; color: var(--muted); font-size: .7rem; text-overflow: ellipsis; white-space: nowrap; }
    .tool-button, .icon-button {
      min-height: 38px;
      border: 1px solid var(--rule);
      border-radius: 11px;
      background: var(--surface);
      color: var(--muted);
      cursor: pointer;
      font-size: .76rem;
      font-weight: 680;
    }
    .tool-button { padding: 0 13px; }
    .icon-button { width: 38px; padding: 0; font-size: 1.05rem; }
    .tool-button:hover, .icon-button:hover { border-color: color-mix(in srgb, var(--accent) 58%, var(--rule)); background: var(--accent-soft); color: var(--accent-strong); }
    .tool-button:focus-visible, .icon-button:focus-visible, input:focus-visible, .term:focus-visible, .toc-links a:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .toc-button { display: none; }
    .layout {
      display: grid;
      grid-template-columns: var(--toc-width) minmax(0, 1fr);
      gap: var(--layout-gap);
      width: 100%;
      max-width: none;
      margin: 0;
      padding: 52px var(--page-gutter) 110px;
      justify-content: start;
      align-items: start;
    }
    .toc-desktop {
      position: sticky;
      top: 94px;
      max-height: calc(100dvh - 118px);
      overflow-y: auto;
      overscroll-behavior: contain;
      padding: 20px 16px 22px;
      border: 1px solid var(--rule);
      border-radius: 18px;
      background: var(--sidebar);
      scrollbar-width: thin;
    }
    .toc-label {
      margin: 0 0 15px;
      color: var(--accent-strong);
      font-size: .69rem;
      font-weight: 760;
      letter-spacing: .14em;
    }
    .toc-links a {
      display: block;
      margin: 0 0 9px;
      padding: 3px 0 3px 12px;
      border-left: 2px solid var(--rule-strong);
      color: var(--muted);
      font-size: .77rem;
      line-height: 1.45;
      text-decoration: none;
    }
    .toc-links a[data-level="3"] { margin-left: 11px; font-size: .71rem; }
    .toc-links a:hover, .toc-links a[aria-current="location"] { border-left-color: var(--accent); color: var(--accent-strong); }
    .toc-links a[aria-current="location"] { font-weight: 690; }
    .toc-empty { color: var(--muted); font-size: .78rem; }
    main { width: 100%; max-width: var(--article-max); }
    main, article { min-width: 0; }
    article { max-width: 100%; }
    .reader-intro {
      margin: 0 0 44px;
      padding: 17px 19px;
      border: 1px solid var(--rule);
      border-radius: 15px;
      background: var(--sidebar);
      color: var(--muted);
      font-size: .8rem;
      line-height: 1.65;
    }
    .reader-intro strong { color: var(--ink); }
    .article-body { color: var(--ink-soft); overflow-wrap: anywhere; }
    .article-body > h1:first-child { margin-top: 0; }
    .article-body h1, .article-body h2, .article-body h3, .article-body h4 {
      color: var(--ink);
      line-height: 1.28;
      letter-spacing: -.025em;
      scroll-margin-top: 92px;
    }
    .article-body h1 { margin: 0 0 29px; font-size: clamp(2.05rem, 5vw, 3.35rem); font-weight: 770; }
    .article-body h2 { margin: 70px 0 22px; padding-top: 13px; border-top: 1px solid var(--rule); font-size: clamp(1.42rem, 3vw, 2rem); }
    .article-body h3 { margin: 44px 0 17px; font-size: 1.2rem; }
    .article-body h4 { margin: 31px 0 12px; font-size: 1rem; }
    .article-body p { margin: 0 0 1.14em; }
    .article-body a, .card-markdown a { color: var(--accent-strong); text-underline-offset: 3px; }
    .article-body a:hover, .card-markdown a:hover { text-decoration-thickness: 2px; }
    .article-body ul, .article-body ol, .card-markdown ul, .card-markdown ol { margin: 0 0 1.3em; padding-left: 1.4em; }
    .article-body li, .card-markdown li { margin: .4em 0; padding-left: .15em; }
    .article-body blockquote, .card-markdown blockquote {
      margin: 25px 0;
      padding: 17px 20px;
      border: 1px solid var(--rule);
      border-left: 4px solid var(--accent);
      border-radius: 0 11px 11px 0;
      background: color-mix(in srgb, var(--sidebar) 56%, var(--surface));
    }
    .article-body blockquote > :last-child, .card-markdown blockquote > :last-child { margin-bottom: 0; }
    code {
      padding: .13em .37em;
      border: 1px solid var(--rule);
      border-radius: 5px;
      background: var(--surface-soft);
      font-family: var(--mono-font);
      font-size: .86em;
      overflow-wrap: anywhere;
    }
    pre {
      width: 100%;
      max-width: 100%;
      margin: 23px 0;
      padding: 19px;
      overflow-x: auto;
      overscroll-behavior-inline: contain;
      border: 1px solid var(--rule);
      border-radius: 12px;
      background: var(--code-bg);
      color: var(--code-ink);
      line-height: 1.62;
      white-space: pre;
    }
    pre code { padding: 0; border: 0; background: transparent; color: inherit; overflow-wrap: normal; }
    .table-scroll {
      width: 100%;
      max-width: 100%;
      margin: 25px 0;
      overflow-x: auto;
      overscroll-behavior-inline: contain;
      border: 1px solid var(--rule);
      border-radius: 11px;
      background: var(--surface);
    }
    .table-scroll table { width: max-content; min-width: 100%; margin: 0; border-collapse: collapse; font-size: .82rem; line-height: 1.55; }
    th, td { padding: 11px 13px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }
    th { background: var(--surface-soft); font-weight: 730; }
    tr:last-child td { border-bottom: 0; }
    .article-body img, .card-markdown img { display: block; max-width: 100%; height: auto; }
    .article-body details { margin: 23px 0; border: 1px solid var(--rule); border-radius: 11px; background: var(--surface); }
    .article-body summary { padding: 13px 16px; cursor: pointer; font-weight: 690; }
    .article-body details > :not(summary) { margin-right: 17px; margin-left: 17px; }
    .term {
      padding: 0;
      border: 0;
      border-bottom: 1px solid color-mix(in srgb, var(--accent) 24%, transparent);
      background: transparent;
      color: inherit;
      font: inherit;
      cursor: help;
      text-decoration: none;
    }
    code > .term { padding: 0; font-family: inherit; font-size: inherit; }
    .term:hover, .term:focus-visible, .term[data-open="true"] {
      border-bottom-color: var(--accent-strong);
      background: transparent;
      color: var(--accent-strong);
    }
    .sheet-scrim { display: none; }
    .term-panel {
      position: fixed;
      z-index: 111;
      display: none;
      flex-direction: column;
      width: min(520px, calc(100vw - 28px));
      max-height: min(72dvh, 680px);
      overflow: hidden;
      border: 1px solid var(--rule-strong);
      border-radius: 18px;
      background: var(--surface-raised);
      box-shadow: var(--shadow);
    }
    .term-panel[data-visible="true"] { display: flex; }
    .term-panel-head {
      display: flex;
      flex: 0 0 auto;
      align-items: flex-start;
      gap: 12px;
      padding: 16px 17px 13px;
      border-bottom: 1px solid var(--rule);
    }
    .term-panel-title { min-width: 0; margin: 0; font-size: 1rem; line-height: 1.4; overflow-wrap: anywhere; }
    .term-panel-alias { margin: 3px 0 0; color: var(--faint); font-family: var(--mono-font); font-size: .68rem; line-height: 1.4; }
    .term-panel-close { flex: 0 0 auto; margin-left: auto; }
    .term-panel-body { min-height: 0; padding: 16px 18px 19px; overflow-y: auto; overscroll-behavior: contain; }
    .card-markdown { color: var(--ink-soft); font-size: .84rem; line-height: 1.7; overflow-wrap: anywhere; }
    .card-markdown > :first-child { margin-top: 0; }
    .card-markdown > :last-child { margin-bottom: 0; }
    .card-markdown h1, .card-markdown h2, .card-markdown h3, .card-markdown h4 { margin: 1.25em 0 .55em; color: var(--ink); font-size: .96rem; line-height: 1.4; }
    .card-markdown p { margin: 0 0 .92em; }
    .term-topology { margin: 0 0 19px; }
    .term-topology figcaption { margin: 0 0 9px; color: var(--ink); font-size: .76rem; font-weight: 680; }
    .term-topology-scroll { max-width: 100%; overflow-x: auto; overscroll-behavior-inline: contain; }
    .term-topology svg { display: block; width: 100%; min-width: 420px; height: auto; }
    .topology-group rect { fill: color-mix(in srgb, var(--sidebar) 52%, transparent); stroke: var(--rule-strong); stroke-width: 1; }
    .topology-group text { fill: var(--muted); font-size: 10px; font-weight: 680; }
    .topology-node rect { fill: var(--surface-soft); stroke: var(--rule-strong); stroke-width: 1; }
    .topology-node text { fill: var(--ink); font-size: 11px; font-weight: 620; }
    .topology-edge path { fill: none; stroke: var(--accent); stroke-width: 1.35; }
    .topology-arrow path { fill: var(--accent); stroke: none; }
    .topology-edge text { fill: var(--accent-strong); font-size: 9px; text-anchor: middle; paint-order: stroke; stroke: var(--surface-raised); stroke-width: 4px; stroke-linejoin: round; }
    dialog {
      width: min(780px, calc(100vw - 28px));
      max-width: 100%;
      max-height: min(82dvh, 820px);
      padding: 0;
      overflow: hidden;
      border: 1px solid var(--rule-strong);
      border-radius: 16px;
      background: var(--surface-raised);
      color: var(--ink);
      box-shadow: var(--shadow);
    }
    dialog::backdrop { background: var(--overlay); backdrop-filter: blur(4px); }
    .dialog-shell { display: flex; flex-direction: column; max-height: min(82dvh, 820px); }
    .dialog-head {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      flex: 0 0 auto;
      align-items: center;
      gap: 11px;
      padding: 16px 17px;
      border-bottom: 1px solid var(--rule);
      background: var(--surface-raised);
    }
    .dialog-head h2 { margin: 0; font-size: 1.06rem; white-space: nowrap; }
    .dialog-head input {
      min-width: 100px;
      height: 39px;
      flex: 1;
      padding: 0 12px;
      border: 1px solid var(--rule);
      border-radius: 8px;
      background: var(--page);
      color: var(--ink);
      font-size: .8rem;
    }
    .dialog-body { min-height: 0; padding: 8px 18px 22px; overflow-y: auto; overscroll-behavior: contain; }
    .search-summary { margin: 10px 0 0; color: var(--faint); font-size: .72rem; }
    .glossary-card { padding: 18px 0; border-bottom: 1px solid var(--rule); }
    .glossary-card:last-child { border-bottom: 0; }
    .glossary-card h3 { margin: 0 0 4px; font-size: .96rem; }
    .glossary-aliases { margin: 0 0 11px; color: var(--faint); font-family: var(--mono-font); font-size: .7rem; overflow-wrap: anywhere; }
    .glossary-empty { padding: 32px 0; color: var(--muted); text-align: center; }
    .toc-dialog .dialog-body { padding-top: 18px; }
    .toc-dialog .toc-links a { font-size: .85rem; }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
    @media (min-width: 1800px) {
      :root { --article-max: 1260px; }
    }
    @media (max-width: 1020px) {
      .toc-desktop { display: none; }
      .toc-button { display: inline-block; }
      .layout { display: block; width: min(920px, 100%); margin: 0 auto; padding-top: 40px; }
    }
    @media (max-width: 700px) {
      body { font-size: 16px; }
      .topbar-inner { flex-wrap: wrap; padding: 9px 13px 10px; }
      .brand { order: -1; flex: 1 0 100%; width: 100%; }
      .brand strong { font-size: .8rem; }
      .brand span { font-size: .67rem; }
      .tool-button { flex: 1; padding: 0 9px; font-size: .71rem; }
      .icon-button { flex: 0 0 38px; }
      .layout { padding: 32px 16px 82px; }
      .reader-intro { margin-bottom: 34px; }
      .article-body h1 { font-size: 2.08rem; }
      .article-body h2 { margin-top: 57px; }
      pre { padding: 16px; }
      .sheet-scrim {
        position: fixed;
        inset: 0;
        z-index: 109;
        background: var(--overlay);
        backdrop-filter: blur(2px);
      }
      .sheet-scrim[data-visible="true"] { display: block; }
      .term-panel {
        inset: auto 0 0 !important;
        width: 100% !important;
        max-height: 78dvh;
        border-right: 0;
        border-bottom: 0;
        border-left: 0;
        border-radius: 18px 18px 0 0;
      }
      .term-panel-head { padding: 15px 16px 12px; }
      .term-panel-body { padding: 15px 16px max(22px, env(safe-area-inset-bottom)); }
      dialog { width: calc(100vw - 20px); }
      .dialog-head { flex-wrap: wrap; padding: 13px; }
      .dialog-head h2 { margin-right: auto; }
      .dialog-head input { order: 3; flex-basis: 100%; }
      .dialog-body { padding-right: 14px; padding-left: 14px; }
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
    }
    @media print {
      .topbar, .reading-progress, .toc-desktop, .reader-intro, .term-panel, .sheet-scrim, dialog { display: none !important; }
      body { background: white; color: black; font-size: 11pt; }
      .layout { display: block; width: auto; padding: 0; }
      .term { border: 0; background: transparent; }
      a { color: black !important; }
    }
`;

const CLIENT_SCRIPT = String.raw`
    (() => {
      'use strict';
      const payload = JSON.parse(document.getElementById('term-card-data').textContent);
      const entries = Object.entries(payload.cards);
      const aliasToCard = new Map();
      for (const [key, card] of entries) {
        for (const alias of card.aliases) aliasToCard.set(alias.toLowerCase(), key);
      }
      const aliases = [...aliasToCard.keys()].sort((a, b) => b.length - a.length || a.localeCompare(b));
      const escapedAliases = aliases.map((value) => value.replace(/[.*+?^$(){}|[\]\\]/g, '\\$&'));
      const aliasMatcher = escapedAliases.length ? new RegExp('(' + escapedAliases.join('|') + ')', 'giu') : null;
      const article = document.getElementById('article-body');
      const panel = document.getElementById('term-panel');
      const panelTitle = document.getElementById('term-panel-title');
      const panelAlias = document.getElementById('term-panel-alias');
      const panelBody = document.getElementById('term-panel-body');
      const panelClose = document.getElementById('term-panel-close');
      const scrim = document.getElementById('sheet-scrim');
      let activeTerm = null;
      let pinned = false;
      let hideTimer = 0;

      function hasBoundary(text, index, length) {
        const asciiWord = /[A-Za-z0-9_]/;
        const first = text[index] || '';
        const last = text[index + length - 1] || '';
        const previous = index > 0 ? text[index - 1] : '';
        const next = index + length < text.length ? text[index + length] : '';
        return !((asciiWord.test(first) && asciiWord.test(previous)) || (asciiWord.test(last) && asciiWord.test(next)));
      }

      function wrapTables(root) {
        root.querySelectorAll('table').forEach((table) => {
          if (table.parentElement.classList.contains('table-scroll')) return;
          const wrapper = document.createElement('div');
          wrapper.className = 'table-scroll';
          table.parentNode.insertBefore(wrapper, table);
          wrapper.append(table);
        });
      }

      function annotateAliases(root) {
        if (!aliasMatcher) return;
        const skipSelector = 'a, button, pre, script, style, textarea, input, summary, h1, h2, h3, h4, h5, h6, nav, .term, .no-term';
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
          acceptNode(node) {
            if (!node.nodeValue.trim() || node.parentElement.closest(skipSelector)) return NodeFilter.FILTER_REJECT;
            const block = node.parentElement.closest('p, li, td, th');
            if (block && /^(?:原文写法|来源写法)(?:（[^）]*）)?\s*[：:]/.test(block.textContent.trim())) return NodeFilter.FILTER_REJECT;
            aliasMatcher.lastIndex = 0;
            return aliasMatcher.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
          },
        });
        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        for (const node of nodes) {
          const text = node.nodeValue;
          const fragment = document.createDocumentFragment();
          let cursor = 0;
          aliasMatcher.lastIndex = 0;
          for (const match of text.matchAll(aliasMatcher)) {
            const index = match.index;
            if (index > cursor) fragment.append(document.createTextNode(text.slice(cursor, index)));
            if (!hasBoundary(text, index, match[0].length)) {
              fragment.append(document.createTextNode(match[0]));
              cursor = index + match[0].length;
              continue;
            }
            const key = aliasToCard.get(match[0].toLowerCase());
            if (!key) {
              fragment.append(document.createTextNode(match[0]));
              cursor = index + match[0].length;
              continue;
            }
            // A span inside inline code preserves the original <code> container.
            // Fenced code is excluded by the closest(pre) check above.
            const term = document.createElement('span');
            term.className = 'term';
            term.tabIndex = 0;
            term.setAttribute('role', 'button');
            term.setAttribute('aria-haspopup', 'dialog');
            term.setAttribute('aria-controls', 'term-panel');
            term.setAttribute('aria-expanded', 'false');
            term.setAttribute('aria-label', match[0] + '，打开术语解释');
            term.dataset.cardKey = key;
            term.dataset.open = 'false';
            term.textContent = match[0];
            fragment.append(term);
            cursor = index + match[0].length;
          }
          if (cursor < text.length) fragment.append(document.createTextNode(text.slice(cursor)));
          node.replaceWith(fragment);
        }
      }

      function clearActiveTerm() {
        if (!activeTerm) return;
        activeTerm.dataset.open = 'false';
        activeTerm.setAttribute('aria-expanded', 'false');
      }

      function positionPanel(term) {
        if (matchMedia('(max-width: 700px)').matches) {
          panel.style.removeProperty('left');
          panel.style.removeProperty('top');
          panel.style.removeProperty('width');
          return;
        }
        const rect = term.getBoundingClientRect();
        const width = Math.min(520, innerWidth - 28);
        panel.style.width = width + 'px';
        panel.style.left = Math.max(14, Math.min(rect.left + rect.width / 2 - width / 2, innerWidth - width - 14)) + 'px';
        panel.style.top = '14px';
        const panelHeight = panel.offsetHeight;
        const below = rect.bottom + 12;
        const above = rect.top - panelHeight - 12;
        panel.style.top = (below + panelHeight <= innerHeight - 12 ? below : Math.max(12, above)) + 'px';
      }

      function fillPanel(term) {
        const card = payload.cards[term.dataset.cardKey];
        if (!card) return false;
        panelTitle.textContent = card.label;
        panelAlias.textContent = '当前标注：' + term.textContent;
        panelBody.innerHTML = card.topologyHtml + card.bodyHtml;
        wrapTables(panelBody);
        return true;
      }

      function openPanel(term, shouldPin) {
        window.clearTimeout(hideTimer);
        if (!shouldPin && pinned && activeTerm !== term) return;
        if (!fillPanel(term)) return;
        if (activeTerm !== term) clearActiveTerm();
        activeTerm = term;
        pinned = shouldPin || pinned;
        term.dataset.open = 'true';
        term.setAttribute('aria-expanded', 'true');
        panel.dataset.visible = 'true';
        scrim.dataset.visible = 'true';
        positionPanel(term);
      }

      function closePanel({ force = false, restoreFocus = false } = {}) {
        if (pinned && !force) return;
        window.clearTimeout(hideTimer);
        const previousTerm = activeTerm;
        clearActiveTerm();
        activeTerm = null;
        pinned = false;
        panel.dataset.visible = 'false';
        scrim.dataset.visible = 'false';
        if (restoreFocus && previousTerm) previousTerm.focus({ preventScroll: true });
      }

      function scheduleClose() {
        window.clearTimeout(hideTimer);
        hideTimer = window.setTimeout(() => closePanel(), 150);
      }

      wrapTables(article);
      annotateAliases(article);

      document.addEventListener('pointerover', (event) => {
        // On the mobile bottom-sheet layout, a touch pointer can emit
        // pointerover before click. Opening the sheet at that point puts the
        // scrim above the term and steals the click that should pin the card.
        // Keep hover previews on the desktop layout; mobile opens on click or
        // keyboard focus instead.
        if (matchMedia('(max-width: 700px)').matches) return;
        const term = event.target.closest('.term');
        if (term) openPanel(term, false);
      });
      document.addEventListener('pointerout', (event) => {
        if (matchMedia('(max-width: 700px)').matches) return;
        const term = event.target.closest('.term');
        if (term && !event.relatedTarget?.closest('.term-panel')) scheduleClose();
      });
      panel.addEventListener('pointerenter', () => window.clearTimeout(hideTimer));
      panel.addEventListener('pointerleave', scheduleClose);
      document.addEventListener('focusin', (event) => {
        const term = event.target.closest('.term');
        // A pointer press focuses a tabindex term before its click fires.
        // Opening the mobile scrim during focusin would intercept that click.
        // Desktop keeps focus previews; mobile keyboard users open with
        // Enter/Space through the keydown handler below.
        if (term && !matchMedia('(max-width: 700px)').matches) openPanel(term, false);
        if (event.target.closest('.term-panel')) window.clearTimeout(hideTimer);
      });
      document.addEventListener('focusout', (event) => {
        if (event.target.closest('.term') || event.target.closest('.term-panel')) scheduleClose();
      });
      document.addEventListener('click', (event) => {
        const term = event.target.closest('.term');
        if (term) {
          event.preventDefault();
          openPanel(term, true);
          return;
        }
        if (!event.target.closest('.term-panel')) closePanel({ force: true });
      });
      document.addEventListener('keydown', (event) => {
        const term = event.target.closest?.('.term');
        if (term && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          openPanel(term, true);
        }
        if (event.key === 'Escape' && panel.dataset.visible === 'true') {
          event.preventDefault();
          closePanel({ force: true, restoreFocus: true });
        }
      });
      panelClose.addEventListener('click', () => closePanel({ force: true, restoreFocus: true }));
      addEventListener('resize', () => { if (activeTerm) positionPanel(activeTerm); });
      addEventListener('scroll', () => {
        if (activeTerm && pinned) positionPanel(activeTerm);
        else closePanel({ force: true });
        updateProgress();
      }, { passive: true });

      const headings = [...article.querySelectorAll('h2, h3')];
      headings.forEach((heading, index) => { if (!heading.id) heading.id = 'section-' + (index + 1); });
      const tocContainers = [...document.querySelectorAll('[data-toc-links]')];
      function buildToc(container) {
        container.replaceChildren();
        if (!headings.length) {
          const empty = document.createElement('p');
          empty.className = 'toc-empty';
          empty.textContent = '正文没有分节标题';
          container.append(empty);
          return;
        }
        for (const heading of headings) {
          const link = document.createElement('a');
          link.href = '#' + heading.id;
          link.dataset.level = heading.tagName.slice(1);
          link.dataset.headingId = heading.id;
          link.textContent = heading.textContent;
          container.append(link);
        }
      }
      tocContainers.forEach(buildToc);

      const tocDialog = document.getElementById('toc-dialog');
      const openToc = document.getElementById('open-toc');
      const closeToc = document.getElementById('close-toc');
      openToc.addEventListener('click', () => tocDialog.showModal());
      closeToc.addEventListener('click', () => tocDialog.close());
      tocDialog.addEventListener('click', (event) => { if (event.target === tocDialog) tocDialog.close(); });
      tocDialog.addEventListener('click', (event) => { if (event.target.closest('a')) tocDialog.close(); });

      if ('IntersectionObserver' in window && headings.length) {
        const visibleHeadings = new Map();
        const observer = new IntersectionObserver((observations) => {
          for (const observation of observations) {
            if (observation.isIntersecting) visibleHeadings.set(observation.target.id, observation.boundingClientRect.top);
            else visibleHeadings.delete(observation.target.id);
          }
          const activeId = [...visibleHeadings.entries()].sort((a, b) => a[1] - b[1])[0]?.[0];
          if (!activeId) return;
          document.querySelectorAll('[data-heading-id]').forEach((link) => {
            if (link.dataset.headingId === activeId) link.setAttribute('aria-current', 'location');
            else link.removeAttribute('aria-current');
          });
        }, { rootMargin: '-82px 0px -72% 0px', threshold: 0 });
        headings.forEach((heading) => observer.observe(heading));
      }

      const glossary = document.getElementById('glossary-dialog');
      const glossaryList = document.getElementById('glossary-list');
      const glossarySearch = document.getElementById('glossary-search');
      const glossarySummary = document.getElementById('glossary-summary');
      function renderGlossary(query = '') {
        const needle = query.trim().toLowerCase();
        const visible = entries.filter(([key, card]) => {
          if (!needle) return true;
          const topologyText = card.topology
            ? [card.topology.caption, ...card.topology.nodes.map((node) => node.label), ...card.topology.edges.map((edge) => edge.label)].join('\n')
            : '';
          return [key, card.label, card.body, topologyText, ...card.aliases, ...(card.mentions || [])].join('\n').toLowerCase().includes(needle);
        }).sort((a, b) => a[1].label.localeCompare(b[1].label, 'zh-CN'));
        glossaryList.replaceChildren();
        glossarySummary.textContent = '显示 ' + visible.length + ' / ' + entries.length + ' 张术语卡';
        if (!visible.length) {
          const empty = document.createElement('p');
          empty.className = 'glossary-empty';
          empty.textContent = '没有匹配的术语卡';
          glossaryList.append(empty);
          return;
        }
        for (const [, card] of visible) {
          const section = document.createElement('section');
          section.className = 'glossary-card';
          const title = document.createElement('h3');
          title.textContent = card.label;
          const aliasesLine = document.createElement('p');
          aliasesLine.className = 'glossary-aliases';
          aliasesLine.textContent = '正文标注：' + card.aliases[0]
            + (card.mentions?.length ? ' · 相关写法（不生成入口）：' + card.mentions.join(' · ') : '');
          const body = document.createElement('div');
          body.className = 'card-markdown';
          body.innerHTML = card.topologyHtml + card.bodyHtml;
          section.append(title, aliasesLine, body);
          glossaryList.append(section);
          wrapTables(body);
        }
      }
      document.getElementById('open-glossary').addEventListener('click', () => {
        closePanel({ force: true });
        glossarySearch.value = '';
        renderGlossary();
        glossary.showModal();
        requestAnimationFrame(() => glossarySearch.focus());
      });
      document.getElementById('close-glossary').addEventListener('click', () => glossary.close());
      glossary.addEventListener('click', (event) => { if (event.target === glossary) glossary.close(); });
      glossarySearch.addEventListener('input', () => renderGlossary(glossarySearch.value));

      const themeButton = document.getElementById('theme-toggle');
      const themeKey = 'teach-through-breakdown-reader-theme-v2';
      const storedTheme = localStorage.getItem(themeKey);
      function applyTheme(theme) {
        document.documentElement.dataset.theme = theme;
        const dark = theme === 'dark';
        themeButton.textContent = dark ? '浅色' : '深色';
        themeButton.setAttribute('aria-label', dark ? '切换到浅色主题' : '切换到深色主题');
      }
      applyTheme(storedTheme || 'light');
      themeButton.addEventListener('click', () => {
        const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        localStorage.setItem(themeKey, next);
      });

      const progress = document.getElementById('reading-progress');
      const progressBar = document.getElementById('reading-progress-bar');
      function updateProgress() {
        const maximum = document.documentElement.scrollHeight - innerHeight;
        const ratio = maximum <= 0 ? 0 : Math.max(0, Math.min(1, scrollY / maximum));
        progressBar.style.transform = 'scaleX(' + ratio + ')';
        progress.setAttribute('aria-valuenow', String(Math.round(ratio * 100)));
      }
      updateProgress();
    })();
`;

export function buildReaderHtml({ markdown, termCards, fallbackTitle, fallbackAudience }) {
  if (typeof markdown !== 'string') throw new TypeError('markdown 必须是字符串');
  const normalized = normalizeTermCards(termCards);
  const title = normalized.meta.title || fallbackTitle || '自包含机制解释阅读器';
  const audience = normalized.meta.audience || fallbackAudience || '面向需要理解完整工作流的读者';
  const cardCount = Object.keys(normalized.cards).length;
  const payload = jsonForScript(normalized);

  return `<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${htmlEscape(audience)}。正文术语支持悬浮、聚焦、点击与键盘打开解释卡。">
  <title>${htmlEscape(title)}</title>
  <style>${STYLES}</style>
</head>
<body data-reader-version="9">
  <div id="reading-progress" class="reading-progress" role="progressbar" aria-label="阅读进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><span id="reading-progress-bar"></span></div>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand"><strong>${htmlEscape(title)}</strong><span>${htmlEscape(audience)} · ${cardCount} 张术语卡</span></div>
      <button id="open-toc" class="tool-button toc-button" type="button">目录</button>
      <button id="open-glossary" class="tool-button" type="button">全部术语</button>
      <button id="theme-toggle" class="tool-button" type="button" aria-label="切换主题">深色</button>
    </div>
  </header>
  <div class="layout">
    <nav class="toc-desktop" aria-label="文章目录"><p class="toc-label">目录</p><div class="toc-links" data-toc-links></div></nav>
    <main id="main-content">
      <aside class="reader-intro" aria-label="术语卡使用说明"><strong>阅读提示：</strong>带细下划线的词可以悬浮、Tab 聚焦、点击，或用 Enter 和空格键打开解释。点击会固定卡片；Escape 和页面空白处可以关闭。手机会从页面底部展开。</aside>
      <article id="article-body" class="article-body">${renderMarkdown(markdown)}</article>
    </main>
  </div>

  <div id="sheet-scrim" class="sheet-scrim" data-visible="false" aria-hidden="true"></div>
  <aside id="term-panel" class="term-panel" role="dialog" aria-modal="false" aria-labelledby="term-panel-title" data-visible="false">
    <div class="term-panel-head">
      <div><h2 id="term-panel-title" class="term-panel-title"></h2><p id="term-panel-alias" class="term-panel-alias"></p></div>
      <button id="term-panel-close" class="icon-button term-panel-close" type="button" aria-label="关闭术语解释">×</button>
    </div>
    <div id="term-panel-body" class="term-panel-body card-markdown"></div>
  </aside>

  <dialog id="toc-dialog" class="toc-dialog" aria-labelledby="toc-dialog-title">
    <div class="dialog-shell">
      <div class="dialog-head"><h2 id="toc-dialog-title">文章目录</h2><button id="close-toc" class="icon-button" type="button" aria-label="关闭目录">×</button></div>
      <nav class="dialog-body toc-links" data-toc-links aria-label="移动端文章目录"></nav>
    </div>
  </dialog>

  <dialog id="glossary-dialog" aria-labelledby="glossary-title">
    <div class="dialog-shell">
      <div class="dialog-head">
        <h2 id="glossary-title">全部术语</h2>
        <label class="sr-only" for="glossary-search">搜索全部术语卡</label>
        <input id="glossary-search" type="search" autocomplete="off" placeholder="搜索术语、别名或解释正文">
        <button id="close-glossary" class="icon-button" type="button" aria-label="关闭术语表">×</button>
      </div>
      <div class="dialog-body"><p id="glossary-summary" class="search-summary" aria-live="polite"></p><div id="glossary-list"></div></div>
    </div>
  </dialog>

  <script id="term-card-data" type="application/json">${payload}</script>
  <script>${CLIENT_SCRIPT}</script>
</body>
</html>`;
}

async function loadDocument(documentSpec) {
  const [markdown, termsText] = await Promise.all([
    fs.readFile(documentSpec.markdownPath, 'utf8'),
    fs.readFile(documentSpec.termsPath, 'utf8'),
  ]);
  let termCards;
  try {
    termCards = JSON.parse(termsText);
  } catch (error) {
    throw new SyntaxError(`${documentSpec.termsPath} 不是有效 JSON：${error.message}`);
  }
  const html = buildReaderHtml({
    markdown,
    termCards,
    fallbackTitle: documentSpec.fallbackTitle,
    fallbackAudience: documentSpec.fallbackAudience,
  });
  return { ...documentSpec, html, markdownLength: markdown.length, cardCount: Object.keys(termCards.cards || {}).length };
}

export async function buildReaderFromFiles(documentSpec) {
  const rendered = await loadDocument(documentSpec);
  await fs.mkdir(path.dirname(rendered.outputPath), { recursive: true });
  await fs.writeFile(rendered.outputPath, rendered.html, 'utf8');
  return {
    outputPath: rendered.outputPath,
    markdownLength: rendered.markdownLength,
    cardCount: rendered.cardCount,
  };
}

function parseCli(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) throw new Error(`无法识别参数：${token}`);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith('--')) throw new Error(`${token} 缺少取值`);
    values.set(token, value);
    index += 1;
  }
  for (const required of ['--markdown', '--cards', '--output']) {
    if (!values.has(required)) throw new Error(`缺少必需参数 ${required}`);
  }
  return {
    markdownPath: path.resolve(values.get('--markdown')),
    termsPath: path.resolve(values.get('--cards')),
    outputPath: path.resolve(values.get('--output')),
    fallbackTitle: values.get('--title') || '自包含机制解释',
    fallbackAudience: values.get('--audience') || '面向需要真正理解该主题的读者',
  };
}

function runSelfTest() {
  const fixtureCards = {
    meta: { title: '测试阅读器', audience: '面向生成逻辑验证' },
    cards: {
      dinov3: {
        label: 'DINOv3',
        aliases: ['DINOv3'],
        mentions: ['dinov3'],
        structure: 'composite',
        topology: {
          caption: '测试拓扑',
          nodes: [
            { id: 'input', label: '外部输入', row: 0, column: 0 },
            { id: 'state', label: '当前状态', row: 1, column: 0 },
            { id: 'rule', label: '推进规则', row: 1, column: 1 },
          ],
          edges: [
            { from: 'input', to: 'state', label: '写入' },
            { from: 'state', to: 'rule', label: '读取' },
          ],
          groups: [{ label: '测试边界', members: ['state', 'rule'] }],
        },
        body: 'DINOv3 是本测试中的唯一解释对象。\n\n- 支持列表\n- 支持任意段落\n\n| 字段 | 含义 |\n|---|---|\n| `body` | 可变长度 Markdown |',
      },
      threshold: {
        label: '阈值',
        aliases: ['threshold'],
        mentions: ['阈值'],
        structure: 'atomic',
        body: '分数线用于把连续分数转换成类别。\n\n```text\nscore >= threshold\n```',
      },
    },
  };
  const internalFixture = '# 内部 fixture\n\n## 机制\n\nDINOv3 与 DINOv30。行内代码 `no_grad`。\n\n```text\nDINOv3 fenced code\n```';
  const externalFixture = '# 外部 fixture\n\n## 解释\n\n阈值和 threshold。外部唯一标记。';
  const normalized = normalizeTermCards(fixtureCards, 'self-test');
  assert.match(normalized.cards.dinov3.bodyHtml, /<table>/);
  assert.match(normalized.cards.dinov3.topologyHtml, /class="term-topology"/);
  assert.match(normalized.cards.dinov3.topologyHtml, /测试边界/);
  assert.equal(hasAsciiWordBoundary(' DINOv3 ', 1, 6), true);
  assert.equal(hasAsciiWordBoundary('xDINOv3', 1, 6), false);
  assert.equal(hasAsciiWordBoundary('DINOv30', 0, 6), false);

  const internalHtml = buildReaderHtml({
    markdown: internalFixture,
    termCards: fixtureCards,
    fallbackTitle: '内部回退标题',
    fallbackAudience: '内部回退读者',
  });
  const externalHtml = buildReaderHtml({
    markdown: externalFixture,
    termCards: fixtureCards,
    fallbackTitle: '外部回退标题',
    fallbackAudience: '外部回退读者',
  });
  assert.match(internalHtml, /内部 fixture/);
  assert.doesNotMatch(internalHtml, /外部唯一标记/);
  assert.match(externalHtml, /外部唯一标记/);
  assert.doesNotMatch(externalHtml, /内部 fixture/);
  assert.match(internalHtml, /data-reader-version="9"/);
  assert.match(internalHtml, /--page: #f7f2e8/);
  assert.match(internalHtml, /--sidebar: #f1ece2/);
  assert.match(internalHtml, /--ink-soft: #47443f/);
  assert.match(internalHtml, /--muted: #5f5b54/);
  assert.match(internalHtml, /--faint: #6b665e/);
  assert.match(internalHtml, /--accent: #6979c9/);
  assert.match(internalHtml, /border-bottom: 1px solid color-mix\(in srgb, var\(--accent\) 24%, transparent\)/);
  assert.doesNotMatch(internalHtml, /linear-gradient\(to top, var\(--accent-soft\)/);
  assert.match(internalHtml, /--page: #0d0f14/);
  assert.match(internalHtml, /--accent: #8298ff/);
  assert.match(internalHtml, /带细下划线的词/);
  assert.doesNotMatch(internalHtml, /带浅色虚线的词/);
  assert.match(internalHtml, /if \(matchMedia\('\(max-width: 700px\)'\)\.matches\) return;/);
  assert.match(internalHtml, /if \(term && !matchMedia\('\(max-width: 700px\)'\)\.matches\) openPanel\(term, false\);/);
  assert.match(internalHtml, /applyTheme\(storedTheme \|\| 'light'\)/);
  assert.match(internalHtml, /id="term-card-data"/);
  assert.match(internalHtml, /--toc-width: clamp\(230px, 15vw, 300px\)/);
  assert.match(internalHtml, /--article-max: 1260px/);
  assert.match(internalHtml, /grid-template-columns: var\(--toc-width\) minmax\(0, 1fr\)/);
  assert.doesNotMatch(internalHtml, /grid-template-columns: 220px minmax\(0, 800px\)/);
  assert.doesNotMatch(internalHtml, /width: min\(1120px, 100%\)/);
  assert.doesNotMatch(internalHtml, /<script[^>]+src=/i);
  assert.doesNotMatch(internalHtml, /<link[^>]+stylesheet/i);
  assert.match(internalHtml, /只读取显式 aliases|card\.aliases/);
  assert.match(internalHtml, /h1, h2, h3, h4, h5, h6, nav/);
  assert.match(internalHtml, /原文写法\|来源写法/);
  new vm.Script(CLIENT_SCRIPT, { filename: 'term_card_reader_client.js' });
  assert.throws(() => normalizeTermCards({
    cards: {
      a: { label: 'A', aliases: ['same'], structure: 'atomic', body: 'A' },
      b: { label: 'B', aliases: ['SAME'], structure: 'atomic', body: 'B' },
    },
  }, 'duplicate-test'), /同时属于/);
  assert.throws(() => normalizeTermCards({
    cards: {
      combined: { label: 'A 与 B', aliases: ['A', 'B'], structure: 'atomic', body: '错误地合并两个入口' },
    },
  }, 'multi-alias-test'), /只含一个正文触发词/);
  assert.throws(() => normalizeTermCards({
    cards: {
      missing: { label: '结构型卡', aliases: ['结构型卡'], structure: 'composite', body: '缺少拓扑' },
    },
  }, 'missing-topology-test'), /必须提供 topology/);
  return { documents: 2, cards: Object.keys(normalized.cards).length, status: 'PASS' };
}

const isMain = Boolean(process.argv[1]) && await Promise.all([
  fs.realpath(process.argv[1]),
  fs.realpath(fileURLToPath(import.meta.url)),
]).then(([entry, current]) => entry === current).catch(() => false);
if (isMain) {
  try {
    if (process.argv.includes('--self-test')) {
      console.log(JSON.stringify(runSelfTest()));
    } else {
      const result = await buildReaderFromFiles(parseCli(process.argv.slice(2)));
      console.log(`${result.outputPath} cards=${result.cardCount} markdown_chars=${result.markdownLength}`);
    }
  } catch (error) {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  }
}
