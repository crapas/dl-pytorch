"""articles/의 색인 마크다운에서 데이터를 뽑아 검색 가능한 단일 HTML을 만든다.

원본은 마크다운이고 이 스크립트가 만드는 HTML은 생성물이다.
색인을 고칠 때는 마크다운을 고치고 이 스크립트를 다시 실행한다.

    python nbtools/build_index_html.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_ABC = ROOT / 'articles/api-index/README.md'
API_PKG = ROOT / 'articles/api-index/API별_설명.md'
MODELS = ROOT / 'articles/model-index/README.md'
OUT = ROOT / 'docs/api-index/index.html'

# 절 번호(예: 8-3) -> 예제 노트북 리다이렉트 경로(docs/08-03/)
SECTION_RE = re.compile(r'^(\d{1,2})-(\d)$')


def split_row(line):
    """마크다운 표의 한 행을 셀 리스트로 나눈다."""
    return [c.strip() for c in line.strip().strip('|').split('|')]


def clean(text):
    """표 셀에서 마크다운 강조와 코드 표기를 걷어 낸다."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    return text.replace('`', '').strip()


def parse_sections(cell):
    """'2-3, 3-2' 같은 절 표기를 리스트로 나눈다."""
    return [s.strip() for s in clean(cell).split(',') if s.strip()]


def parse_api_abc(path):
    """ABC순 API 색인: | **A** | `optim.Adam` | 2-3, 3-2 |"""
    rows, letter = [], ''
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.startswith('|') or set(line) <= set('|-: '):
            continue
        cells = split_row(line)
        if len(cells) < 3 or cells[1] == 'API':
            continue
        if clean(cells[0]):
            letter = clean(cells[0])
        rows.append({
            'letter': letter,
            'name': clean(cells[1]),
            'sections': parse_sections(cells[2]),
        })
    return rows


def parse_api_pkg(path):
    """접두사별 API 색인: ## 그룹 제목 아래 | 종류 | 이름 | 절 | 설명 |"""
    rows, group, kind = [], '', ''
    for line in path.read_text(encoding='utf-8').splitlines():
        heading = re.match(r'^##\s+(.+?)\s*$', line)
        if heading:
            group, kind = re.sub(r'\s+', ' ', heading.group(1)).strip(), ''
            continue
        if not line.startswith('|') or set(line) <= set('|-: '):
            continue
        cells = split_row(line)
        if len(cells) < 4 or cells[1] == '이름':
            continue
        if clean(cells[0]):
            kind = clean(cells[0])
        rows.append({
            'group': group,
            'kind': kind,
            'name': clean(cells[1]),
            'sections': parse_sections(cells[2]),
            'desc': clean(cells[3]),
        })
    return rows


def parse_models(path):
    """모델 색인: ## N부 제목 아래 | 모델 | 이름 | 절 | 내용 |"""
    rows, part, num, name = [], '', '', ''
    for line in path.read_text(encoding='utf-8').splitlines():
        heading = re.match(r'^##\s+(.+?)\s*$', line)
        if heading:
            part = heading.group(1).strip()
            continue
        if not line.startswith('|') or set(line) <= set('|-: '):
            continue
        cells = split_row(line)
        if len(cells) < 4 or cells[0] == '모델':
            continue
        if clean(cells[0]):
            num, name = clean(cells[0]), clean(cells[1])
        rows.append({
            'part': part,
            'num': num,
            'name': name,
            'sections': parse_sections(cells[2]),
            'desc': clean(cells[3]),
        })
    return rows


TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>찾아보기 — 직접 구현하는 딥러닝 with 파이토치</title>
<style>
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --line: #e2e2e2;
  --accent: #1f5fa8; --chip: #f1f3f5; --mark: #fff3bf;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14161a; --fg: #e6e6e6; --muted: #9aa0a6; --line: #2b2f36;
    --accent: #79aef5; --chip: #22262d; --mark: #5c4b12;
  }
}
:root[data-theme="dark"] {
  --bg: #14161a; --fg: #e6e6e6; --muted: #9aa0a6; --line: #2b2f36;
  --accent: #79aef5; --chip: #22262d; --mark: #5c4b12;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: system-ui, -apple-system, "Segoe UI", "Noto Sans KR", sans-serif;
  line-height: 1.6;
}
.wrap { max-width: 62rem; margin: 0 auto; padding: 1.5rem 16px 4rem; }
h1 { font-size: 1.35rem; margin: 0 0 .3rem; }
.lead { color: var(--muted); font-size: .9rem; margin: 0 0 1.2rem; }
.lead a { color: var(--accent); }
.controls { position: sticky; top: 0; background: var(--bg);
  padding: .8rem 0; border-bottom: 1px solid var(--line); z-index: 5; }
#q { width: 100%; padding: .7rem .9rem; font-size: 1rem;
  border: 1px solid var(--line); border-radius: 8px;
  background: var(--bg); color: var(--fg); }
#q:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.tabs { display: flex; gap: .4rem; margin-top: .7rem; flex-wrap: wrap; }
.tab { padding: .35rem .8rem; font-size: .88rem; cursor: pointer;
  border: 1px solid var(--line); border-radius: 999px;
  background: var(--bg); color: var(--fg); }
.tab[aria-selected="true"] { background: var(--accent); color: #fff;
  border-color: var(--accent); }
.count { color: var(--muted); font-size: .82rem; margin: .7rem 0 .3rem; }
table { width: 100%; border-collapse: collapse; font-size: .9rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { font-weight: 600; font-size: .82rem; color: var(--muted);
  position: sticky; top: 5.4rem; background: var(--bg); }
tbody tr:hover { background: var(--chip); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em; }
.sec { display: inline-block; margin-right: .3rem; font-size: .82rem;
  padding: .05rem .4rem; border-radius: 4px; background: var(--chip);
  color: var(--accent); text-decoration: none; white-space: nowrap; }
.sec:hover { text-decoration: underline; }
.sec.plain { color: var(--muted); }
.grp { font-weight: 600; font-size: .86rem; padding-top: 1.1rem;
  color: var(--accent); background: var(--bg); }
mark { background: var(--mark); color: inherit; padding: 0 .1em; border-radius: 2px; }
.empty { color: var(--muted); padding: 2rem 0; }
.hint { color: var(--muted); font-size: .8rem; margin-top: .4rem; }
@media (max-width: 40rem) {
  th:nth-child(1), td:nth-child(1) { display: none; }
  th { top: 6.6rem; }
}
</style>
</head>
<body>
<div class="wrap">
<h1>찾아보기</h1>
<p class="lead">『직접 구현하는 딥러닝 with 파이토치』 본문과 예제 코드에 등장하는 API와 예제 모델의 색인입니다.
절 번호를 누르면 해당 절의 예제 노트북으로 이동합니다.
저장소는 <a href="https://github.com/crapas/dl-pytorch">github.com/crapas/dl-pytorch</a>입니다.</p>

<div class="controls">
  <input id="q" type="search" placeholder="API 이름, 설명, 모델 이름, 절 번호로 검색 (예: Conv2d, 어텐션, 8-3)"
         autocomplete="off" autofocus>
  <div class="tabs" role="tablist">
    <button class="tab" role="tab" data-tab="abc" aria-selected="true">API — 이름순</button>
    <button class="tab" role="tab" data-tab="pkg" aria-selected="false">API — 패키지별</button>
    <button class="tab" role="tab" data-tab="model" aria-selected="false">예제 모델</button>
  </div>
</div>

<p class="count" id="count"></p>
<noscript>
  <p class="empty">이 페이지의 검색 기능은 자바스크립트를 사용합니다.
  자바스크립트를 켤 수 없다면 저장소의 마크다운 색인을 이용해 주세요.</p>
  <ul>
    <li><a href="https://github.com/crapas/dl-pytorch/blob/main/articles/api-index/README.md">파이토치 API 찾아보기 — 이름순</a></li>
    <li><a href="https://github.com/crapas/dl-pytorch/blob/main/articles/api-index/API%EB%B3%84_%EC%84%A4%EB%AA%85.md">파이토치 API 찾아보기 — 패키지별</a></li>
    <li><a href="https://github.com/crapas/dl-pytorch/blob/main/articles/model-index/README.md">예제 모델 찾아보기</a></li>
  </ul>
</noscript>
<div id="view"></div>
<p class="hint">이 페이지는 저장소의 마크다운 색인에서 생성합니다.
원본은 <a href="https://github.com/crapas/dl-pytorch/tree/main/articles/api-index">articles/api-index</a>와
<a href="https://github.com/crapas/dl-pytorch/tree/main/articles/model-index">articles/model-index</a>에 있습니다.</p>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const view = document.getElementById('view');
const countEl = document.getElementById('count');
const q = document.getElementById('q');
let tab = 'abc';

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// 절 번호를 예제 노트북 리다이렉트 링크로 바꾼다(8-3 -> ../08-03/)
function secLinks(list) {
  return list.map(s => {
    const m = /^(\\d{1,2})-(\\d)$/.exec(s);
    if (!m) return `<span class="sec plain">${esc(s)}</span>`;
    const path = String(m[1]).padStart(2, '0') + '-0' + m[2];
    return `<a class="sec" href="../${path}/" title="${esc(s)}절 예제 노트북">${esc(s)}</a>`;
  }).join('');
}

function hit(row, needle) {
  if (!needle) return true;
  return row._hay.includes(needle);
}

function mark(text, needle) {
  const safe = esc(text);
  if (!needle) return safe;
  const i = safe.toLowerCase().indexOf(needle);
  if (i < 0) return safe;
  return safe.slice(0, i) + '<mark>' + safe.slice(i, i + needle.length)
       + '</mark>' + safe.slice(i + needle.length);
}

function render() {
  const needle = q.value.trim().toLowerCase();
  const rows = DATA[tab].filter(r => hit(r, needle));
  countEl.textContent = needle
    ? `${rows.length}건 (전체 ${DATA[tab].length}건 중)`
    : `전체 ${rows.length}건`;
  if (!rows.length) { view.innerHTML = '<p class="empty">일치하는 항목이 없습니다.</p>'; return; }

  let html = '', group = null;
  if (tab === 'abc') {
    html = '<table><thead><tr><th style="width:3rem"></th><th>API</th>'
         + '<th style="width:12rem">절</th></tr></thead><tbody>';
    for (const r of rows) {
      const letter = r.letter !== group ? esc(r.letter) : '';
      group = r.letter;
      html += `<tr><td>${letter ? '<b>' + letter + '</b>' : ''}</td>`
           + `<td><code>${mark(r.name, needle)}</code></td>`
           + `<td>${secLinks(r.sections)}</td></tr>`;
    }
  } else if (tab === 'pkg') {
    html = '<table><thead><tr><th style="width:4rem">종류</th><th style="width:14rem">이름</th>'
         + '<th style="width:8rem">절</th><th>설명</th></tr></thead><tbody>';
    for (const r of rows) {
      if (r.group !== group) {
        group = r.group;
        html += `<tr><td class="grp" colspan="4">${esc(r.group)}</td></tr>`;
      }
      html += `<tr><td>${esc(r.kind)}</td>`
           + `<td><code>${mark(r.name, needle)}</code></td>`
           + `<td>${secLinks(r.sections)}</td>`
           + `<td>${mark(r.desc, needle)}</td></tr>`;
    }
  } else {
    html = '<table><thead><tr><th style="width:3rem">모델</th><th style="width:14rem">이름</th>'
         + '<th style="width:6rem">절</th><th>그 절에서 다루는 내용</th></tr></thead><tbody>';
    for (const r of rows) {
      if (r.part !== group) {
        group = r.part;
        html += `<tr><td class="grp" colspan="4">${esc(r.part)}</td></tr>`;
      }
      html += `<tr><td><b>${esc(r.num)}</b></td>`
           + `<td>${mark(r.name, needle)}</td>`
           + `<td>${secLinks(r.sections)}</td>`
           + `<td>${mark(r.desc, needle)}</td></tr>`;
    }
  }
  view.innerHTML = html + '</tbody></table>';
}

for (const b of document.querySelectorAll('.tab')) {
  b.addEventListener('click', () => {
    tab = b.dataset.tab;
    for (const o of document.querySelectorAll('.tab'))
      o.setAttribute('aria-selected', String(o === b));
    render();
  });
}
q.addEventListener('input', render);
render();
</script>
</body>
</html>
"""


def main():
    abc = parse_api_abc(API_ABC)
    pkg = parse_api_pkg(API_PKG)
    models = parse_models(MODELS)

    # 검색용 문자열을 미리 만들어 둔다(소문자 정규화)
    for r in abc:
        r['_hay'] = ' '.join([r['name']] + r['sections']).lower()
    for r in pkg:
        r['_hay'] = ' '.join([r['group'], r['kind'], r['name'], r['desc']]
                             + r['sections']).lower()
    for r in models:
        r['_hay'] = ' '.join([r['part'], r['num'], r['name'], r['desc']]
                             + r['sections']).lower()

    data = {'abc': abc, 'pkg': pkg, 'model': models}
    html = TEMPLATE.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding='utf-8')
    print(f'{OUT.relative_to(ROOT)} 생성')
    print(f'  API 이름순 {len(abc)}건 / 패키지별 {len(pkg)}건 / 모델 {len(models)}건')
    print(f'  크기 {OUT.stat().st_size / 1024:.1f} KB')


if __name__ == '__main__':
    main()
