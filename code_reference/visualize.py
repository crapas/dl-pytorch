"""딥러닝 입문서 시각화 라이브러리 (visualize.py).

본 모듈은 책 전체에서 공통으로 사용하는 시각화 함수를 모은다.
3단계 재설계 결과로 다음 정책을 따른다.

1. 화면 표시는 컬러로 한다.
2. 모듈 변수 ``SAVE_GRAYSCALE = True`` 일 때, 화면에 표시된 그림과 동일한
   그림의 회색조 버전을 300DPI로 자동 저장한다. 호출 측은 저장 사실을 신경 쓰지
   않는다. 파일명은 ``NOTEBOOK_NAME`` + ``_<일련번호>.png`` 형식.
3. 다중 시리즈는 색·선스타일·마커를 동시에 변동시켜, 회색조 변환 후에도 시리즈가
   구분되도록 설계한다.
4. 산포도 함수에는 ``sample_ratio`` 파라미터가 있어 점 일부만 무작위 표시할 수 있다.

공개 함수
    설정: configure, reset_counter
    학습 곡선: plot_history, plot_histories
    분포/특징: plot_scatter, plot_decision_boundary
    이미지·격자: plot_images, plot_feature_map
    분류 평가: plot_confusion_matrix
    어텐션: plot_attention
"""

from io import BytesIO
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from .common import set_korean_plot_env


# ===========================================================================
# 모듈 설정 — 회색조 자동 저장 정책
# ===========================================================================

SAVE_GRAYSCALE = False
"""True 일 때 모든 시각화 함수가 화면 표시와 동시에 회색조 PNG를 저장한다."""

NOTEBOOK_NAME = None
"""저장 파일명의 접두어. 노트북 셀 첫머리에서 명시적으로 설정 권장.

예시: ``visualize.NOTEBOOK_NAME = '02-02_example'``
``None`` 이면 ``'figure'`` 로 대체된다.
"""

OUTPUT_DIR = None
"""회색조 PNG를 저장할 디렉터리. ``None`` 이면 현재 작업 디렉터리(노트북 위치)."""

_save_counter = 0


def configure(save_grayscale=None, notebook_name=None, output_dir=None):
    """모듈 설정을 한 번에 변경한다.

    인자
        save_grayscale: 회색조 저장 활성화 여부 (bool). ``None``이면 변경하지 않음.
        notebook_name: 저장 파일명 접두어. ``None``이면 변경하지 않음.
        output_dir: 저장 디렉터리. ``None``이면 변경하지 않음.

    동일 인자를 ``None`` 으로 명시 해제하려면 모듈 변수에 직접 ``None`` 을 할당하면 된다.
    """
    global SAVE_GRAYSCALE, NOTEBOOK_NAME, OUTPUT_DIR
    if save_grayscale is not None:
        SAVE_GRAYSCALE = bool(save_grayscale)
    if notebook_name is not None:
        NOTEBOOK_NAME = str(notebook_name)
    if output_dir is not None:
        OUTPUT_DIR = str(output_dir)


def reset_counter():
    """저장 파일명의 일련번호를 1부터 다시 매긴다.

    노트북 시작 셀에서 한 번 호출해 두면, 셀 실행 순서가 바뀌어도 파일 인덱스가
    예측 가능해진다.
    """
    global _save_counter
    _save_counter = 0


# ===========================================================================
# 내부 헬퍼
# ===========================================================================

# 회색조 변환 후에도 구분되도록 색 · 선스타일 · 마커를 동시에 변동시킨다.
_COLOR_CYCLE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
_LINESTYLE_CYCLE = ['-', '--', '-.', ':']
_MARKER_CYCLE = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']


def _series_style(index):
    """시리즈 인덱스 → (color, linestyle, marker) 튜플."""
    return {
        'color': _COLOR_CYCLE[index % len(_COLOR_CYCLE)],
        'linestyle': _LINESTYLE_CYCLE[index % len(_LINESTYLE_CYCLE)],
        'marker': _MARKER_CYCLE[index % len(_MARKER_CYCLE)],
    }


def _next_save_path():
    """다음 회색조 저장 파일의 절대 경로를 반환하고 카운터를 1 증가시킨다."""
    global _save_counter
    _save_counter += 1
    base = NOTEBOOK_NAME or 'figure'
    filename = f'{base}_{_save_counter:02d}.png'
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return os.path.join(OUTPUT_DIR, filename)
    return filename


def _save_grayscale(fig):
    """fig 을 회색조로 변환해 300DPI PNG로 저장한다."""
    try:
        from PIL import Image
    except ImportError:
        print('[visualize] Pillow가 설치되어 있지 않아 회색조 저장을 건너뜁니다.')
        return

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                facecolor='white')
    buf.seek(0)
    img = Image.open(buf).convert('L')   # 휘도 기반 회색조 변환
    save_path = _next_save_path()
    img.save(save_path, dpi=(300, 300))


def _finalize(fig):
    """모든 시각화 함수의 마지막 단계.

    SAVE_GRAYSCALE이면 회색조 저장을 먼저 수행한 뒤 plt.show()로 화면 표시.
    """
    if SAVE_GRAYSCALE:
        _save_grayscale(fig)
    plt.show()


def _to_numpy_1d(value):
    """스칼라/리스트/배열/텐서를 1차원 numpy 배열로 통일."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value).squeeze()
    if arr.ndim == 0:
        arr = np.expand_dims(arr, axis=0)
    return arr


def _confidence_ellipse_params(points, confidence=0.95):
    """2D 점 집합의 신뢰 타원 파라미터(center, width, height, angle)를 계산한다.

    confidence 는 (0, 1) 범위여야 하며, 자유도 2 카이제곱 분포의 닫힌형식을 사용한다.
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return None
    if not 0 < confidence < 1:
        raise ValueError('confidence 는 0 초과 1 미만이어야 합니다.')

    cov = np.cov(pts, rowvar=False)
    if cov.shape != (2, 2):
        return None

    # 자유도 2 카이제곱 분포의 분위수: q = -2 ln(1 - p)
    q = -2.0 * np.log(1.0 - confidence)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width = 2.0 * np.sqrt(vals[0] * q)
    height = 2.0 * np.sqrt(vals[1] * q)
    if width == 0 and height == 0:
        return None

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    center = pts.mean(axis=0)
    return center, width, height, angle


# ===========================================================================
# 학습 곡선
# ===========================================================================

def plot_history(history, *, title=None, x_label='에포크', y_label='손실',
                 y_lim=None, x_from_0=False, ticker_step=None,
                 ignore_before=1, ignore_after=float('inf'),
                 label=None, figsize=(10, 6), font_scale=1.0):
    """단일 시계열(history)을 선 그래프로 그린다.

    인자
        history: 숫자 리스트(에포크가 1부터 자동 매김) 또는 (x, y) 튜플 리스트.
        title: 그래프 제목.
        x_label, y_label: 축 라벨.
        y_lim: ``(ymin, ymax)`` 튜플.
        x_from_0: True이면 x축이 0부터 시작.
        ticker_step: x 눈금 간격. None(기본)이면 에포크 수와 무관하게
            x 눈금·보조선을 약 5개로 자동 제한한다. 정수를 주면 그 간격으로 고정.
        ignore_before, ignore_after: 표시 구간 필터.
        label: 범례 표시명. None이면 범례 없음.
        figsize: 그림 크기.
        font_scale: 모든 텍스트(제목·축 라벨·눈금·범례)에 적용할 폰트 크기
            배율. 1.0이 기본. 책에 작게 싣는 그림은 1.4~1.8 정도가 적정.
    """
    set_korean_plot_env()

    if isinstance(history[0], (list, tuple)):
        pairs = history
    else:
        pairs = [(i + 1, y) for i, y in enumerate(history)]

    filtered = [(x, y) for x, y in pairs if ignore_before <= x <= ignore_after]
    if len(filtered) < 2:
        raise ValueError('둘 이상의 점을 사용해 플롯을 그려야 합니다.')

    x_list, y_list = zip(*filtered)

    fig, ax = plt.subplots(figsize=figsize)

    style = _series_style(0)
    ax.plot(x_list, y_list, color=style['color'], linestyle=style['linestyle'],
            linewidth=2, label=label)

    x_min = 0 if x_from_0 else min(x_list)
    x_max = max(x_list)
    if ticker_step is None:
        # 에포크 수와 무관하게 x 눈금·보조선을 약 5개로 제한 (정수 눈금)
        if x_from_0:
            ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    else:
        xticks = list(range(x_min, x_max, ticker_step))
        if not xticks or xticks[-1] != x_max:
            xticks.append(x_max)
        ax.set_xticks(xticks)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.tick_params(axis='both', labelsize=10 * font_scale)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10 * font_scale)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10 * font_scale)
    if title:
        ax.set_title(title, fontsize=12 * font_scale)
    if label:
        ax.legend(fontsize=10 * font_scale)
    ax.grid(True, linestyle='--', alpha=0.6)

    _finalize(fig)


def plot_histories(series, *, title=None, x_label='에포크', y_label='손실',
                   y_lim=None, x_lim=None, x_ticks=None, y_ticks=None,
                   x_ticker_step=None, show_legend=True,
                   linewidth=2.0, figsize=(10, 6), font_scale=1.0):
    """다중 시계열을 한 축에 그린다. 회색조 변환 후에도 구분되도록 색·선스타일·마커를 동시에 변동.

    인자
        series: ``{라벨: [y, ...] 또는 [(x, y), ...]}`` 딕셔너리.
                또는 ``[(라벨, history), ...]`` 리스트.
        title, x_label, y_label: 그래프 텍스트.
        y_lim, x_lim: 축 범위.
        x_ticks, y_ticks: 명시적 눈금.
        x_ticker_step: x 눈금 간격(정수).
        show_legend: 범례 표시 여부.
        linewidth: 선 굵기.
        figsize: 그림 크기.
        font_scale: 모든 텍스트(제목·축 라벨·눈금·범례)에 적용할 폰트 크기
            배율. 1.0이 기본. 책에 작게 싣는 그림은 1.4~1.8 정도가 적정.
    """
    set_korean_plot_env()

    if isinstance(series, dict):
        items = list(series.items())
    else:
        items = list(series)

    if not items:
        raise ValueError('series 가 비어 있습니다.')

    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, history) in enumerate(items):
        if isinstance(history[0], (list, tuple)):
            x_list, y_list = zip(*history)
        else:
            x_list = list(range(1, len(history) + 1))
            y_list = list(history)

        style = _series_style(i)
        ax.plot(x_list, y_list, color=style['color'],
                linestyle=style['linestyle'], linewidth=linewidth, label=label)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    elif x_ticker_step is not None and x_lim is not None:
        if int(x_ticker_step) != x_ticker_step:
            raise ValueError('x_ticker_step의 값은 정수여야 합니다.')
        start = math.floor(min(x_lim))
        end = math.ceil(max(x_lim))
        ticks = list(range(start, end, int(x_ticker_step)))
        if ticks[-1] < max(x_lim):
            ticks.append(ticks[-1] + int(x_ticker_step))
        ax.set_xticks(ticks)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    title_size = 12 * font_scale
    label_size = 10 * font_scale
    tick_size = 10 * font_scale
    legend_size = 10 * font_scale

    ax.tick_params(axis='both', labelsize=tick_size)
    if x_label:
        ax.set_xlabel(x_label, fontsize=label_size)
    if y_label:
        ax.set_ylabel(y_label, fontsize=label_size)
    if title:
        ax.set_title(title, fontsize=title_size)
    if show_legend:
        ax.legend(fontsize=legend_size)
    ax.grid(True, linestyle='--', alpha=0.6)

    _finalize(fig)


# ===========================================================================
# 산포도
# ===========================================================================

def plot_scatter(X, y=None, *, sample_ratio=1.0, title=None,
                 x_label=None, y_label=None, x_lim=None, y_lim=None,
                 class_names=None, point_size=30, figsize=(8, 6),
                 font_scale=1.0, add_ellipses=False,
                 ellipse_confidence=0.95, ellipse_linewidth=1.6):
    """2D 산포도를 그린다.

    인자
        X: shape ``(N, 2)`` 의 텐서/배열 또는 ``(x1_list, x2_list)`` 튜플.
        y: shape ``(N,)`` 의 클래스 라벨. ``None`` 이면 단일 색.
        sample_ratio: 0 < r ≤ 1. r < 1 이면 전체 중 r 비율만 무작위 추출.
        title, x_label, y_label: 텍스트.
        x_lim, y_lim: 축 범위.
        class_names: 라벨별 범례명. ``None`` 이면 숫자.
        point_size: 점 크기.
        figsize: 그림 크기.
        font_scale: 모든 텍스트에 적용할 폰트 크기 배율. 1.0이 기본.
        add_ellipses: True이면 클래스별 신뢰 타원을 함께 그린다(y 필요).
        ellipse_confidence: 타원 신뢰수준(기본 0.95).
        ellipse_linewidth: 타원 선 굵기.
    """
    set_korean_plot_env()

    # X 정규화
    if isinstance(X, (list, tuple)) and len(X) == 2 and np.ndim(X[0]) == 1:
        x1 = np.asarray(X[0])
        x2 = np.asarray(X[1])
    else:
        X_arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != 2:
            raise ValueError('X 는 (N, 2) 형태여야 합니다.')
        x1, x2 = X_arr[:, 0], X_arr[:, 1]

    if y is not None:
        y_arr = _to_numpy_1d(y)
        if len(y_arr) != len(x1):
            raise ValueError('X 와 y 의 길이가 다릅니다.')
    else:
        y_arr = None

    if not 0 < sample_ratio <= 1:
        raise ValueError('sample_ratio 는 0 초과 1 이하여야 합니다.')
    if sample_ratio < 1.0:
        n = len(x1)
        k = max(1, int(n * sample_ratio))
        idx = np.random.choice(n, size=k, replace=False)
        x1, x2 = x1[idx], x2[idx]
        if y_arr is not None:
            y_arr = y_arr[idx]

    fig, ax = plt.subplots(figsize=figsize)

    if y_arr is None:
        ax.scatter(x1, x2, s=point_size, alpha=0.7, color=_COLOR_CYCLE[0])
    else:
        unique_labels = sorted(np.unique(y_arr).tolist())
        for i, lbl in enumerate(unique_labels):
            mask = (y_arr == lbl)
            style = _series_style(i)
            label = (class_names[int(lbl)] if class_names is not None
                     else str(lbl))
            ax.scatter(x1[mask], x2[mask], s=point_size, alpha=0.7,
                       color=style['color'], marker=style['marker'],
                       label=label)
            if add_ellipses:
                points = np.column_stack([x1[mask], x2[mask]])
                params = _confidence_ellipse_params(
                    points, confidence=ellipse_confidence,
                )
                if params is not None:
                    center, width, height, angle = params
                    ellipse = Ellipse(
                        xy=center,
                        width=width,
                        height=height,
                        angle=angle,
                        fill=False,
                        edgecolor=style['color'],
                        linewidth=ellipse_linewidth,
                        linestyle=style['linestyle'],
                        alpha=0.95,
                    )
                    ax.add_patch(ellipse)
        ax.legend(fontsize=10 * font_scale)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.tick_params(axis='both', labelsize=10 * font_scale)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10 * font_scale)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10 * font_scale)
    if title:
        ax.set_title(title, fontsize=12 * font_scale)
    ax.grid(True, linestyle='--', alpha=0.4)

    _finalize(fig)


def plot_regression_scatter(X, y, *, reference_fn=None, reference_label=None,
                            sample_ratio=1.0, title=None,
                            x_label=None, y_label=None,
                            x_lim=None, y_lim=None,
                            point_size=30, figsize=(8, 6),
                            font_scale=1.0):
    """1D 회귀 산포도. 기준선(참값/모델 예측)을 함께 그릴 수 있다.

    인자
        X: shape ``(N,)`` 의 1D 텐서/배열.
        y: shape ``(N,)`` 의 1D 텐서/배열.
        reference_fn: x 텐서를 받아 y 값을 반환하는 함수. None이면 산포만.
        reference_label: 기준선 범례명.
        sample_ratio: 0 < r ≤ 1. r < 1 이면 그 비율만 무작위 표시.
        title, x_label, y_label: 텍스트.
        x_lim, y_lim: 축 범위.
        point_size: 점 크기.
        figsize: 그림 크기.
        font_scale: 모든 텍스트에 적용할 폰트 크기 배율. 1.0이 기본.
    """
    set_korean_plot_env()

    x_arr = _to_numpy_1d(X)
    y_arr = _to_numpy_1d(y)
    if len(x_arr) != len(y_arr):
        raise ValueError('X 와 y 의 길이가 다릅니다.')

    if not 0 < sample_ratio <= 1:
        raise ValueError('sample_ratio 는 0 초과 1 이하여야 합니다.')
    if sample_ratio < 1.0:
        n = len(x_arr)
        k = max(1, int(n * sample_ratio))
        idx = np.random.choice(n, size=k, replace=False)
        x_arr, y_arr = x_arr[idx], y_arr[idx]

    fig, ax = plt.subplots(figsize=figsize)

    scatter_style = _series_style(0)
    ax.scatter(x_arr, y_arr, s=point_size, alpha=0.6,
               color=scatter_style['color'],
               marker=scatter_style['marker'],
               label='관측값')

    if reference_fn is not None:
        x_min = x_lim[0] if x_lim is not None else float(x_arr.min())
        x_max = x_lim[1] if x_lim is not None else float(x_arr.max())
        xs = torch.linspace(x_min, x_max, 400)
        ys = reference_fn(xs)
        if isinstance(ys, torch.Tensor):
            ys = ys.detach().cpu().numpy()
        line_style = _series_style(1)
        ax.plot(xs.numpy(), ys, color=line_style['color'],
                linestyle=line_style['linestyle'], linewidth=2,
                label=reference_label or '기준선')

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.tick_params(axis='both', labelsize=10 * font_scale)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10 * font_scale)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10 * font_scale)
    if title:
        ax.set_title(title, fontsize=12 * font_scale)
    if reference_fn is not None or sample_ratio < 1.0:
        ax.legend(fontsize=10 * font_scale)
    ax.grid(True, linestyle='--', alpha=0.4)

    _finalize(fig)


def plot_activation_compare(activations, *, x_range=(-5, 5), num_points=400,
                            title=None, x_label='입력', y_label='출력',
                            x_lim=None, y_lim=None,
                            show_legend=True, linewidth=2.0, figsize=(8, 6),
                            font_scale=1.0):
    """여러 활성화 함수를 한 축에 비교 시각화한다.

    회색조 변환 후에도 구분되도록 색·선스타일을 동시에 변동시킨다.

    인자
        activations: ``{라벨: 함수}`` dict 또는 ``[(라벨, 함수), ...]`` list.
                     함수는 1D 텐서를 받아 1D 텐서/배열을 반환해야 한다.
        x_range: 입력 범위 ``(xmin, xmax)``. 기본 ``(-5, 5)``.
        num_points: x 격자 점 수.
        title, x_label, y_label: 텍스트.
        x_lim, y_lim: 축 범위. None이면 자동.
        show_legend: 범례 표시 여부.
        linewidth: 선 굵기.
        figsize: 그림 크기.
        font_scale: 모든 텍스트에 적용할 폰트 크기 배율. 1.0이 기본.
    """
    set_korean_plot_env()

    if isinstance(activations, dict):
        items = list(activations.items())
    else:
        items = list(activations)
    if not items:
        raise ValueError('activations 가 비어 있습니다.')

    xs = torch.linspace(x_range[0], x_range[1], num_points)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, fn) in enumerate(items):
        ys = fn(xs)
        if isinstance(ys, torch.Tensor):
            ys = ys.detach().cpu().numpy()
        style = _series_style(i)
        ax.plot(xs.numpy(), ys, color=style['color'],
                linestyle=style['linestyle'], linewidth=linewidth, label=label)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.tick_params(axis='both', labelsize=10 * font_scale)
    if x_label:
        ax.set_xlabel(x_label, fontsize=10 * font_scale)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10 * font_scale)
    if title:
        ax.set_title(title, fontsize=12 * font_scale)
    if show_legend:
        ax.legend(fontsize=10 * font_scale)
    ax.grid(True, linestyle='--', alpha=0.4)

    _finalize(fig)


# ===========================================================================
# 결정 경계 (1·2장 퍼셉트론 시각화 통합본)
# ===========================================================================

def _draw_sample_points(ax, sample, sample_name='', font_scale=1.0):
    """sample = (x1_list, x2_list, y_list) 를 정답에 따라 빈 원/채운 원으로 그린다."""
    points = {0: ([], []), 1: ([], [])}
    for x1, x2, y in zip(sample[0], sample[1], sample[2]):
        ax.text(x1 + 0.05, x2 - 0.1, f'({x1}, {x2})',
                fontsize=10 * font_scale)
        idx = 0 if y < 0.5 else 1
        points[idx][0].append(x1)
        points[idx][1].append(x2)

    ax.scatter(*points[0], marker='o', edgecolor='black', facecolor='none',
               label=f'{sample_name} (클래스 0)' if sample_name else '클래스 0')
    ax.scatter(*points[1], marker='o', edgecolor='black', facecolor='black',
               label=f'{sample_name} (클래스 1)' if sample_name else '클래스 1')


def _apply_decision_boundary_axes(ax, x_lim, y_lim, x_label, y_label,
                                  font_scale=1.0):
    """결정 경계 시각화용 축 스타일을 적용한다 (원점 교차, 그리드 등)."""
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position('zero')
        ax.spines[spine].set_alpha(0.5)
    for spine in ['right', 'top']:
        ax.spines[spine].set_color('none')

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_label:
        ax.set_xlabel(x_label, loc='right', fontsize=10 * font_scale)
        ax.xaxis.set_label_coords(0.98, 0.14)
    if y_label:
        ax.set_ylabel(y_label, loc='top', rotation=0,
                      fontsize=10 * font_scale)
        ax.yaxis.set_label_coords(0.14, 0.95)
    ax.set_xticks([1])
    ax.set_yticks([1])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(labelbottom=False, labelleft=False)


def plot_decision_boundary(sample, *, boundary_fn=None, params=None,
                           title=None, sample_name='', baseline_name=None,
                           x_lim=(-0.5, 1.5), y_lim=(-0.5, 1.5),
                           x_label=None, y_label=None,
                           show_legend=True, figsize=(8, 6), font_scale=1.0):
    """샘플 데이터와 결정 경계를 함께 시각화한다.

    인자
        sample: ``(x1_list, x2_list, y_list)`` 3-튜플.
        boundary_fn: ``x`` 텐서를 받아 결정 경계 y값을 반환하는 함수. **단일 경계**.
        params: ``[(w1, w2, b), ...]`` 직선 파라미터 목록. **다중 경계**.
                boundary_fn과 동시에 지정하면 ValueError.
        title, sample_name, baseline_name: 텍스트.
        x_lim, y_lim: 축 범위 (기본 (-0.5, 1.5)).
        show_legend: 범례 표시 여부.
        figsize: 그림 크기.
        font_scale: 모든 텍스트(제목·좌표·축·범례)에 적용할 폰트 크기 배율.
            1.0이 기본.
    """
    set_korean_plot_env()
    if boundary_fn is not None and params is not None:
        raise ValueError('boundary_fn 과 params 는 동시에 지정할 수 없습니다.')

    fig, ax = plt.subplots(figsize=figsize)

    _draw_sample_points(ax, sample, sample_name, font_scale=font_scale)

    x_range = torch.linspace(x_lim[0], x_lim[1], 400)
    if boundary_fn is not None:
        ax.plot(x_range.numpy(), boundary_fn(x_range).numpy()
                if isinstance(boundary_fn(x_range), torch.Tensor)
                else boundary_fn(x_range),
                color=_COLOR_CYCLE[0], linewidth=1.5, label=baseline_name)
    elif params is not None:
        for i, (w1, w2, b) in enumerate(params):
            if w2 == 0:
                continue
            y_values = -(w1 / w2) * x_range.numpy() - (b / w2)
            style = _series_style(i)
            ax.plot(x_range.numpy(), y_values, color=style['color'],
                    linestyle=style['linestyle'], linewidth=1)

    _apply_decision_boundary_axes(ax, x_lim, y_lim, x_label, y_label,
                                  font_scale=font_scale)

    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
                  fontsize=10 * font_scale)
    if title:
        ax.set_title(title, pad=15, fontsize=20 * font_scale)

    _finalize(fig)


# ===========================================================================
# 혼동 행렬
# ===========================================================================

def plot_confusion_matrix(confusion_matrix, *, class_names=None,
                          show_errors_separately=True,
                          show_errors_only=False, figsize=None,
                          font_scale=1.0):
    """혼동 행렬을 히트맵으로 시각화한다.

    인자
        confusion_matrix: ``common_new.get_confusion_matrix()`` 가 반환한 텐서.
        class_names: 클래스 이름 목록. ``None`` 이면 숫자 인덱스.
        show_errors_separately: True이면 오른쪽에 오분류만 강조한 히트맵을 추가.
        show_errors_only: True이면 오분류 히트맵만 단독으로 그린다.
        figsize: 그림 크기. ``None`` 이면 자동.
        font_scale: 모든 텍스트(패널 제목·축 라벨·클래스 이름·셀 값)에 적용할
            폰트 크기 배율. 1.0이 기본.
    """
    set_korean_plot_env()
    cm = (confusion_matrix.numpy() if isinstance(confusion_matrix, torch.Tensor)
          else np.asarray(confusion_matrix))
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    errors_only = cm.copy()
    np.fill_diagonal(errors_only, 0)

    if show_errors_only:
        panels = [(errors_only, '혼동 행렬 (오분류)')]
    else:
        panels = [(cm, '혼동 행렬 (전체)')]
        if show_errors_separately:
            panels.append((errors_only, '혼동 행렬 (오분류)'))

    if figsize is None:
        figsize = (7 * len(panels), 6)

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, facecolor='white')
    if len(panels) == 1:
        axes = [axes]

    for ax, (data, panel_title) in zip(axes, panels):
        im = ax.imshow(data, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_title(panel_title, fontsize=16 * font_scale, pad=10)
        ax.set_xlabel('예측 레이블', fontsize=12 * font_scale)
        ax.set_ylabel('실제 레이블', fontsize=12 * font_scale)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, fontsize=10 * font_scale)
        ax.set_yticklabels(class_names, fontsize=10 * font_scale)

        threshold = data.max() / 2 if data.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                value = data[i, j]
                if panel_title == '혼동 행렬 (오분류)' and i == j:
                    text = 'NA'
                else:
                    text = format(int(value), 'd')
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=9 * font_scale,
                        color='white' if value > threshold else 'black')

    plt.tight_layout()
    _finalize(fig)


# ===========================================================================
# 이미지 격자
# ===========================================================================

def _prepare_image_for_display(img):
    """텐서/배열을 imshow가 받을 수 있는 형태로 정규화."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    else:
        img = np.asarray(img)
    if img.ndim == 1:
        side = int(img.shape[0] ** 0.5)
        img = img.reshape(side, side)
    elif img.ndim == 3:
        if img.shape[0] == 1:
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    return img


def plot_images(imgs, descs=None, *, images_per_row=8, fig_title_size=14,
                figsize=None, font_scale=1.0, title=None):
    """이미지 목록을 격자 형태로 시각화한다.

    인자
        imgs: 이미지 텐서/넘파이 배열의 리스트.
        descs: 이미지별 설명 문자열 리스트. ``None`` 이면 빈 제목.
        images_per_row: 한 행에 표시할 이미지 수.
        fig_title_size: 제목 폰트 크기 (font_scale 곱하기 전 베이스).
        figsize: 그림 크기. ``None`` 이면 자동 계산.
        font_scale: 모든 제목 텍스트에 적용할 폰트 크기 배율. 1.0이 기본.
    """
    set_korean_plot_env()
    n_imgs = len(imgs)
    if descs is None:
        descs = [''] * n_imgs
    if len(descs) != n_imgs:
        raise ValueError('imgs 와 descs 의 길이가 다릅니다.')

    n_rows = math.ceil(n_imgs / images_per_row)
    if figsize is None:
        figsize = (min(images_per_row * 1.6, 14), n_rows * 2.0)

    fig, axes = plt.subplots(n_rows, images_per_row, figsize=figsize,
                             facecolor='white')

    # plt.subplots 는 (n_rows, n_cols) 에 따라 단일 Axes / 1D / 2D 를 반환한다.
    # 모든 케이스를 (n_rows, images_per_row) 2D 로 통일한다.
    axes = np.asarray(axes).reshape(n_rows, images_per_row)

    for idx, (img, desc) in enumerate(zip(imgs, descs)):
        row, col = divmod(idx, images_per_row)
        ax = axes[row, col]
        img = _prepare_image_for_display(img)
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(desc, fontsize=fig_title_size * font_scale)
        ax.axis('off')

    for idx in range(n_imgs, n_rows * images_per_row):
        row, col = divmod(idx, images_per_row)
        axes[row, col].axis('off')
    if title:
        fig.suptitle(title, fontsize=16 * font_scale)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.tight_layout()
    _finalize(fig)


# ===========================================================================
# 특징 지도 (5장 CNN)
# ===========================================================================

def plot_feature_map(feature_map, *, title=None, images_per_row=8,
                     fig_title_size=12, figsize=None, channel_names=None,
                     font_scale=1.0):
    """CNN 특징 지도(채널별)를 격자로 시각화한다.

    인자
        feature_map: shape ``(C, H, W)`` 의 텐서/배열. 배치 차원이 포함된 ``(1, C, H, W)``
            도 자동 처리.
        title: 전체 제목.
        images_per_row: 한 행에 표시할 채널 수.
        fig_title_size: 채널 설명 폰트 크기 (font_scale 곱하기 전 베이스).
        figsize: 그림 크기. ``None`` 이면 자동.
        channel_names: 채널별 설명. ``None`` 이면 ``'채널 i'`` 형태.
        font_scale: 모든 텍스트에 적용할 폰트 크기 배율. 1.0이 기본.
    """
    set_korean_plot_env()
    if isinstance(feature_map, torch.Tensor):
        fm = feature_map.detach().cpu().numpy()
    else:
        fm = np.asarray(feature_map)
    if fm.ndim == 4 and fm.shape[0] == 1:
        fm = fm[0]
    if fm.ndim != 3:
        raise ValueError(f'feature_map 은 (C, H, W) 형태여야 합니다. shape={fm.shape}')

    c = fm.shape[0]
    if channel_names is None:
        channel_names = [f'채널 {i}' for i in range(c)]

    n_rows = math.ceil(c / images_per_row)
    if figsize is None:
        figsize = (min(images_per_row * 1.6, 14), n_rows * 1.8)

    fig, axes = plt.subplots(n_rows, images_per_row, figsize=figsize,
                             facecolor='white')
    axes = np.asarray(axes).reshape(n_rows, images_per_row)

    for idx in range(c):
        row, col = divmod(idx, images_per_row)
        ax = axes[row, col]
        ax.imshow(fm[idx], cmap='viridis')
        ax.set_title(channel_names[idx], fontsize=fig_title_size * font_scale)
        ax.axis('off')

    for idx in range(c, n_rows * images_per_row):
        row, col = divmod(idx, images_per_row)
        axes[row, col].axis('off')

    if title:
        fig.suptitle(title, fontsize=16 * font_scale)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout()
    _finalize(fig)


# ===========================================================================
# 어텐션 가중치 (9·10장)
# ===========================================================================

def plot_attention(attn_weights, *, source_tokens=None, target_tokens=None,
                   title=None, figsize=(8, 6), cmap='Greys', font_scale=1.0, gamma=1.0):
    """어텐션 가중치를 히트맵으로 시각화한다.

    인자
        attn_weights: shape ``(target_len, source_len)`` 의 텐서/배열.
        source_tokens: 소스 토큰 리스트 (x축 라벨).
        target_tokens: 타깃 토큰 리스트 (y축 라벨).
        title: 그래프 제목.
        figsize: 그림 크기.
        cmap: 컬러맵 이름.
        font_scale: 모든 텍스트(제목·축·토큰)에 적용할 폰트 크기 배율.
            1.0이 기본.
        gamma: 컬러맵 감마 보정. 1.0이 기본. 1보다 크면 밝게, 1보다 작으면 어둡게.
    """
    set_korean_plot_env()
    if isinstance(attn_weights, torch.Tensor):
        attn = attn_weights.detach().cpu().numpy()
    else:
        attn = np.asarray(attn_weights)
    if attn.ndim != 2:
        raise ValueError(f'attn_weights 는 2차원이어야 합니다. shape={attn.shape}')

    target_len, source_len = attn.shape

    fig, ax = plt.subplots(figsize=figsize)
    if gamma != 1.0:
        from matplotlib.colors import PowerNorm
        norm = PowerNorm(gamma, vmin=0.0, vmax=float(attn.max()))
        im = ax.imshow(attn, cmap=cmap, aspect='auto', norm=norm)
    else:
        im = ax.imshow(attn, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax)

    if source_tokens is not None:
        if len(source_tokens) != source_len:
            raise ValueError('source_tokens 길이가 attn 의 열 수와 다릅니다.')
        ax.set_xticks(range(source_len))
        ax.set_xticklabels(source_tokens, rotation=45, ha='right',
                           fontsize=10 * font_scale)
    if target_tokens is not None:
        if len(target_tokens) != target_len:
            raise ValueError('target_tokens 길이가 attn 의 행 수와 다릅니다.')
        ax.set_yticks(range(target_len))
        ax.set_yticklabels(target_tokens, fontsize=10 * font_scale)

    ax.set_xlabel('입력 토큰', fontsize=10 * font_scale)
    ax.set_ylabel('출력 토큰', fontsize=10 * font_scale)
    if title:
        ax.set_title(title, fontsize=12 * font_scale)
    plt.tight_layout()
    _finalize(fig)
