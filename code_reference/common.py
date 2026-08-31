"""딥러닝 입문서 공용 라이브러리 (common.py).

이 모듈은 환경 / 재현성, 분류 모델 평가, 모델 파라미터 카운트, 시간 측정,
GPU 메모리 보고 등 딥러닝 학습 흐름 자체가 아닌 공통 헬퍼를 모은다.
시각화 함수는 본 모듈에서 분리해 `code_reference.visualize` 모듈에 둔다.

포함 함수 / 클래스
    환경: get_device, check_colab, set_korean_plot_env
    재현성: set_seed
    분류 평가: get_accuracy, get_confusion_matrix, collect_error_samples
    파라미터: count_params, print_param_summary
    시간 측정: Timer (컨텍스트 매니저)
    GPU: report_gpu_memory, clear_gpu_cache

설계 원칙
    - 장마다 구현이 달라지는 학습 함수(train_epoch, validation_epoch,
      train_loop 등)는 본 모듈에서 의도적으로 제외한다. 각 장의 노트북에서
      본문에 맞춰 직접 정의한다.
    - 학습 함수를 제외하는 원칙의 예외로, 재현성 일괄 설정 함수 set_seed는 본 모듈에
      둔다. 노트북마다 반복되는 시드 고정 코드를 줄이기 위해서다. 어느 라이브러리의
      시드를 고정하는지 짚어야 하는 자리에서는 노트북 본문에 설명을 덧붙인다.
    - 로컬 PyTorch 환경(Mac · Windows · Linux)과 Google Colab 환경 모두에서
      동작해야 한다.
"""

import math
import os
import random
import subprocess
import sys
import time
from unicodedata import east_asian_width

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# 환경
# ---------------------------------------------------------------------------

def get_device(device_description=None):
    """학습에 사용할 장치를 반환한다.

    device_description에 지정한 디바이스를 우선 사용하며, 지정하지 않은 경우
    사용 가능한 디바이스를 cuda > mps > xla > cpu 순서로 선택한다.
    디바이스를 지정하면 실제 하드웨어가 없더라도 이 단계에서 예외가 발생하지
    않을 수 있으므로 주의.

    인자
        device_description: 'cuda', 'mps', 'cpu' 등. None이면 자동 선택.

    반환
        torch.device 객체.
    """
    if device_description:
        try:
            device = torch.device(device_description)
        except RuntimeError:
            raise ValueError(
                f'파이토치에 {device_description}로 정의된 하드웨어가 없습니다.'
            )
        print(f'{device_description.upper()}를 사용합니다.')
        return device

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA를 사용합니다.')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('MPS를 사용합니다.')
    else:
        # 구글 TPU(XLA) 사용 여부는 장치 객체를 가져올 수 있는지로 판단
        try:
            import torch_xla.core.xla_model as xm   # 라이브러리가 없으면 ImportError
            device = xm.xla_device()                 # 장치를 못 가져오면 RuntimeError
            print('XLA를 사용합니다.')
        except (ImportError, RuntimeError):
            device = torch.device('cpu')
            print('CPU를 사용합니다.')
    return device


def set_seed(seed=42, deterministic=False):
    """재현성을 위해 난수 생성기의 시드를 한 번에 고정한다.

    파이썬 random, 넘파이, 파이토치(CPU와 모든 GPU)의 시드를 함께 설정한다.
    노트북마다 반복되는 다음 코드를 대체한다.

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    인자
        seed: 사용할 시드값. 기본 42.
        deterministic: True이면 cuDNN의 비결정적 알고리즘 선택까지 막아
                       같은 하드웨어에서 실행 결과를 완전히 고정한다.
                       학습 속도가 느려질 수 있어 기본값은 False.

    반환
        설정한 시드값(int).

    참고
        시드를 고정해도 CPU와 GPU는 연산 방식이 달라 결과가 다를 수 있다.
        여러 워커를 사용하는 데이터로더는 worker_init_fn으로 워커별 시드를
        따로 지정해야 완전히 재현된다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def check_colab():
    """실습 환경이 구글 코랩인지 확인한다.

    반환
        bool. 코랩이면 True, 아니면 False.
    """
    try:
        import google.colab   # noqa: F401
        return True
    except ImportError:
        return False


def set_korean_plot_env():
    """matplotlib 시각화에 한글 표기를 위해 폰트를 지정한다.

    윈도는 맑은 고딕, 맥오에스는 애플 고딕, 리눅스/콜랩은 나눔 고딕을 가정한다.
    폰트가 설치되지 않은 경우 코랩 환경이면 자동 설치하고, 그 외에는
    FileNotFoundError를 발생시킨다.
    """
    font_names = {
        'win32': 'Malgun Gothic',
        'darwin': 'AppleGothic',
        'linux': 'NanumGothic',
    }

    if sys.platform not in font_names:
        raise OSError(f'지원하지 않는 운영체제입니다: {sys.platform}')
    font_name = font_names[sys.platform]
    font_list = [f.name for f in fm.fontManager.ttflist]

    if font_name not in font_list:
        if check_colab():
            # 구글 코랩: 나눔 폰트 설치 후 노트북 세션 재시작 필요
            try:
                subprocess.run(
                    ['sudo', 'apt-get', 'install', '-y', 'fonts-nanum'],
                    check=True,
                )
                subprocess.run(['sudo', 'fc-cache', '-fv'], check=True)
                cache_dir = os.path.expanduser('~/.cache/matplotlib')
                if os.path.exists(cache_dir):
                    subprocess.run(['rm', '-rf', cache_dir], check=True)
                    print('폰트 설치를 마쳤습니다. 노트북 세션을 다시 시작하세요.')
            except subprocess.CalledProcessError as e:
                print(f'폰트 설치 중 오류 발생: {e}')
                raise
        else:
            raise FileNotFoundError(
                f'{sys.platform} 운영체제에서 {font_name} 폰트를 찾을 수 없습니다. '
                '해당 폰트를 설치하고 커널을 재시작한 후 다시 시도하세요.'
            )
    else:
        plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# 분류 평가
# ---------------------------------------------------------------------------

def get_accuracy(model, data_loader, device=None):
    """분류 모델의 정확도(%), 정답 샘플 수, 전체 샘플 수를 반환한다.

    데이터로더 기반의 미니배치 평가용. 텐서 직접 입력 형태(3장에서 본문이
    직접 보여주는 버전)는 본 함수가 다루지 않으므로 노트북에서 따로 정의한다.

    인자
        model: 평가할 모델 (eval 모드로 자동 전환)
        data_loader: (X, Y)를 배치 단위로 yield 하는 DataLoader
        device: 모델·텐서를 옮길 장치. None이면 CPU 사용.

    반환
        (accuracy, n_correct, n_samples)
            accuracy: 정확도(%)
            n_correct: 정답 샘플 수
            n_samples: 전체 샘플 수
    """
    if device is None:
        device = torch.device('cpu')
    model.eval()
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            Y_pred = model(X)
            Y_pred_labels = Y_pred.argmax(dim=1)
            n_correct += (Y_pred_labels == Y).sum().item()
            n_samples += X.size(0)
    return n_correct / n_samples * 100, n_correct, n_samples


def get_confusion_matrix(model, data_loader, num_classes=10, device=None):
    """분류 모델의 혼동 행렬을 계산한다.

    인자
        model: 평가할 모델 (eval 모드로 자동 전환)
        data_loader: (X, Y)를 배치 단위로 yield 하는 DataLoader
        num_classes: 분류 클래스 수 (행/열 크기)
        device: 모델·텐서를 옮길 장치. None이면 CPU 사용.

    반환
        torch.Tensor: shape (num_classes, num_classes), dtype torch.long.
        행 = 실제 레이블, 열 = 예측 레이블.
    """
    if device is None:
        device = torch.device('cpu')
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            Y_pred = model(X)
            Y_pred_labels = Y_pred.argmax(dim=1)
            for y, y_pred_label in zip(Y, Y_pred_labels):
                confusion_matrix[y.item()][y_pred_label.item()] += 1
    return confusion_matrix


def collect_error_samples(model, data_loader, error_pairs, n_samples=2):
    """지정한 오분류 패턴의 샘플을 수집한다.

    인자
        model: 평가할 모델 (eval 모드로 자동 전환)
        data_loader: (X, Y)를 배치 단위로 yield 하는 DataLoader
        error_pairs: [(실제 레이블, 예측 레이블), ...] 형태의 목록
        n_samples: 패턴별로 수집할 샘플 수

    반환
        dict: {(y_true, y_pred): [이미지 텐서, ...]} 형태.
        모든 패턴에 n_samples 만큼 모이면 조기 종료한다.
    """
    model.eval()
    collected = {pair: [] for pair in error_pairs}
    with torch.no_grad():
        for X, Y in data_loader:
            Y_pred = model(X).argmax(dim=1)
            for img, y_true, y_pred in zip(X, Y, Y_pred):
                pair = (y_true.item(), y_pred.item())
                if pair in collected and len(collected[pair]) < n_samples:
                    collected[pair].append(img)
            if all(len(v) >= n_samples for v in collected.values()):
                break
    return collected


# ---------------------------------------------------------------------------
# 파라미터 카운트
# ---------------------------------------------------------------------------

def count_params(module, trainable_only=False):
    """모듈의 파라미터 수를 센다.

    인자
        module: nn.Module 인스턴스
        trainable_only: True이면 requires_grad=True 인 파라미터만 센다.

    반환
        int. 파라미터 수.
    """
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def print_param_summary(modules, label_width=14):
    """여러 모듈의 파라미터 수와 학습 비율을 한 표로 출력한다.

    인자
        modules: {표시 라벨: 모듈} 형태의 딕셔너리.
        label_width: 라벨 컬럼 너비.

    출력 형식
        프로젝션 (학습)  :    2,757,888
        CLIP (동결)      :  151,277,313
        GPT-2 (동결)     :  124,439,808
        학습 비율: 0.99%
    """
    counts = {name: count_params(m) for name, m in modules.items()}
    total = sum(counts.values())
    for name, c in counts.items():
        print(f'{name:<{label_width}}: {c:>12,}')
    trainable = sum(count_params(m, trainable_only=True) for m in modules.values())
    print(f'학습 비율: {trainable / total * 100:.2f}%')


# ---------------------------------------------------------------------------
# 시간 측정
# ---------------------------------------------------------------------------

class Timer:
    """경과 시간 측정 컨텍스트 매니저.

    사용 예
        with common.Timer() as t:
            train_loss = train_epoch(...)
            val_loss = evaluate(...)
        print(f'에포크 시간: {t.elapsed:.1f}초')

    속성
        elapsed: with 블록을 빠져나온 직후 경과 초(float).
    """

    def __enter__(self):
        self.start = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.time() - self.start


# ---------------------------------------------------------------------------
# GPU 메모리
# ---------------------------------------------------------------------------

def report_gpu_memory(label=''):
    """현재 할당된 GPU 메모리를 GB 단위로 출력한다.

    CUDA가 사용 불가능한 환경에서는 아무 동작도 하지 않는다(MPS·CPU·XLA 환경).

    인자
        label: 출력 앞에 붙일 짧은 식별 문구. 비어 있으면 생략.
    """
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    prefix = f'{label} ' if label else ''
    print(f'{prefix}GPU 메모리: {allocated:.2f} GB')


def clear_gpu_cache():
    """torch.cuda.empty_cache()를 안전 분기 후 호출한다.

    CUDA가 사용 불가능한 환경에서는 아무 동작도 하지 않는다.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 학습 로그
# ---------------------------------------------------------------------------

def _disp_width(text):
    """한글 등 동아시아 전각 문자를 2로 계산한 표시 폭."""
    return sum(2 if east_asian_width(ch) in ('W', 'F') else 1 for ch in str(text))


def _pad(text, width, align='>'):
    """표시 폭(_disp_width) 기준으로 좌/우 정렬 패딩한다."""
    text = str(text)
    space = ' ' * max(0, width - _disp_width(text))
    return space + text if align == '>' else text + space


def _format_duration(seconds):
    """초를 'M:SS' 또는 'H:MM:SS'로 표기한다."""
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{secs:02d}'
    return f'{minutes}:{secs:02d}'


class EpochLogger:
    """학습 로그를 표로 출력하고, 손실 이력을 모아 학습 곡선까지 그린다.

    설계 원칙
        - 출력 간격을 고정하지 않고 **출력 줄 수**(``target_rows``)를 고정한다.
          간격은 ``every = ceil(EPOCHS / target_rows)``로 자동 계산하며, 첫
          에포크와 마지막 에포크는 항상 출력한다(에포크 수와 무관하게 표가
          ~10행으로 유지된다).
        - ``summary()``는 **항상 전체 학습 시간**을 함께 출력한다.
        - ``plot()``은 책 예제에 그림이 없더라도 **항상 학습 곡선**을 그린다.
          (손실 이력은 출력 여부와 무관하게 매 에포크 누적된다.)

    인자
        epochs: 전체 에포크 수.
        columns: 표의 메트릭 열 이름. 기본 ``('훈련 손실', '검증 손실', '정확도(%)')``.
        formats: 각 열의 ``str.format`` 템플릿. 기본 ``('{:.4f}', '{:.4f}', '{:.2f}%')``.
            정확도는 백분율 수(0~100)로 전달한다(``common.get_accuracy`` 반환값과 동일).
        target_rows: 목표 출력 행 수(기본 10).
        minimize: 최적 에포크 판정 기준 열 이름(기본 ``'검증 손실'``). 해당 열이
            없으면 첫 손실 열을 사용하고, 없으면 최적 추적을 끈다.
        show_time: 누적 학습 시간 열 표시 여부(기본 True).

    사용 예
        log = common.EpochLogger(EPOCHS)
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(...)
            valid_loss, valid_acc = evaluate(...)
            log.row(epoch, train_loss, valid_loss, valid_acc)
        log.summary()          # 최적 에포크 + 전체 학습 시간
        log.plot()             # 학습 곡선(항상)
    """

    def __init__(self, epochs, columns=('훈련 손실', '검증 손실', '정확도(%)'),
                 formats=('{:.4f}', '{:.4f}', '{:.2f}%'), target_rows=10,
                 minimize='검증 손실', show_time=True):
        self.epochs = int(epochs)
        self.columns = list(columns)
        self.formats = list(formats)
        self.target_rows = max(1, int(target_rows))
        self.every = max(1, math.ceil(self.epochs / self.target_rows))
        self.show_time = show_time

        if minimize in self.columns:
            self.minimize = minimize
        else:
            loss_cols = [c for c in self.columns if '손실' in c]
            self.minimize = loss_cols[0] if loss_cols else None

        self.history = {c: [] for c in self.columns}
        self.epochs_seen = []
        self.best_epoch = None
        self.best_value = None
        # 로거 생성 시점(학습 루프 직전)을 학습 시작 시각으로 삼아 누적 시간을 잰다.
        # row()는 에포크가 끝난 뒤 호출되므로, 여기서 잡지 않으면 첫 에포크 소요 시간이 누락된다.
        self._t0 = time.perf_counter()
        self._elapsed = None       # summary() 시 전체 학습 시간(초) 보관
        self._header_done = False

        self._epoch_w = max(_disp_width(f'{self.epochs}/{self.epochs}'), 5)
        self._col_w = [max(_disp_width(c), 11) for c in self.columns]
        self._time_w = max(_disp_width('시간'), 7)

    def _print_header(self):
        cells = [_pad('epoch', self._epoch_w)]
        cells += [_pad(c, w) for c, w in zip(self.columns, self._col_w)]
        if self.show_time:
            cells.append(_pad('시간', self._time_w))
        print('  '.join(cells))
        self._header_done = True

    def row(self, epoch, *values, is_print=True):
        """한 에포크의 메트릭을 기록한다. 출력 여부는 내부에서 판단한다."""
        for col, value in zip(self.columns, values):
            self.history[col].append(value)
        self.epochs_seen.append(epoch)

        if self.minimize is not None:
            target = values[self.columns.index(self.minimize)]
            if self.best_value is None or target < self.best_value:
                self.best_value, self.best_epoch = target, epoch

        if not self._header_done and is_print:
            self._print_header()

        if epoch == 1 or epoch == self.epochs or epoch % self.every == 0:
        
            cells = [_pad(f'{epoch}/{self.epochs}', self._epoch_w)]
            cells += [_pad(fmt.format(v), w)
                        for v, fmt, w in zip(values, self.formats, self._col_w)]
            if self.show_time:
                elapsed = time.perf_counter() - self._t0
                cells.append(_pad(_format_duration(elapsed), self._time_w))
            if is_print:
                print('  '.join(cells))

    def summary(self, stopped=None):
        """최적 에포크와 전체 학습 시간을 표 하단에 출력한다(항상 시간 포함)."""
        total = time.perf_counter() - self._t0 if self._t0 is not None else 0.0
        self._elapsed = total       # 전체 학습 시간을 보관해 표 등에서 재사용
        line_w = self._epoch_w + sum(self._col_w) + 2 * len(self._col_w)
        if self.show_time:
            line_w += self._time_w + 2
        print('-' * line_w)

        parts = []
        # 검증 손실을 기록(추적)할 때만 최적 에포크를 출력한다.
        # 검증 손실 없이 훈련 손실만 기록하는 학습에서는 '최적 에포크'가 의미가 없다.
        if self.minimize == '검증 손실' and self.best_epoch is not None:
            fmt = self.formats[self.columns.index(self.minimize)]
            parts.append(f'최적 {self.best_epoch} 에포크')
            parts.append(f'{self.minimize} {fmt.format(self.best_value)}')
        parts.append(f'전체 학습 시간 {_format_duration(total)}')
        if stopped:
            parts.append(f'({stopped})')
        print(' · '.join(parts))

    @property
    def elapsed(self):
        """전체 학습 시간(초). summary() 후엔 그때 잰 값을, 그 전엔 현재까지 경과를 반환."""
        if self._elapsed is not None:
            return self._elapsed
        return time.perf_counter() - self._t0 if self._t0 is not None else 0.0

    @property
    def num_epochs(self):
        """실제로 기록된(실행된) 에포크 수. 조기 종료 시 best_epoch가 아니라 실행 에포크 수다."""
        return len(self.epochs_seen)

    @property
    def seconds_per_epoch(self):
        """에포크당 평균 학습 시간(초) = 전체 학습 시간 / 실제 실행 에포크 수."""
        n = len(self.epochs_seen)
        return self.elapsed / n if n else 0.0

    def plot(self, title='학습 곡선', **kwargs):
        """수집한 손실 이력으로 학습 곡선을 그린다(그림은 항상 표시).

        손실 열이 하나면 단일 곡선, 둘 이상이면 한 축에 함께 그린다.
        """
        from . import visualize as viz

        loss_series = {c: self.history[c] for c in self.columns
                       if '손실' in c or '오차' in c}
        if not loss_series and self.minimize is not None:
            loss_series = {self.minimize: self.history[self.minimize]}
        if not loss_series:
            return
        if len(loss_series) == 1:
            (label, hist), = loss_series.items()
            viz.plot_history(hist, title=title, label=label, **kwargs)
        else:
            viz.plot_histories(loss_series, title=title, **kwargs)


def save_model(model, path):
    """모델 파라미터(state_dict)를 파일에 저장한다(책 [코드 4-6] 방식).

    상위 폴더가 없으면 만든다. 한 번 학습한 모델을 다음 절에서 다시 학습하지
    않고 불러와 재사용하기 위한 표준 헬퍼.
    """
    import os
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device=None):
    """저장된 state_dict 를 model 에 적재하고, device 로 옮긴 뒤 eval()로 둔다.

    인자
        model: 파라미터를 적재할 모델 인스턴스(같은 구조의 빈 모델).
        path: state_dict 파일 경로.
        device: 옮길 디바이스(None이면 그대로).
    반환: 파라미터가 적재된 model.
    """
    state = torch.load(path, map_location=device if device is not None else 'cpu')
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
    model.eval()
    return model
