# `code_reference` API 문서

딥러닝 입문서 예제 노트북이 공통으로 사용하는 보조 라이브러리다. 두 서브모듈로 나뉜다.

| 모듈 | 책임 |
| --- | --- |
| [`common`](#common-모듈) | 환경/재현성, 분류 평가, 파라미터 카운트, 시간 측정, GPU 메모리, 학습 로그, 모델 저장·적재 |
| [`visualize`](#visualize-모듈) | 학습 곡선, 산포도, 결정 경계, 이미지·특징 지도, 혼동 행렬, 어텐션 시각화 |

> 설계상 장마다 구현이 달라지는 학습 함수(`train_epoch`, `evaluate`, `train_loop` 등)와 재현성 일괄 설정(`set_seed`)은 **의도적으로 제외**한다. 각 장 노트북에서 본문에 맞춰 직접 정의한다.

## 함수 요약표

### `common`

| 분류 | 심볼 | 한 줄 요약 |
| --- | --- | --- |
| 환경 | [`get_device`](#get_devicedevice_descriptionnone---torchdevice) | 학습 장치를 반환(미지정 시 cuda>mps>xla>cpu 자동 선택). |
| 환경 | [`check_colab`](#check_colab---bool) | 실습 환경이 구글 코랩인지 판별. |
| 환경 | [`set_korean_plot_env`](#set_korean_plot_env---none) | matplotlib 한글 폰트 설정(OS별). |
| 분류 평가 | [`get_accuracy`](#get_accuracymodel-data_loader-devicenone---float-int-int) | 미니배치 정확도(%)·정답수·전체수 반환. |
| 분류 평가 | [`get_confusion_matrix`](#get_confusion_matrixmodel-data_loader-num_classes10-devicenone---torchtensor) | 혼동 행렬 텐서 계산(행=실제, 열=예측). |
| 분류 평가 | [`collect_error_samples`](#collect_error_samplesmodel-data_loader-error_pairs-n_samples2---dict) | 지정한 오분류 패턴의 샘플 이미지 수집. |
| 파라미터 | [`count_params`](#count_paramsmodule-trainable_onlyfalse---int) | 모듈 파라미터 수를 센다(학습 대상만 선택 가능). |
| 파라미터 | [`print_param_summary`](#print_param_summarymodules-label_width14---none) | 여러 모듈의 파라미터 수·학습 비율을 표로 출력. |
| 시간 | [`Timer`](#class-timer) | 경과 시간 측정 컨텍스트 매니저(`.elapsed`). |
| GPU | [`report_gpu_memory`](#report_gpu_memorylabel---none) | 할당된 GPU 메모리를 GB로 출력(CUDA 없으면 무동작). |
| GPU | [`clear_gpu_cache`](#clear_gpu_cache---none) | GPU 캐시 비우기(CUDA 없으면 무동작). |
| 학습 로그 | [`EpochLogger`](#class-epochlogger) | 학습 로그 표 출력 + 손실 이력 누적 + 학습 곡선. |
| 모델 I/O | [`save_model`](#save_modelmodel-path---none) | `state_dict`를 파일로 저장(상위 폴더 자동 생성). |
| 모델 I/O | [`load_model`](#load_modelmodel-path-devicenone---model) | `state_dict` 적재 후 device 이동·`eval()`. |

### `visualize`

| 분류 | 심볼 | 한 줄 요약 |
| --- | --- | --- |
| 설정 | [`configure`](#configuresave_grayscalenone-notebook_namenone-output_dirnone---none) | 회색조 저장·파일명 접두어·디렉터리를 한 번에 설정. |
| 설정 | [`reset_counter`](#reset_counter---none) | 저장 파일명 일련번호를 1부터 다시 매김. |
| 학습 곡선 | [`plot_history`](#plot_historyhistory--titlenone-x_label에포크-y_label손실-y_limnone-x_from_0false-ticker_stepnone-ignore_before1-ignore_afterinf-labelnone-figsize10-6-font_scale10) | 단일 시계열 선 그래프. |
| 학습 곡선 | [`plot_histories`](#plot_historiesseries--titlenone-x_label에포크-y_label손실-y_limnone-x_limnone-x_ticksnone-y_ticksnone-x_ticker_stepnone-show_legendtrue-linewidth20-figsize10-6-font_scale10) | 다중 시계열을 한 축에(회색조 구분). |
| 산포도 | [`plot_scatter`](#plot_scatterx-ynone--sample_ratio10-titlenone-x_labelnone-y_labelnone-x_limnone-y_limnone-class_namesnone-point_size30-figsize8-6-font_scale10-add_ellipsesfalse-ellipse_confidence095-ellipse_linewidth16) | 2D 산포도(클래스별 색·신뢰 타원 옵션). |
| 산포도 | [`plot_regression_scatter`](#plot_regression_scatterx-y--reference_fnnone-reference_labelnone-sample_ratio10-titlenone-x_labelnone-y_labelnone-x_limnone-y_limnone-point_size30-figsize8-6-font_scale10) | 1D 회귀 산포도 + 기준선. |
| 활성화 | [`plot_activation_compare`](#plot_activation_compareactivations--x_range-5-5-num_points400-titlenone-x_label입력-y_label출력-x_limnone-y_limnone-show_legendtrue-linewidth20-figsize8-6-font_scale10) | 여러 활성화 함수를 한 축에 비교. |
| 결정 경계 | [`plot_decision_boundary`](#plot_decision_boundarysample--boundary_fnnone-paramsnone-titlenone-sample_name-baseline_namenone-x_lim-05-15-y_lim-05-15-x_labelnone-y_labelnone-show_legendtrue-figsize8-6-font_scale10) | 샘플 + 결정 경계(단일/다중) 시각화. |
| 이미지 | [`plot_images`](#plot_imagesimgs-descsnone--images_per_row8-fig_title_size14-figsizenone-font_scale10-titlenone) | 이미지 목록을 격자로 표시. |
| 이미지 | [`plot_feature_map`](#plot_feature_mapfeature_map--titlenone-images_per_row8-fig_title_size12-figsizenone-channel_namesnone-font_scale10) | CNN 특징 지도(채널별)를 격자로. |
| 분류 평가 | [`plot_confusion_matrix`](#plot_confusion_matrixconfusion_matrix--class_namesnone-show_errors_separatelytrue-show_errors_onlyfalse-figsizenone-font_scale10) | 혼동 행렬 히트맵(오분류 강조 옵션). |
| 어텐션 | [`plot_attention`](#plot_attentionattn_weights--source_tokensnone-target_tokensnone-titlenone-figsize8-6-cmapgreys-font_scale10-gamma10) | 어텐션 가중치 히트맵(감마 보정). |

## 권장 import 패턴

```python
import sys
sys.path.append('../../')                 # 워크스페이스 루트를 path에 추가
from code_reference import common
from code_reference import visualize as viz
```

---

## `common` 모듈

### 환경

#### `get_device(device_description=None) -> torch.device`
학습에 사용할 장치를 반환한다.

- **device_description**: `'cuda'`, `'mps'`, `'cpu'` 등. `None`이면 `cuda > mps > xla > cpu` 순서로 자동 선택.
- 디바이스를 명시하면 실제 하드웨어가 없어도 이 단계에서 예외가 나지 않을 수 있으니 주의.
- 선택한 장치를 안내 메시지로 출력한다.

#### `check_colab() -> bool`
실습 환경이 구글 코랩인지 확인한다. 코랩이면 `True`.

#### `set_korean_plot_env() -> None`
matplotlib 한글 표기를 위해 폰트를 지정한다. 윈도=맑은 고딕, 맥=애플 고딕, 리눅스/코랩=나눔 고딕. 폰트가 없으면 코랩에서는 자동 설치(세션 재시작 필요), 그 외 환경에서는 `FileNotFoundError`. 지원하지 않는 OS는 `OSError`.

> `visualize`의 모든 함수가 내부에서 자동 호출하므로, 시각화만 쓸 때는 직접 부를 필요가 없다.

### 분류 평가

#### `get_accuracy(model, data_loader, device=None) -> (float, int, int)`
데이터로더 기반 미니배치 정확도 평가. 모델을 `eval()`로 전환한다.

- 반환: `(accuracy, n_correct, n_samples)` — `accuracy`는 백분율(0~100).
- `device=None`이면 CPU.
- 텐서를 직접 받는 형태(3장 본문 버전)는 다루지 않는다.

#### `get_confusion_matrix(model, data_loader, num_classes=10, device=None) -> torch.Tensor`
혼동 행렬을 계산한다. 모델을 `eval()`로 전환한다.

- 반환: shape `(num_classes, num_classes)`, dtype `torch.long`. **행 = 실제 레이블, 열 = 예측 레이블**.
- 반환값을 `visualize.plot_confusion_matrix`에 그대로 넘길 수 있다.

#### `collect_error_samples(model, data_loader, error_pairs, n_samples=2) -> dict`
지정한 오분류 패턴의 샘플 이미지를 모은다.

- **error_pairs**: `[(실제 레이블, 예측 레이블), ...]`.
- 반환: `{(y_true, y_pred): [이미지 텐서, ...]}`. 모든 패턴에 `n_samples`만큼 모이면 조기 종료.

### 파라미터 카운트

#### `count_params(module, trainable_only=False) -> int`
모듈의 파라미터 수를 센다. `trainable_only=True`이면 `requires_grad=True`인 파라미터만.

#### `print_param_summary(modules, label_width=14) -> None`
여러 모듈의 파라미터 수와 학습 비율을 한 표로 출력한다.

- **modules**: `{표시 라벨: 모듈}`.
- 출력 예시:
  ```
  프로젝션 (학습)  :    2,757,888
  CLIP (동결)      :  151,277,313
  GPT-2 (동결)     :  124,439,808
  학습 비율: 0.99%
  ```

### 시간 측정

#### `class Timer`
경과 시간 측정 컨텍스트 매니저.

- 속성 **elapsed**: with 블록을 빠져나온 직후 경과 초(float).
```python
with common.Timer() as t:
    train_loss = train_epoch(...)
    val_loss = evaluate(...)
print(f'에포크 시간: {t.elapsed:.1f}초')
```

### GPU 메모리

#### `report_gpu_memory(label='') -> None`
현재 할당된 GPU 메모리를 GB 단위로 출력한다. CUDA가 없으면(MPS·CPU·XLA) 아무 동작도 하지 않는다. `label`은 출력 앞에 붙일 식별 문구.

#### `clear_gpu_cache() -> None`
`torch.cuda.empty_cache()`를 안전 분기 후 호출한다. CUDA가 없으면 무동작.

### 학습 로그

#### `class EpochLogger`
학습 로그를 표로 출력하고, 손실 이력을 모아 학습 곡선까지 그린다.

생성자:
```python
EpochLogger(epochs,
            columns=('훈련 손실', '검증 손실', '정확도(%)'),
            formats=('{:.4f}', '{:.4f}', '{:.2f}%'),
            target_rows=10,
            minimize='검증 손실',
            show_time=True)
```

| 인자 | 설명 |
| --- | --- |
| `epochs` | 전체 에포크 수. |
| `columns` | 표의 메트릭 열 이름. |
| `formats` | 각 열의 `str.format` 템플릿. 정확도는 백분율 수(0~100)로 전달(`get_accuracy` 반환값과 동일). |
| `target_rows` | 목표 출력 **행 수**(기본 10). 간격을 고정하지 않고 행 수를 고정 — `every = ceil(epochs / target_rows)`로 자동 계산하며 첫·마지막 에포크는 항상 출력. |
| `minimize` | 최적 에포크 판정 기준 열(기본 `'검증 손실'`). 없으면 첫 손실 열, 그것도 없으면 최적 추적을 끈다. |
| `show_time` | 누적 학습 시간 열 표시 여부. |

**메서드**

- `row(epoch, *values, is_print=True)` — 한 에포크 메트릭을 기록한다. 출력 여부(간격)는 내부에서 판단하며, 손실 이력은 출력과 무관하게 매 에포크 누적된다.
- `summary(stopped=None)` — 표 하단에 최적 에포크와 **전체 학습 시간**을 출력(항상 시간 포함). `stopped`에 조기 종료 사유 문자열을 넘기면 괄호로 덧붙인다. 검증 손실을 추적할 때만 최적 에포크를 출력한다.
- `plot(title='학습 곡선', **kwargs)` — 수집한 손실 이력으로 학습 곡선을 **항상** 그린다(손실 열 1개=단일 곡선, 2개 이상=한 축에 함께). `**kwargs`는 `visualize`의 plot 함수로 전달.

**속성(읽기 전용)**

| 속성 | 의미 |
| --- | --- |
| `elapsed` | 전체 학습 시간(초). `summary()` 후엔 그때 잰 값, 그 전엔 현재까지 경과. |
| `num_epochs` | 실제로 실행된 에포크 수(조기 종료 시 `best_epoch`가 아님). |
| `seconds_per_epoch` | 에포크당 평균 시간 = 전체 시간 / 실행 에포크 수. |

> 학습 시작 시각은 **로거 생성 시점**(학습 루프 직전)이다. `row()`는 에포크 종료 후 호출되므로, 생성 시점에 시작 시각을 잡지 않으면 첫 에포크 시간이 누락된다.

```python
log = common.EpochLogger(EPOCHS)
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(...)
    valid_loss, valid_acc = evaluate(...)
    log.row(epoch, train_loss, valid_loss, valid_acc)
log.summary()          # 최적 에포크 + 전체 학습 시간
log.plot()             # 학습 곡선(항상)
```

### 모델 저장·적재

#### `save_model(model, path) -> None`
모델 파라미터(`state_dict`)를 파일에 저장한다([코드 4-6] 방식). 상위 폴더가 없으면 만든다.

#### `load_model(model, path, device=None) -> model`
저장된 `state_dict`를 `model`에 적재하고, `device`로 옮긴 뒤 `eval()`로 둔다. `model`은 같은 구조의 빈 모델. 적재된 모델을 반환한다.

---

## `visualize` 모듈

화면 표시는 컬러로 하되, 회색조 자동 저장 정책과 다중 시리즈 구분 설계를 따른다.

### 회색조 자동 저장 정책

화면에는 컬러로 표시하고, `SAVE_GRAYSCALE = True`일 때 **같은 그림의 회색조 버전을 300DPI PNG로 자동 저장**한다. 호출 측은 저장을 신경 쓰지 않는다. 다중 시리즈는 색·선스타일·마커를 동시에 변동시켜 회색조 변환 후에도 구분된다. 산포도 함수에는 `sample_ratio`로 점 일부만 표시할 수 있다.

**모듈 변수**

| 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `SAVE_GRAYSCALE` | `False` | `True`이면 화면 표시와 동시에 회색조 PNG 저장. |
| `NOTEBOOK_NAME` | `None` | 저장 파일명 접두어. `None`이면 `'figure'`. 노트북 첫 셀에서 설정 권장. |
| `OUTPUT_DIR` | `None` | 저장 디렉터리. `None`이면 현재 작업 디렉터리. |

파일명 형식: `{NOTEBOOK_NAME}_{일련번호:02d}.png` (예: `02-02_example_01.png`).

#### `configure(save_grayscale=None, notebook_name=None, output_dir=None) -> None`
모듈 설정을 한 번에 변경한다. 각 인자가 `None`이면 해당 항목은 변경하지 않는다.

```python
viz.configure(save_grayscale=True, notebook_name='09-03_example')
```

#### `reset_counter() -> None`
저장 파일명의 일련번호를 1부터 다시 매긴다. 노트북 시작 셀에서 호출해 두면 셀 실행 순서가 바뀌어도 파일 인덱스가 예측 가능해진다.

### 학습 곡선

#### `plot_history(history, *, title=None, x_label='에포크', y_label='손실', y_lim=None, x_from_0=False, ticker_step=None, ignore_before=1, ignore_after=inf, label=None, figsize=(10, 6), font_scale=1.0)`
단일 시계열을 선 그래프로 그린다.

- **history**: 숫자 리스트(에포크 1부터 자동 매김) 또는 `(x, y)` 튜플 리스트.
- **ticker_step**: x 눈금 간격. `None`이면 에포크 수와 무관하게 약 5개로 자동 제한(정수 눈금). 정수를 주면 그 간격으로 고정.
- **ignore_before / ignore_after**: 표시 구간 필터(초반 급락 구간 제외 등).
- **label**: 범례 표시명. `None`이면 범례 없음.
- **font_scale**: 모든 텍스트 폰트 배율. 책에 작게 싣는 그림은 1.4~1.8 권장.
- 유효 점이 2개 미만이면 `ValueError`.

#### `plot_histories(series, *, title=None, x_label='에포크', y_label='손실', y_lim=None, x_lim=None, x_ticks=None, y_ticks=None, x_ticker_step=None, show_legend=True, linewidth=2.0, figsize=(10, 6), font_scale=1.0)`
다중 시계열을 한 축에 그린다. 회색조 후에도 구분되도록 색·선스타일·마커를 동시에 변동.

- **series**: `{라벨: [y...] 또는 [(x, y)...]}` 딕셔너리, 또는 `[(라벨, history), ...]` 리스트.
- **x_ticker_step**: x 눈금 간격(정수). `x_lim`과 함께 줄 때 적용. 정수가 아니면 `ValueError`.

### 산포도 / 회귀 / 활성화

#### `plot_scatter(X, y=None, *, sample_ratio=1.0, title=None, x_label=None, y_label=None, x_lim=None, y_lim=None, class_names=None, point_size=30, figsize=(8, 6), font_scale=1.0, add_ellipses=False, ellipse_confidence=0.95, ellipse_linewidth=1.6)`
2D 산포도를 그린다.

- **X**: shape `(N, 2)` 텐서/배열, 또는 `(x1_list, x2_list)` 튜플.
- **y**: shape `(N,)` 클래스 라벨. `None`이면 단일 색.
- **sample_ratio**: `0 < r ≤ 1`. `r < 1`이면 그 비율만 무작위 추출.
- **class_names**: 라벨별 범례명. `None`이면 숫자.
- **add_ellipses**: `True`이면 클래스별 신뢰 타원을 함께 그린다(`y` 필요). `ellipse_confidence`(기본 0.95), `ellipse_linewidth`로 조절.

#### `plot_regression_scatter(X, y, *, reference_fn=None, reference_label=None, sample_ratio=1.0, title=None, x_label=None, y_label=None, x_lim=None, y_lim=None, point_size=30, figsize=(8, 6), font_scale=1.0)`
1D 회귀 산포도. 기준선(참값/모델 예측)을 함께 그릴 수 있다.

- **X, y**: shape `(N,)` 1D 텐서/배열.
- **reference_fn**: x 텐서를 받아 y를 반환하는 함수. `None`이면 산포만. `reference_label`로 범례명 지정.

#### `plot_activation_compare(activations, *, x_range=(-5, 5), num_points=400, title=None, x_label='입력', y_label='출력', x_lim=None, y_lim=None, show_legend=True, linewidth=2.0, figsize=(8, 6), font_scale=1.0)`
여러 활성화 함수를 한 축에 비교한다. 원점 교차선을 함께 그린다.

- **activations**: `{라벨: 함수}` dict 또는 `[(라벨, 함수), ...]` list. 함수는 1D 텐서를 받아 1D 텐서/배열을 반환.
- **x_range**: 입력 범위, **num_points**: x 격자 점 수.

### 결정 경계

#### `plot_decision_boundary(sample, *, boundary_fn=None, params=None, title=None, sample_name='', baseline_name=None, x_lim=(-0.5, 1.5), y_lim=(-0.5, 1.5), x_label=None, y_label=None, show_legend=True, figsize=(8, 6), font_scale=1.0)`
샘플 데이터와 결정 경계를 함께 시각화한다(1·2장 퍼셉트론 통합본). 정답에 따라 빈 원(클래스 0)/채운 원(클래스 1)으로 표시하고 원점 교차 축을 적용한다.

- **sample**: `(x1_list, x2_list, y_list)` 3-튜플.
- **boundary_fn**: x 텐서를 받아 경계 y를 반환하는 함수. **단일 경계**.
- **params**: `[(w1, w2, b), ...]` 직선 파라미터 목록. **다중 경계**.
- `boundary_fn`과 `params`를 **동시에 지정하면 `ValueError`**.

### 이미지 / 특징 지도

#### `plot_images(imgs, descs=None, *, images_per_row=8, fig_title_size=14, figsize=None, font_scale=1.0, title=None)`
이미지 목록을 격자로 시각화한다.

- **imgs**: 이미지 텐서/배열의 리스트. 1D(정사각 복원), `(1,H,W)`, `(3,H,W)`, `(H,W)` 모두 자동 처리.
- **descs**: 이미지별 설명. `None`이면 빈 제목. 길이가 `imgs`와 다르면 `ValueError`.

#### `plot_feature_map(feature_map, *, title=None, images_per_row=8, fig_title_size=12, figsize=None, channel_names=None, font_scale=1.0)`
CNN 특징 지도(채널별)를 격자로 시각화한다(5장).

- **feature_map**: shape `(C, H, W)` 텐서/배열. `(1, C, H, W)`도 자동 처리. 그 외 형태는 `ValueError`.
- **channel_names**: 채널별 설명. `None`이면 `'채널 i'`. 컬러맵은 `viridis` 고정.

### 혼동 행렬

#### `plot_confusion_matrix(confusion_matrix, *, class_names=None, show_errors_separately=True, show_errors_only=False, figsize=None, font_scale=1.0)`
혼동 행렬을 히트맵으로 시각화한다.

- **confusion_matrix**: `common.get_confusion_matrix()`가 반환한 텐서.
- **show_errors_separately**: `True`이면 오른쪽에 오분류만 강조한 히트맵을 추가(대각=`NA`).
- **show_errors_only**: `True`이면 오분류 히트맵만 단독으로.

### 어텐션

#### `plot_attention(attn_weights, *, source_tokens=None, target_tokens=None, title=None, figsize=(8, 6), cmap='Greys', font_scale=1.0, gamma=1.0)`
어텐션 가중치를 히트맵으로 시각화한다(9·10장).

- **attn_weights**: shape `(target_len, source_len)` 텐서/배열. 2차원이 아니면 `ValueError`.
- **source_tokens**: x축 라벨(길이 = 열 수), **target_tokens**: y축 라벨(길이 = 행 수). 길이가 안 맞으면 `ValueError`.
- **cmap**: 컬러맵 이름(기본 `'Greys'`).
- **gamma**: 컬러맵 감마 보정(`PowerNorm`). 1.0이 기본. 1보다 크면 약한 값을 밝게, 1보다 작으면 어둡게. 한쪽 셀이 압도적으로 진해 약한 어텐션이 안 보일 때 1 미만으로 조절.
- 축 라벨은 `입력 토큰`(x), `출력 토큰`(y) 고정.

---

## 부록: 공개 API 한눈에

**common**
`get_device` · `check_colab` · `set_korean_plot_env` · `get_accuracy` · `get_confusion_matrix` · `collect_error_samples` · `count_params` · `print_param_summary` · `Timer` · `report_gpu_memory` · `clear_gpu_cache` · `EpochLogger` · `save_model` · `load_model`

**visualize**
`configure` · `reset_counter` · `plot_history` · `plot_histories` · `plot_scatter` · `plot_regression_scatter` · `plot_activation_compare` · `plot_decision_boundary` · `plot_images` · `plot_feature_map` · `plot_confusion_matrix` · `plot_attention`
