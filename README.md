# 멀티모달까지 직접 구현하는 딥러닝 with 파이토치

본 저장소는 『멀티모달까지 직접 구현하는 딥러닝 with 파이토치』의 본문 예제와 연습 문제를 실행할 수 있는 노트북, 그리고 공통 라이브러리를 모은 것이다.

## 디렉터리 구조

```
.
├── README.md                # 이 문서
├── requirements.txt         # 모든 노트북의 공통 의존성
├── code_reference/          # 공통 라이브러리 (importable)
│   ├── __init__.py
│   ├── common.py            # 환경 / 평가 / 파라미터 / 시간 / GPU 헬퍼
│   └── visualize.py         # 시각화 헬퍼 (별칭: viz)
├── code_examples/           # 본문 예제 노트북 (절 단위)
│   ├── ch01/
│   │   ├── 01-01_example.ipynb
│   │   ├── 01-02_example.ipynb
│   │   └── ...
│   └── ch02/ ... ch13/
├── exercises/               # 연습 문제 풀이 노트북 (절 단위)
│   ├── ch01/
│   │   ├── 01-01_exercise.ipynb
│   │   └── ...
│   └── ch02/ ... ch13/
├── data/                    # 저자 제공 예제 데이터 (깃 추적)
│   ├── README.md
│   ├── ch3_spiral_data.csv
│   ├── alice_in_wonderland.txt
│   └── ...
├── downloads/               # 자동·외부 다운로드 데이터 (깃 추적 제외)
│   ├── README.md
│   ├── MNIST/ FashionMNIST/ ...   # torchvision 자동 다운로드
│   └── glove.6B.100d.txt          # 수동 다운로드
└── checkpoint/              # 학습 산출물 저장 위치 (깃 추적 제외)
    └── README.md
```

**`data/` vs `downloads/` 분리**

- `data/` — 저자가 제공하는 작은 예제 파일(CSV, 이미지, 텍스트). 깃 저장소에 함께 들어 있다.
- `downloads/` — torchvision · HuggingFace 등이 자동으로 받는 데이터셋과 사용자가 별도로 받아 두는 대형 외부 데이터. 디렉터리 내용은 `.gitignore` 로 추적에서 제외되며 첫 노트북 실행 시 자동 생성된다.
- `checkpoint/` — 노트북이 저장하는 모델 파라미터와 미세조정 산출물. 역시 `.gitignore` 로 추적에서 제외된다.

`code_examples/` 와 `exercises/` 아래에는 노트북(`.ipynb`) 외의 파일이 생기지 않는다. 내려받은 데이터는 `downloads/`, 학습 산출물은 `checkpoint/` 로 모이도록 모든 노트북의 경로를 통일했다.

노트북에서는 두 경로를 명시적으로 구분해 사용한다:

```python
# 자동 다운로드 (예: MNIST)
datasets.MNIST(root='../../downloads', train=True, download=True, transform=...)

# 저자 제공 (예: 회오리 CSV)
with open('../../data/ch3_spiral_data.csv') as f:
    ...

# 학습 산출물 저장 (예: 모델 파라미터)
torch.save(model.state_dict(), '../../checkpoint/model_relu.pt')
```

## 실행 환경

다음 환경에서 노트북이 정상 실행되도록 설계되었다.

- 로컬 Mac (Apple Silicon · Intel)
- 로컬 Windows (CUDA 또는 CPU)
- 로컬 Linux (CUDA 또는 CPU)
- Google Colab (CPU · GPU 런타임)

표준 실행 UI는 VS Code의 Jupyter 확장이다. JupyterLab, Jupyter Notebook 클래식, 코랩 웹 UI도 동작한다.

### 노트북 출력 셀의 생성 환경

각 노트북에 함께 저장된 출력 셀(손실값·정확도·시간·이미지 등)은 다음 환경에서 생성한 결과이다.

- **OS**: Linux (Windows 11 + WSL2)
- **GPU**: NVIDIA CUDA 13.2
- **Python**: 3.14
- **PyTorch**: 2.12 (cu132 wheel)

독자의 환경(Mac · Windows 네이티브 · Colab · CPU 전용 등)에 따라 다음 정도의 차이가 발생할 수 있으며, 정성적 결론(학습 경향·정확도 수준 등)은 동일하게 유지된다.

- 부동소수점 연산 순서·정밀도 차이로 인한 손실·정확도의 마지막 자리 수 차이
- CPU·MPS 환경에서의 실행 시간 증가
- 일부 라이브러리(`bitsandbytes` 등) CUDA 미지원 환경에서의 실행 불가 (아래 CUDA / GPU 필요 여부 표 참조)

### CUDA / GPU 필요 여부

대부분의 노트북은 **CPU 환경에서도 동작**한다. 다만 일부 노트북은 GPU(특히 CUDA)에서만 동작하거나, CPU에서는 실행 시간이 길어 사실상 권장하지 않는다.

| 장 · 노트북 | CUDA 필요 여부 | 비고 |
|------------|---------------|------|
| 1~7장 전체 | ✓ CPU 가능 | CPU만으로도 충분 (텐서 기초·MLP·CNN 기초·RNN·임베딩) |
| 8장 (전이 학습) | △ CPU 가능 (느림) | ResNet·VGG 추론은 빠르지만 학습은 GPU 권장 |
| 9·10장 (Seq2Seq·트랜스포머) | △ CPU 가능 (느림) | 학습 데이터 규모를 줄이면 CPU 학습 가능 |
| 11장 (VAE·GAN·디퓨전) | △ CPU 가능 (느림) | 학습 시간 길어 GPU 권장 |
| **12-01** (LLM 추론) | △ CPU 가능 (매우 느림) | `torch_dtype=torch.float16` 대신 `float32` 로 변경하면 CPU 동작. 응답이 분 단위로 느려질 수 있음 |
| **12-02** (SFT) | ⚠ CUDA 권장 | 학습량 큼. CPU 학습은 권장하지 않음 |
| **12-03** (QLoRA) | ✕ **CUDA 필수** | 4비트 양자화(`bitsandbytes`)는 CUDA 전용 |
| **12-04** (RAG) | △ CPU 가능 | 임베딩 인코딩·검색은 CPU 가능. LLM 호출은 외부 API 사용 시 GPU 불필요 |
| **13-01** (멀티모달 추론) | △ CPU 가능 | CLIP·BLIP-2 추론은 CPU에서도 동작 (느림) |
| **13-02** (멀티모달 학습) | △ CPU 가능 (매우 느림) | 본문 안내 기준 CPU에서 200샘플·3에포크에 20~30분. GPU 권장 |

#### 환경별 정리

- **Google Colab**: GPU 런타임 무료 사용. **12-03 (QLoRA) 실행 시에는 반드시 GPU 런타임 선택** (런타임 → 런타임 유형 변경 → T4 GPU 등).
- **Mac (Apple Silicon)**: MPS 백엔드를 통해 일부 가속 가능. 다만 `bitsandbytes` 가 MPS를 지원하지 않으므로 **12-03 (QLoRA) 는 동작하지 않는다**. 다른 12장 노트북은 `torch_dtype=torch.float32` 로 변경하면 동작.
- **로컬 CUDA GPU**: 모든 노트북 정상 동작.
- **로컬 CPU**: 12-03 외에는 모두 동작 (12-01·12-02·13-02 는 느림 감수).

## 시작하기

### 로컬 (Mac · Windows · Linux + VS Code)

```bash
# 1. 저장소 받기
git clone https://github.com/crapas/dl-pytorch.git
cd dl-pytorch

# 2. 가상 환경 만들기 (선택, Python 3.12 ~ 3.14 권장)
python -m venv .venv
source .venv/bin/activate          # Mac · Linux
# 또는: .venv\Scripts\activate     # Windows

# 3. 공통 의존성 설치
pip install -r requirements.txt
```

`requirements.txt` 는 CPU 빌드 PyTorch를 가정한다. CUDA GPU를 사용하려면 §[CUDA 환경 설정](#cuda-환경-설정)을 추가로 참고한다.

```bash
# 4. 노트북 실행 (VS Code 권장)
# code_examples/ch01/01-01_example.ipynb 을 열고, 셀을 위에서 아래로 실행
# 인터프리터로 위에서 만든 .venv 또는 conda 환경을 선택
```

### CUDA 환경 설정

NVIDIA GPU 가 있는 환경이라면 **처음부터 본 절을 따라 CUDA 환경을 설정하기를 권한다.** 1~7장은 CPU 에서도 충분히 빠르지만 8장 이후로는 학습 시간이 점점 길어져, CUDA 환경에서 학습 회당 수십~수백 배의 차이가 난다. 또 12장 일부 절(특히 12-03 QLoRA)은 CUDA 가 없으면 아예 동작하지 않으므로, 책 전반을 따라가려면 CUDA 환경이 필요하다.

CPU 만 사용 가능한 환경(Apple Silicon · Intel Mac, GPU 없는 노트북 등)에서는 본 절을 건너뛰고 CPU 빌드 PyTorch 로 진행한다. 이 경우 12-03 QLoRA 와 같은 CUDA 전용 절은 코랩의 GPU 런타임에서 별도 실행하는 방식으로 보완한다.

#### (1) NVIDIA 드라이버 확인

CUDA 13.x 휠을 사용하려면 **NVIDIA 드라이버 580 이상**(2025년 하반기 출시) 권장. 다음 명령으로 현재 드라이버·CUDA 호환 버전을 확인한다.

```bash
nvidia-smi
```

출력 우측 상단의 `CUDA Version:` 이 12.8 이상이면 cu128, 13.x 이상이면 cu132 휠을 그대로 쓸 수 있다.

OS별 드라이버 설치 안내:

- **Windows**: NVIDIA App 또는 GeForce Experience 에서 자동 업데이트. 또는 [공식 드라이버](https://www.nvidia.com/Download/index.aspx) 다운로드.
- **Windows + WSL2**: Windows 측에 NVIDIA 드라이버만 설치. WSL 내부에는 별도 설치 불필요. WSL 안에서 `nvidia-smi` 가 정상 출력되면 성공.
- **Linux 네이티브**: 배포판 패키지(`nvidia-driver-XXX`) 또는 NVIDIA 공식 `.run` 설치 프로그램.
- **Mac**: NVIDIA CUDA 미지원. 12-03 (QLoRA) 등 CUDA 필수 노트북은 실행 불가. 다른 12장 노트북은 `torch_dtype=torch.float32` 로 변경 후 CPU 실행 가능.

#### (2) PyTorch CUDA 휠 설치

`requirements.txt` 의 PyTorch 는 CPU 빌드이므로, CUDA 환경에서는 다음 명령으로 **재설치(또는 덮어쓰기)** 한다. cu132(CUDA 13.2) 휠이 최신 안정 조합이다.

```bash
pip install --upgrade torch torchvision \
    --index-url https://download.pytorch.org/whl/cu132
```

드라이버가 CUDA 12.8 까지만 지원한다면 cu128 휠을 사용한다.

```bash
pip install --upgrade torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
```

설치 후 다음으로 동작 확인.

```python
import torch
print(torch.__version__)            # 예: 2.12.0+cu132
print(torch.cuda.is_available())    # True 여야 함
print(torch.cuda.get_device_name(0))
```

#### (3) bitsandbytes (12-03 QLoRA 전용)

`bitsandbytes` 는 CUDA 전용 4비트 양자화 라이브러리다. 12-03 노트북 첫 셀에 주석으로 설치 명령이 있다.

```bash
pip install bitsandbytes
```

Mac · CPU 전용 환경에서는 설치는 되어도 4비트 연산이 동작하지 않아 12-03 노트북이 실행 중 실패한다.

### Google Colab

각 노트북 첫 셀에 주석 처리된 형태로 다음 명령이 안내되어 있다. 코랩에서 실행할 때만 주석을 풀고 한 번 실행하면 된다.

```python
# 코랩에서 실행 시 (한 번만):
# !git clone https://github.com/crapas/dl-pytorch.git
# %cd dl-pytorch/code_examples/ch01
```

이후 노트북의 셀들은 로컬·코랩 동일하게 동작한다.

## 장 특화 라이브러리

`requirements.txt` 에는 모든 노트북의 공통 의존성만 들어 있다. 일부 장에는 추가 라이브러리가 필요한데, 해당 노트북 첫 셀에 주석으로 설치 명령이 적혀 있다.

| 장 | 추가 의존성 |
|----|------------|
| 8장 | `timm`, `kagglehub` |
| 12장 | `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `sentence-transformers`, `groq` |
| 13장 | `transformers`, `datasets`, `pillow` |

## 공통 라이브러리 사용 패턴

노트북에서 공통 라이브러리는 다음과 같이 임포트한다.

```python
import sys
sys.path.append('../../')                 # 저장소 루트를 모듈 검색 경로에 추가
from code_reference import common
# viz.configure()에서 save_grayscale=True로 지정하면 노트북에 표시되는 시각화 이미지를 파일로 저장함
from code_reference import visualize as viz

viz.configure(save_grayscale=False)
```

`code_examples/chNN/` 과 `exercises/chNN/` 디렉터리에서 `'../../'` 는 저장소 루트를 가리키고, 그 아래의 `code_reference/` 가 패키지로 임포트된다.

`viz.configure(save_grayscale=False)` 는 시각화 결과를 화면에만 표시하고 파일로는 남기지 않는다는 뜻이다. `True` 로 바꾸면 화면에 표시된 그림과 같은 내용의 회색조 300DPI PNG 가 노트북과 같은 디렉터리에 저장된다.

## 시드 정책

본 저장소의 노트북은 `SEED = 42` 를 기본 시드로 사용한다. 본문(책)은 일부 절에서 특정 시드(예: 1-3절의 시드 8)를 사용해 시각화 결과를 만든 부분이 있는데, 이 경우 노트북 상단에 명시한다.

## 라이선스

`LICENSE` 파일 참조.
