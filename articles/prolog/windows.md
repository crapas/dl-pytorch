# 실습 환경 구성 — 윈도우

> **대상**: 윈도우 10 이상을 쓰는 독자
> 리눅스 환경이 필요하거나 WSL2를 쓰려면 [리눅스 + CPU](linux-cpu.md) 또는 [리눅스 + CUDA](linux-cuda.md)를 보자.

윈도우에서는 파이토치를 그대로 설치해 쓸 수 있다. NVIDIA 그래픽 카드가 있으면 CUDA 버전을, 없으면 CPU 버전을 설치한다.

---

## 1. 파이썬 설치

[python.org](https://www.python.org/downloads/windows/)에서 **3.12 이상 3.14 이하** 버전을 받는다. 이 책은 3.14에서 검증했다.

설치 프로그램을 실행할 때 **첫 화면의 `Add python.exe to PATH`를 반드시 체크한다.** 이것을 빠뜨리면 명령 프롬프트에서 `python`을 찾지 못한다.

> 마이크로소프트 스토어의 파이썬은 권한 문제로 가상 환경이나 패키지 설치가 꼬이는 경우가 있어 권하지 않는다.

설치를 마쳤으면 확인한다.

```powershell
python --version
```

## 2. 저장소 받기

[Git for Windows](https://git-scm.com/download/win)를 설치한 뒤 예제 저장소를 받는다.

```powershell
git clone https://github.com/crapas/dl-pytorch.git
cd dl-pytorch
```

## 3. 가상 환경 만들기

```powershell
python -m venv .venv
.venv\Scripts\activate
```

파워셸에서 `이 시스템에서 스크립트를 실행할 수 없으므로`라는 오류가 나면 실행 정책을 현재 사용자에 한해 풀어 준다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

활성화에 성공하면 프롬프트 앞에 `(.venv)`가 붙는다. 이후의 모든 명령은 이 상태에서 실행한다.

## 4. 패키지 설치

### NVIDIA 그래픽 카드가 없는 경우

```powershell
pip install -r requirements.txt
```

윈도우용 기본 파이토치 휠은 CPU 전용이라 따로 지정할 것이 없다.

### NVIDIA 그래픽 카드가 있는 경우

먼저 드라이버가 지원하는 CUDA 버전을 확인한다.

```powershell
nvidia-smi
```

출력 오른쪽 위 `CUDA Version:` 값에 맞춰 설치 명령을 고른다.

| 드라이버의 CUDA 버전 | 설치 명령의 인덱스 |
|---|---|
| 13.2 이상 | `cu132` |
| 13.0 이상 | `cu130` |
| 12.6 이상 | `cu126` |

```powershell
pip install -r requirements.txt
pip install --upgrade --force-reinstall torch torchvision ^
    --index-url https://download.pytorch.org/whl/cu132
```

`requirements.txt`를 먼저 설치한 다음 파이토치만 CUDA 버전으로 덮어쓰는 순서다. 명령 프롬프트가 아니라 파워셸을 쓴다면 줄바꿈 기호 `^`를 backtick(`` ` ``)으로 바꾼다.

> CUDA 툴킷을 따로 설치할 필요는 없다. 파이토치 휠에 필요한 CUDA 라이브러리가 들어 있다. 준비해야 할 것은 NVIDIA 드라이버뿐이다.

## 5. 설치 확인

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- CPU 환경이면 `2.12.0 False`
- CUDA 환경이면 `2.12.0+cu132 True`

CUDA를 설치했는데 `False`가 나오면 드라이버 버전과 휠 버전이 맞지 않는 경우가 대부분이다. 한 단계 낮은 인덱스(`cu130`, `cu126`)로 다시 설치해 본다.

## 6. 노트북 실행

[VS Code](https://code.visualstudio.com/)를 설치하고 확장 두 가지를 더한다.

- Python
- Jupyter

VS Code로 `dl-pytorch` 폴더를 연 다음 `code_examples/ch01/01-01_example.ipynb`를 연다. 오른쪽 위에서 커널을 고를 때 **방금 만든 `.venv`를 선택한다.** 그다음 셀을 위에서 아래로 실행한다.

## 7. 알아 둘 것

### 한글 그래프

노트북은 `common.set_korean_plot_env()`로 맑은 고딕을 사용한다. 윈도우에는 기본으로 설치되어 있어 따로 할 일이 없다.

### 경로 길이 제한

저장소를 `C:\Users\사용자\Documents\...` 처럼 깊은 곳에 두면 일부 패키지 설치가 260자 경로 제한에 걸릴 수 있다. `C:\dev\dl-pytorch` 처럼 짧은 경로를 권한다.

### 12-3절(QLoRA)

4비트 양자화를 담당하는 `bitsandbytes`는 윈도우 휠을 제공하지만 조합에 따라 동작하지 않는 경우가 있다. 이 절만큼은 [구글 코랩](google-colab.md)의 GPU 런타임이나 [WSL2 + CUDA](linux-cuda.md) 환경을 권한다.

### CPU 환경의 한계

1~7장은 CPU만으로 충분하다. 8장부터는 학습 시간이 길어지고, 12-3절은 CUDA가 없으면 동작하지 않는다. 저장소 최상위 [`README.md`](../../README.md)의 'CUDA / GPU 필요 여부' 표에 장별 기준이 정리되어 있다.
