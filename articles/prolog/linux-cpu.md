# 실습 환경 구성 — 리눅스 (CPU)

> **대상**: NVIDIA 그래픽 카드가 없는 리눅스 환경. 우분투 24.04를 기준으로 설명한다.
> NVIDIA 그래픽 카드가 있다면 [리눅스 + CUDA](linux-cuda.md)를 보자.
> 윈도우에서 WSL2로 리눅스를 쓰려면 이 문서 끝의 [부록](#부록-windows-wsl2에서-구성하기)을 먼저 읽자.

---

## 1. 파이썬 설치

이 책은 파이썬 **3.12 이상 3.14 이하**를 지원하며 3.14에서 검증했다. 우분투 24.04의 기본 파이썬은 3.12이므로 그대로 써도 된다.

```bash
python3 --version
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git
```

3.14를 쓰고 싶다면 deadsnakes 저장소를 더한다.

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.14 python3.14-venv python3.14-dev
```

> `python3.x-dev`를 함께 설치하는 것을 권한다. 지금 당장은 필요 없지만, 최신 파이토치에서 일부 연산이 실행 시점에 C 확장을 컴파일하는데 이때 이 패키지의 헤더가 필요하다.

## 2. 저장소 받기

```bash
git clone https://github.com/crapas/dl-pytorch.git
cd dl-pytorch
```

## 3. 가상 환경 만들기

```bash
python3 -m venv .venv          # 3.14를 설치했다면 python3.14 -m venv .venv
source .venv/bin/activate
```

프롬프트 앞에 `(.venv)`가 붙으면 성공이다. 이후의 모든 명령은 이 상태에서 실행한다.

## 4. 패키지 설치

리눅스에서는 **반드시 CPU 인덱스를 지정한다.**

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

인덱스를 지정하지 않으면 PyPI의 기본 휠이 설치되는데, 리눅스용 기본 휠은 CUDA를 포함한 판이라 쓰지도 않을 라이브러리를 3GB 가까이 내려받는다. CPU 인덱스를 쓰면 그만큼을 아낀다.

## 5. 설치 확인

```bash
python -c "import torch; print(torch.__version__)"
```

`2.12.0+cpu`가 나오면 된다.

## 6. 한글 폰트 설치

노트북이 그래프에 한글을 쓰므로 나눔 고딕이 필요하다.

```bash
sudo apt-get install -y fonts-nanum
fc-cache -fv
rm -rf ~/.cache/matplotlib
```

이 과정을 건너뛰면 첫 셀의 `common.set_korean_plot_env()`에서 `FileNotFoundError`와 함께 설치 명령을 안내받는다. 설치한 뒤에는 **노트북 커널을 다시 시작해야** 반영된다.

## 7. 노트북 실행

[VS Code](https://code.visualstudio.com/)에 Python 확장과 Jupyter 확장을 더한 뒤 `dl-pytorch` 폴더를 연다. `code_examples/ch01/01-01_example.ipynb`를 열고 오른쪽 위에서 커널로 **방금 만든 `.venv`를 선택한 다음** 셀을 위에서 아래로 실행한다.

터미널에서 주피터랩을 쓰려면 이렇게 한다.

```bash
pip install jupyterlab
jupyter lab
```

## 8. CPU 환경의 한계

1~7장은 CPU만으로 충분하다. 8장부터는 학습 시간이 길어지며, 특히 다음 절은 유의해야 한다.

| 절 | CPU에서 |
|---|---|
| 8~11장 | 동작하지만 느리다 |
| 12-1, 13장 | 동작하지만 매우 느리다 |
| 12-2 (미세 조정) | 권하지 않는다 |
| **12-3 (QLoRA)** | **동작하지 않는다.** `bitsandbytes` 4비트 양자화는 CUDA 전용이다 |

12-3절은 [구글 코랩](google-colab.md)의 GPU 런타임에서 실행하면 된다. 장별 기준은 저장소 최상위 [`README.md`](../../README.md)의 'CUDA / GPU 필요 여부' 표에 정리되어 있다.

---

## 부록: 윈도우 WSL2에서 구성하기

윈도우에서 리눅스 환경을 쓰고 싶다면 WSL2가 가장 간편하다. 설치한 뒤에는 위의 1~8절을 그대로 따르면 된다.

### 설치

관리자 권한 파워셸에서 다음 한 줄이면 WSL2와 우분투가 함께 설치된다.

```powershell
wsl --install -d Ubuntu-24.04
```

재부팅한 뒤 우분투 창이 열리면 사용자 이름과 비밀번호를 정한다. 이후로는 시작 메뉴의 `Ubuntu`로 들어가면 된다.

### 저장소를 둘 위치

**반드시 리눅스 파일 시스템 안에 두자.**

```bash
cd ~                                     # /home/사용자 아래
git clone https://github.com/crapas/dl-pytorch.git
```

`/mnt/c/...` 아래(윈도우 디스크)에 두면 파일 접근이 윈도우와 리눅스 사이를 오가며 느려진다. 데이터셋을 많이 읽는 학습에서는 차이가 크게 난다.

### VS Code 연결

윈도우 쪽 VS Code에 **WSL 확장**을 설치한다. 그다음 WSL 터미널에서 이렇게 하면 VS Code가 리눅스에 붙은 채로 열린다.

```bash
code .
```

이 상태에서 커널을 고르면 WSL 안의 `.venv`가 목록에 나타난다.

### 알아 둘 것

- 한글 폰트는 WSL 안에 따로 설치해야 한다(6절 그대로).
- WSL2는 기본적으로 호스트 메모리의 절반까지 쓴다. 부족하면 윈도우 사용자 폴더에 `.wslconfig` 파일을 만들어 `memory=16GB` 처럼 늘린다.
- NVIDIA 그래픽 카드가 있다면 CPU 대신 [리눅스 + CUDA](linux-cuda.md) 쪽을 보자. WSL2에서도 CUDA를 쓸 수 있다.
