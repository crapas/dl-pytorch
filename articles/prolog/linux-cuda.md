# 실습 환경 구성 — 리눅스 (CUDA)

> **대상**: NVIDIA 그래픽 카드가 있는 리눅스 환경. 우분투 24.04를 기준으로 설명한다.
> 그래픽 카드가 없다면 [리눅스 + CPU](linux-cpu.md)를 보자.
> 윈도우에서 WSL2로 구성하려면 이 문서 끝의 [부록](#부록-windows-wsl2에서-구성하기)을 먼저 읽자.

이 책의 노트북 출력은 이 환경(우분투 + CUDA, 파이썬 3.14, 파이토치 2.12)에서 만들었다. 책의 모든 절이 제약 없이 동작한다.

---

## 1. NVIDIA 드라이버 확인

CUDA에 필요한 것은 **드라이버뿐이다.** CUDA 툴킷은 따로 설치하지 않아도 된다. 파이토치 휠 안에 필요한 CUDA 라이브러리가 들어 있다.

```bash
nvidia-smi
```

표가 출력되면 준비된 것이다. 오른쪽 위 `CUDA Version:` 값을 기억해 두자. 4절에서 쓴다.

명령을 찾지 못하면 드라이버를 설치한다.

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

> CUDA 13.x 휠을 쓰려면 드라이버 580 이상을 권한다.

## 2. 파이썬 설치

이 책은 파이썬 **3.12 이상 3.14 이하**를 지원하며 3.14에서 검증했다. 우분투 24.04의 기본 파이썬은 3.12다.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-dev git
```

3.14를 쓰려면 deadsnakes 저장소를 더한다.

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.14 python3.14-venv python3.14-dev
```

> **`python3.x-dev`를 반드시 함께 설치하자.** CPU 환경보다 중요하다. 최신 파이토치는 일부 GPU 연산을 실행 시점에 컴파일하는데, 그 과정에서 이 패키지의 헤더와 `gcc`가 필요하다. 없으면 평범한 `loss.backward()`가 컴파일 오류로 멈추는 일이 생긴다. `gcc`는 `build-essential`로 설치한다.

```bash
sudo apt-get install -y build-essential
```

## 3. 저장소 받기와 가상 환경

```bash
git clone https://github.com/crapas/dl-pytorch.git
cd dl-pytorch

python3 -m venv .venv          # 3.14를 설치했다면 python3.14 -m venv .venv
source .venv/bin/activate
```

## 4. 패키지 설치

1절에서 확인한 드라이버의 CUDA 버전에 맞춰 인덱스를 고른다.

| 드라이버의 CUDA 버전 | 인덱스 |
|---|---|
| 13.2 이상 | `cu132` |
| 13.0 이상 | `cu130` |
| 12.6 이상 | `cu126` |

```bash
pip install --index-url https://download.pytorch.org/whl/cu132 torch torchvision
pip install -r requirements.txt
```

파이토치를 먼저 CUDA 인덱스에서 설치하고, 그다음 나머지 의존성을 설치하는 순서다. 반대로 하면 `requirements.txt`가 CPU 휠을 끌어와 덮어쓸 수 있다.

## 5. 설치 확인

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`2.12.0+cu132 True NVIDIA GeForce ...` 처럼 나오면 된다.

`False`가 나온다면 드라이버가 지원하는 CUDA 버전보다 높은 휠을 설치한 경우가 대부분이다. 한 단계 낮은 인덱스로 다시 설치한다.

```bash
pip install --upgrade --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu130 torch torchvision
```

## 6. 한글 폰트 설치

```bash
sudo apt-get install -y fonts-nanum
fc-cache -fv
rm -rf ~/.cache/matplotlib
```

설치한 뒤에는 **노트북 커널을 다시 시작해야** 반영된다.

## 7. 장별 추가 라이브러리

8장 이후에는 절마다 필요한 라이브러리가 다르다. 각 노트북 첫 셀 주석에 설치 명령이 적혀 있으며, 미리 설치해 두려면 다음과 같다.

```bash
pip install timm kagglehub                      # 8-4
pip install transformers datasets accelerate    # 12-1, 12-2, 13장
pip install bitsandbytes peft                   # 12-3 (QLoRA)
pip install sentence-transformers groq          # 12-4 (RAG)
```

## 8. 노트북 실행

[VS Code](https://code.visualstudio.com/)에 Python 확장과 Jupyter 확장을 더한 뒤 `dl-pytorch` 폴더를 연다. `code_examples/ch01/01-01_example.ipynb`를 열고 커널로 **`.venv`를 선택한 다음** 셀을 위에서 아래로 실행한다.

## 9. 그래픽 메모리가 모자랄 때

12장과 13장은 모델이 커서 그래픽 메모리를 많이 쓴다. `CUDA out of memory`를 만나면 이렇게 대처한다.

- 배치 크기를 줄인다.
- 12-3절처럼 4비트 양자화를 쓰는 절을 먼저 실행해 본다.
- `nvidia-smi`로 다른 프로세스가 메모리를 쥐고 있는지 확인한다. 이전 노트북 커널이 살아 있는 경우가 흔하다.

---

## 부록: 윈도우 WSL2에서 구성하기

WSL2에서도 CUDA를 그대로 쓸 수 있다. 설치한 뒤에는 위의 2~9절을 따르면 된다.

### 설치

관리자 권한 파워셸에서 실행한다.

```powershell
wsl --install -d Ubuntu-24.04
```

재부팅한 뒤 우분투 창에서 사용자 이름과 비밀번호를 정한다.

### 드라이버는 윈도우 쪽에만 설치한다

여기가 가장 헷갈리는 대목이다.

- **윈도우**: NVIDIA 드라이버를 설치한다. NVIDIA 앱이나 [공식 드라이버](https://www.nvidia.com/Download/index.aspx)에서 받는다.
- **WSL 안**: 드라이버를 설치하지 **않는다.** 리눅스용 드라이버를 WSL에 설치하면 오히려 망가진다.

WSL 터미널에서 다음이 정상 출력되면 준비가 끝난 것이다.

```bash
nvidia-smi
```

### 저장소를 둘 위치

**반드시 리눅스 파일 시스템 안에 두자.**

```bash
cd ~
git clone https://github.com/crapas/dl-pytorch.git
```

`/mnt/c/...` 아래에 두면 데이터셋을 읽을 때마다 윈도우와 리눅스 사이를 오가 학습이 눈에 띄게 느려진다.

### VS Code 연결

윈도우 쪽 VS Code에 **WSL 확장**을 설치한 뒤, WSL 터미널에서 `code .`을 실행한다. VS Code가 리눅스에 붙은 채로 열리고, 커널 목록에 WSL 안의 `.venv`가 나타난다.

### 알아 둘 것

- WSL2는 기본적으로 호스트 메모리의 절반까지 쓴다. 부족하면 윈도우 사용자 폴더에 `.wslconfig`를 만들어 `memory=16GB` 처럼 늘린다.
- 그래픽 메모리는 윈도우와 나눠 쓴다. 학습 중에는 게임이나 영상 편집처럼 GPU를 쓰는 프로그램을 닫아 두자.
- 한글 폰트는 WSL 안에 따로 설치해야 한다(6절 그대로).
