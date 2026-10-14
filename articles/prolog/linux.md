# 리눅스에서 실습 환경 준비하기

> AI가 자동 생성한 후, 저자가 검토, 수정한 파일임을 밝혀 둡니다.

> **대상** 리눅스를 사용하는 독자. **우분투 24.04**를 기준으로 설명한다.
>
> 다른 배포판을 사용한다면 패키지를 설치하는 명령만 그 배포판의 것으로 바꾸면 된다. 나머지 과정은 같다.
>
> 윈도우에서 WSL2로 리눅스를 사용한다면 이 문서 대신 [windows_wsl2.md](windows_wsl2.md)를 기준으로 실습을 준비해야 한다.

전체 과정은 이렇다.

1. 파이썬 확인과 설치
2. VS Code와 확장 설치
3. 예제 저장소 내려받기
4. 가상 환경 만들기
5. 파이토치와 라이브러리 설치
6. 한글 글꼴 설치
7. 노트북 실행

NVIDIA 그래픽 카드를 사용하려면 **8절 'NVIDIA 그래픽 카드 사용하기'** 를 함께 본다.

---

## 1. 파이썬 확인과 설치

### 버전 확인

우분투 24.04에는 파이썬 3.12가 기본으로 들어 있다. 이 책이 지원하는 범위(**3.12 이상 3.14 이하**) 안이므로 그대로 사용해도 된다.

```bash
python3 --version
```

### 필요한 도구 설치

패키지 목록을 갱신하고 가상 환경과 패키지 설치에 필요한 도구를 설치한다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev git build-essential
```

각각 이런 역할을 한다.

| 패키지 | 역할 |
|---|---|
| `python3-venv` | 가상 환경을 만드는 도구 |
| `python3-pip` | 파이썬 패키지를 설치하는 도구 |
| `python3-dev` | 파이썬의 헤더 파일 |
| `build-essential` | C 코드를 컴파일하는 도구 모음 |
| `git` | 예제 저장소를 내려받는 데 사용 |

> **`python3-dev`와 `build-essential`을 꼭 함께 설치하자.** 지금 당장은 필요 없어 보인다. 그런데 최신 파이토치는 일부 연산을 실행하는 시점에 C 코드로 컴파일한다. 이때 이 패키지들이 필요하다. 없으면 평범해 보이는 코드가 컴파일 오류로 멈추는 일이 생기는데, 원인을 짐작하기 어려운 종류의 문제라 미리 설치해 두는 편이 낫다.

### 더 높은 버전의 파이썬을 사용하려면

기본 파이썬으로도 충분하지만 더 높은 버전을 사용하고 싶다면 **deadsnakes**를 추가한다. 우분투가 공식으로 제공하지 않는 파이썬 버전을 모아 둔 저장소다.

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.14 python3.14-venv python3.14-dev
```

설치한 뒤에는 `python3` 대신 `python3.14`로 실행한다.

> **기본 파이썬을 바꾸지는 말자.** 우분투의 여러 시스템 도구가 기본 파이썬(`/usr/bin/python3`)을 사용한다. 심볼릭 링크를 바꾸거나 기본 버전을 교체하면 시스템이 망가질 수 있다. 가상 환경을 만들 때 원하는 버전을 지정하는 것으로 충분하다.

## 2. VS Code와 확장 설치

[VS Code 내려받기 페이지](https://code.visualstudio.com/)에서 리눅스용 패키지를 받는다. 우분투라면 `.deb` 파일을 받아 설치한다.

```bash
sudo apt install -y ./code_*.deb
```

또는 스냅으로 설치할 수도 있다.

```bash
sudo snap install code --classic
```

VS Code를 실행한 뒤 확장을 설치한다.

1. 왼쪽 세로 막대에서 **확장**(Extensions) 아이콘을 누른다. 네모 블록 네 개가 모인 모양이고, 단축키는 `Ctrl+Shift+X`다.
2. 검색창에 확장 이름을 입력하고, **마이크로소프트(Microsoft)가 배포한 것**을 골라 설치한다. 이름이 같은 확장이 여러 개 보일 수 있으므로 배포자를 확인한다.

설치할 확장은 다음과 같다.

| 확장 | 역할 |
|---|---|
| **Python** | 파이썬 문법 강조, 자동 완성, 가상 환경 인식 |
| **Jupyter** | 노트북 파일을 열고 셀을 실행 |

이 확장이 있어야 노트북 파일을 열고 셀을 실행할 수 있다. 메뉴를 한국어로 보고 싶다면 **Korean Language Pack**도 함께 설치한다.

## 3. 예제 저장소 내려받기

### Git 확인

예제 저장소는 **Git**으로 받는다. Git은 소스 코드의 변경 이력을 관리하는 도구로, 저장소를 통째로 받고 나중에 갱신하는 데 사용한다.

1절에서 `git`을 함께 설치했으므로 이미 준비되어 있을 것이다. 확인한다.

```bash
git --version
```

`git version 2.x.x`처럼 출력되면 된다. 명령을 찾을 수 없다면 설치한다.

```bash
sudo apt install -y git
```

### 내려받기

```bash
cd ~
git clone https://github.com/crapas/dl-pytorch.git
cd dl-pytorch
```

`~`는 자신의 홈 폴더를 뜻한다.

나중에 예제를 갱신하려면 저장소 폴더에서 실행한다.

```bash
git pull
```

## 4. 가상 환경 만들기

### 가상 환경이란

파이썬 프로젝트마다 필요한 라이브러리와 버전이 다르다. 모든 라이브러리를 한곳에 설치하면 프로젝트끼리 버전이 충돌한다. A 프로젝트는 파이토치 2.12가 필요한데 B 프로젝트는 2.8이 필요한 상황이 생기는 것이다.

**가상 환경**(virtual environment)은 프로젝트마다 라이브러리를 담는 상자를 따로 두는 장치다. 이 책의 라이브러리를 이 책 전용 상자에 담아 두면 다른 프로젝트에 영향을 주지도 받지도 않는다. 상자를 통째로 지우고 다시 만들 수도 있어서 설치가 꼬였을 때 되돌리기도 쉽다.

최근 배포판에서는 가상 환경이 선택이 아니다. 시스템 파이썬에 `pip`로 패키지를 설치하려고 하면 이런 오류와 함께 막힌다.

```
error: externally-managed-environment
```

시스템이 사용하는 파이썬을 보호하기 위한 장치다. **이 오류를 우회하지 말고 가상 환경을 만들자.**

### 만들고 활성화하기

저장소 폴더(`~/dl-pytorch`)에서 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3.14를 따로 설치했다면 첫 줄을 `python3.14 -m venv .venv`로 바꾼다.

성공하면 프롬프트 앞에 `(.venv)`가 붙는다.

```
(.venv) 사용자@컴퓨터:~/dl-pytorch$
```

**이후의 모든 명령은 이 상태에서 실행한다.**

### 가상 환경 사용법

| 하고 싶은 일 | 명령 |
|---|---|
| 활성화 | `source .venv/bin/activate` |
| 비활성화 | `deactivate` |
| 지금 활성화되어 있는지 확인 | 프롬프트 앞의 `(.venv)` 표시를 본다 |
| 어느 파이썬을 사용하는지 확인 | `which python` |
| 통째로 지우기 | `deactivate` 후 `rm -rf .venv` |

**터미널을 새로 열 때마다 활성화해야 한다.** 활성화하지 않은 채로 설치하면 위의 오류가 나거나 엉뚱한 곳에 설치된다. 잘 되던 것이 갑자기 안 될 때 가장 먼저 확인할 것이 `(.venv)` 표시다.

> 위의 예제에서 .venv는 가상환경의 이름인 동시에, 가상 환경 실행에 필요한 파일이 저장되는 디렉터리의 이름이다. 경우에 따라서는 같은 프로젝트에 서로 다른 조합의 라이브러리를 사용해야 할 때도 있다. 예를 들어 CUDA 적용 환경과 CUDA 비적용 환경을 함께 사용한다면, 같은 디렉터리에서 이름이 다른 두 개의 가상 환경을 만들어 사용하면 된다. .venv와 .venv_cuda 두 개의 가상 환경을 만든 경우, .venv에 설치된 라이브러리는 .venv_cuda에 영향을 주지 않으며, 사용할 수도 없고, 그 반대 방향도 성립한다.

## 5. 파이토치와 라이브러리 설치

### NVIDIA 그래픽 카드가 없는 경우

**리눅스에서는 CPU 버전의 파이토치를 설치 경로를 지정해 따로 설치해야 한다.** 윈도우와 다른 점이다.

```bash
pip install -c requirements.txt torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

그냥 `pip install torch`라고 하면 CUDA가 포함된 설치본이 받아진다. 리눅스용 기본 설치본이 그렇게 되어 있기 때문이다. 그래픽 카드가 없다면 사용하지도 않을 라이브러리를 몇 GB씩 내려받게 되므로, **CPU용 경로를 지정해 저장 공간을 아낄 수 있다.**

`--index-url` 뒤의 주소가 "어디서 받을지"를 지정하는 부분이고, 끝의 `cpu`가 CPU 전용이라는 뜻이다.

`requirements.txt`는 저장소에 들어 있는 파일로, 필요한 라이브러리와 버전이 적혀 있다. 그래서 **저장소 폴더(`~/dl-pytorch`)에서 실행해야 한다.**

**`-c requirements.txt`를 빼먹지 말자.** `-c`는 설치할 것을 정하지 않고 **버전의 범위만 제한**하는 옵션이다. 이것이 없으면 CPU용 인덱스에서 가장 최신 버전의 파이토치가 설치된다. 그런데 `requirements.txt`는 `torch~=2.12.0`처럼 버전을 제한하고 있어서, 다음 명령이 방금 설치한 파이토치를 지우고 **PyPI의 기본 설치본(CUDA 포함)으로 바꿔 버린다.** 오류 없이 끝나기 때문에 알아차리기 어렵다. `-c requirements.txt`를 붙이면 두 명령이 같은 버전 조건을 따르므로 이런 일이 생기지 않는다.

### NVIDIA 그래픽 카드가 있는 경우

**8절을 먼저 읽자.** 설치 방법에 차이가 있다.

### 설치 확인

```bash
python -c "import torch; print(torch.__version__)"
```

`2.12.1+cpu`처럼 버전의 **끝이 `+cpu`로 출력되면** 성공이다. `+cpu`가 없이 `2.12.1`처럼 출력된다면 PyPI의 기본 설치본(CUDA 포함)이 설치된 것이다. 가상 환경을 지우고(`deactivate` 후 `rm -rf .venv`) 가상 환경 만들기부터 다시 하되, 파이토치 설치 명령에 `-c requirements.txt`를 빠뜨리지 않았는지 확인한다.

### pip 사용법

책을 진행하면서 절마다 추가 라이브러리를 설치하게 되므로 몇 가지만 알아 두자.

| 하고 싶은 일 | 명령 |
|---|---|
| 설치 | `pip install 패키지이름` |
| 여러 개 한 번에 | `pip install 패키지1 패키지2` |
| 파일에 적힌 목록대로 | `pip install -r requirements.txt` |
| 버전을 지정해 설치 | `pip install "패키지이름>=1.2"` |
| 설치된 목록 보기 | `pip list` |
| 특정 패키지 정보 보기 | `pip show 패키지이름` |
| 삭제 | `pip uninstall 패키지이름` |

**버전을 지정할 때는 따옴표로 감싸자.** 따옴표가 없으면 `>` 기호를 셸이 '출력을 파일로 보내라'는 뜻으로 해석해, 버전 조건은 무시되고 이상한 이름의 빈 파일이 만들어진다.

## 6. 한글 글꼴 설치

리눅스에는 한글 글꼴이 기본으로 들어 있지 않은 경우가 많다. 노트북이 그래프에 한글 폰트(나눔고딕 폰트)를 사용하므로 설치해야 한다.

```bash
sudo apt install -y fonts-nanum
fc-cache -fv
rm -rf ~/.cache/matplotlib
```

마지막 줄은 matplotlib가 저장해 둔 글꼴 목록을 지우는 명령이다. matplotlib는 시작할 때마다 글꼴을 찾는 대신 목록을 저장해 두고 재사용하므로, 이것을 지워야 새로 설치한 글꼴을 찾는다.

이 과정을 건너뛰면 첫 셀에서 글꼴을 찾을 수 없다는 안내를 만난다. **설치한 뒤에는 노트북 커널을 다시 시작해야 반영된다.**

## 7. 노트북 실행

저장소 폴더에서 VS Code를 연다.

```bash
cd ~/dl-pytorch
code .
```

1. 왼쪽 탐색기에서 `code_examples` → `ch01` → `01-01_example.ipynb`를 연다.
2. **오른쪽 위의 '커널 선택'을 누르고 `.venv`를 고른다.**
3. 첫 셀부터 `Shift+Enter`로 차례대로 실행한다.

**2번이 가장 중요하다.** 커널은 노트북의 코드를 실제로 실행하는 파이썬을 가리킨다. 가상 환경이 아닌 다른 파이썬을 고르면 설치한 라이브러리를 찾지 못해 `ModuleNotFoundError`가 난다.

### 터미널에서 주피터랩을 사용하려면

VS Code 대신 브라우저에서 노트북을 열 수도 있다.

```bash
pip install jupyterlab
jupyter lab
```

브라우저가 저절로 열린다. 열리지 않으면 터미널에 출력된 주소를 복사해 브라우저 주소창에 붙여 넣는다.

---

## 8. NVIDIA 그래픽 카드 사용하기

> 그래픽 카드가 없거나 사용하지 않을 생각이라면 이 절을 건너뛰어도 된다.

### CUDA란

**CUDA**는 NVIDIA가 만든 기술로, 그래픽 카드로 일반적인 계산을 수행할 수 있게 해 준다. 딥러닝의 학습은 같은 계산을 엄청나게 많이 되풀이하는 작업이라 CPU보다 그래픽 카드가 훨씬 빠르다. 모델에 따라 수십 배 차이가 난다.

파이토치는 CUDA를 사용하는 설치본을 따로 제공한다. 이것을 설치하면 코드를 거의 바꾸지 않고도 그래픽 카드로 학습할 수 있다.

### 준비할 것은 드라이버뿐이다

인터넷을 찾아보면 'CUDA 툴킷을 설치하라'는 안내가 많다. 그런데 **파이토치를 사용할 목적이라면 CUDA 툴킷을 따로 설치할 필요가 없다.** 필요한 CUDA 라이브러리가 파이토치 설치본 안에 들어 있기 때문이다.

준비할 것은 **NVIDIA 그래픽 드라이버** 하나다.

### 드라이버 확인과 설치

```bash
nvidia-smi
```

표가 출력되면 준비된 것이다. **표 오른쪽 위의 `CUDA Version:` 값을 적어 두자.** 다음 단계에서 사용한다.

이 값은 "이 드라이버가 지원하는 CUDA의 최대 버전"을 뜻한다. CUDA가 설치되어 있다는 뜻이 아니다.

명령을 찾을 수 없다면 드라이버를 설치한다. 우분투에서는 적절한 드라이버를 자동으로 골라 주는 명령이 있다.

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

재시작한 뒤 `nvidia-smi`를 다시 실행해 확인한다.

> 그래픽 카드가 최신이라면 배포판이 제공하는 드라이버가 지원하지 못할 수 있다. 그때는 [NVIDIA 드라이버 페이지](https://www.nvidia.com/Download/index.aspx)에서 직접 받아 설치한다.

### 파이토치 설치

파이토치는 CUDA 버전별로 설치본을 따로 제공한다. **드라이버가 지원하는 CUDA 버전보다 높은 것을 설치하면 동작하지 않으므로**, 앞에서 확인한 값보다 같거나 낮은 것을 고른다.

설치 명령은 [파이토치 공식 설치 안내 페이지](https://pytorch.org/get-started/locally/)에서 자신의 환경을 고르면 그 자리에서 만들어 준다. 운영체제, 설치 도구(Pip), 언어(Python), 연산 플랫폼(CUDA 버전)을 차례로 고르면 된다. **CUDA 버전은 `nvidia-smi`에서 확인한 값 이하의 것 중 가장 높은 것**을 고른다.

명령의 모양은 이렇다.

```bash
pip install --upgrade -c requirements.txt torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

끝의 `cu132`가 CUDA 13.2용이라는 뜻이고, 이 숫자가 버전에 따라 달라진다. 저장소 폴더(`~/dl-pytorch`)에서 실행한다.

**`-c requirements.txt`를 빼먹지 말자.** 5절에서 설명한 대로 `requirements.txt`의 버전 조건을 파이토치 설치에도 똑같이 적용하는 옵션이다. 이것이 없으면 인덱스의 최신 버전이 설치되고, 다음 명령인 `pip install -r requirements.txt`가 그 버전을 지우고 **PyPI의 기본 설치본으로 바꿔 버릴 수 있다.** `--upgrade`는 이미 CPU용 파이토치가 설치되어 있어도 CUDA용으로 교체하도록 한다.

파이토치를 먼저, `requirements.txt`를 나중에 설치한다. 두 명령의 버전 조건이 같으므로 두 번째 명령은 방금 설치한 CUDA용 파이토치를 그대로 둔다.

### 확인

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`2.12.1+cu132 True`처럼 버전의 **끝이 `+cu132`이고** 두 번째 값이 **`True`** 로 나오면 성공이다. 끝에 `+cu132`가 없거나 `+cpu`라면 CUDA용이 아닌 파이토치가 설치된 것이므로 위의 설치 명령을 다시 실행한다. 그래픽 카드 이름까지 확인하려면 이렇게 한다.

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**`False`가 나온다면** 드라이버가 지원하는 것보다 높은 CUDA 버전을 설치한 경우가 대부분이다. 한 단계 낮은 버전으로 다시 설치한다.

```bash
pip install --force-reinstall -c requirements.txt torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130
```

`--force-reinstall`은 이미 설치된 것을 지우고 새로 설치하라는 뜻이다.

### 그래픽 메모리가 모자랄 때

12장과 13장은 모델이 커서 그래픽 메모리를 많이 사용한다. `CUDA out of memory` 오류를 만나면 이렇게 대처한다.

- **배치 크기를 줄인다.** 노트북 위쪽의 `BATCH_SIZE`를 절반으로 낮춰 본다. 가장 효과가 확실하다.
- **다른 노트북의 커널을 종료한다.** 이전에 실행한 노트북이 메모리를 쥐고 있는 경우가 흔하다. `nvidia-smi`로 메모리를 차지한 프로세스를 확인할 수 있다.
- **그래픽 카드를 사용하는 다른 프로그램을 닫는다.**
- 그래도 모자라면 CPU를 사용하거나, 해당 절만 [구글 코랩](google-colab.md)에서 실습을 고려해 보자.

---

## 9. 자주 만나는 문제

| 증상 | 원인과 해결 |
|---|---|
| `ensurepip is not available` 오류로 가상 환경이 만들어지지 않는다 | 실행한 파이썬 버전에 맞는 `venv` 패키지가 없다. 오류 메시지가 알려 주는 `python3.XX-venv`를 설치한다. 기본 파이썬은 `python3-venv`, deadsnakes로 설치한 3.14는 `python3.14-venv`다(1절) |
| `error: externally-managed-environment` | 가상 환경을 활성화하지 않았다. `source .venv/bin/activate` |
| `ModuleNotFoundError: No module named 'torch'` | 커널로 `.venv`를 고르지 않았거나, 가상 환경 밖에서 설치했다 |
| `ModuleNotFoundError: No module named 'code_reference'` | VS Code에서 저장소 최상위가 아닌 다른 폴더를 열었다 |
| 커널 목록에 `.venv`가 없다 | VS Code를 다시 시작한다. 그래도 없으면 폴더를 제대로 열었는지 확인한다 |
| 그래프의 한글이 네모로 보인다 | 글꼴 설치 후 커널을 다시 시작하지 않았다(6절) |
| 글꼴을 찾을 수 없다는 안내가 나온다 | 6절대로 설치하고 커널을 다시 시작한다 |
| 파이토치 설치 용량이 몇 GB다 | CPU용 경로를 지정하지 않았거나, 파이토치 설치 명령에 `-c requirements.txt`를 빠뜨렸다(5절) |
| `torch.__version__`에 `+cpu`(CPU) 또는 `+cu132`(CUDA)가 붙지 않는다 | `pip install -r requirements.txt`가 파이토치를 PyPI 기본 설치본으로 바꿨다. 파이토치 설치 명령에 `-c requirements.txt`를 붙여 다시 설치한다(5절, 8절) |
| `torch.cuda.is_available()`이 `False` | 드라이버보다 높은 CUDA 버전을 설치했다. 8절 참고 |
| 컴파일 오류와 함께 학습이 멈춘다 | `python3-dev`와 `build-essential`을 설치한다(1절) |
| `torchvision` 설치가 거부되거나 `__annotate__` 오류가 난다 | 파이썬 **3.14.1**의 알려진 문제다. 3.14.2 이상으로 올린다 |

## 10. CPU만으로 진행할 때

**모든 예제는 CPU만 있어도 실행할 수 있다.** 시간이 더 오래 걸릴 뿐이다.

다만 8장 이후의 예제는 실행 시간이 꽤 길 수 있으므로 GPU를 사용할 수 있다면 GPU 사용을 권하며, 그렇지 않다면 [구글 코랩](google-colab.md)에서 실행하는 방법도 고려해 보자.
