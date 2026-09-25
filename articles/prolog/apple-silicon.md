# 실습 환경 구성 — 애플 실리콘 맥

> **대상**: M1 이후 칩을 쓰는 맥. 인텔 맥도 설치 과정은 같으나 가속을 받지 못한다.

맥에는 NVIDIA 그래픽 카드가 없어 CUDA를 쓸 수 없다. 대신 애플이 제공하는 **MPS**(Metal Performance Shaders) 백엔드로 내장 GPU를 쓴다. 이 책의 노트북은 `common.get_device()`가 알아서 MPS를 골라 주므로 코드를 고칠 일은 없다.

---

## 1. 준비

터미널에서 명령행 개발 도구를 설치한다. `git`이 함께 들어온다.

```bash
xcode-select --install
```

패키지 관리자로 [홈브루](https://brew.sh)를 쓰면 편하다.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 2. 파이썬 설치

이 책은 파이썬 **3.12 이상 3.14 이하**를 지원하며 3.14에서 검증했다.

```bash
brew install python@3.14
python3.14 --version
```

> 맥에 기본으로 깔린 `/usr/bin/python3`은 시스템이 쓰는 것이라 건드리지 않는 편이 좋다. 홈브루로 따로 설치해 쓰자.

## 3. 저장소 받기와 가상 환경

```bash
git clone https://github.com/crapas/dl-pytorch.git
cd dl-pytorch

python3.14 -m venv .venv
source .venv/bin/activate
```

프롬프트 앞에 `(.venv)`가 붙으면 성공이다.

## 4. 패키지 설치

```bash
pip install -r requirements.txt
```

인덱스를 따로 지정할 것이 없다. 애플 실리콘용 기본 휠에 MPS 지원이 들어 있다.

## 5. 설치 확인

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

`2.12.0 True`가 나오면 된다. 실제로 가속이 걸리는지 확인하려면 이렇게 한다.

```bash
python -c "
import torch
x = torch.randn(1000, 1000, device='mps')
print((x @ x).shape, x.device)"
```

## 6. 노트북 실행

[VS Code](https://code.visualstudio.com/)에 Python 확장과 Jupyter 확장을 더한 뒤 `dl-pytorch` 폴더를 연다. `code_examples/ch01/01-01_example.ipynb`를 열고 커널로 **`.venv`를 선택한 다음** 셀을 위에서 아래로 실행한다.

5장 이후 노트북은 `common.get_device()`로 장치를 고르는데, 맥에서는 다음이 출력된다.

```
현재의 하드웨어 가속기 장치: mps
```

## 7. 한글 그래프

노트북은 애플 고딕을 사용한다. 맥에 기본으로 설치되어 있어 따로 할 일이 없다.

## 8. MPS에서 알아 둘 것

### 속도

MPS는 CPU보다 확실히 빠르지만 같은 값대의 NVIDIA 그래픽 카드에는 미치지 못한다. 1~7장은 쾌적하고, 8장 이후는 기다림이 길어진다. 통합 메모리를 쓰는 구조라 메모리가 넉넉한 모델(32GB 이상)일수록 큰 모델을 다루기 수월하다.

### 12-3절(QLoRA)은 동작하지 않는다

4비트 양자화를 담당하는 `bitsandbytes`가 CUDA 전용이라 MPS에서는 동작하지 않는다. 이 절은 [구글 코랩](google-colab.md)의 GPU 런타임에서 실행하자.

### 12장과 13장의 자료형

12장과 13장 일부 노트북은 모델을 `torch.float16`으로 불러온다. MPS에서 문제가 생기면 `torch.float32`로 바꾸면 동작한다. 메모리를 두 배로 쓰고 느려지지만 결과는 같다.

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
```

### 아직 구현되지 않은 연산

MPS 백엔드는 CUDA만큼 모든 연산을 지원하지는 않는다. `NotImplementedError`와 함께 `aten::...` 연산 이름이 나오면 그 연산만 CPU로 돌리도록 환경 변수를 켜면 된다.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

VS Code에서 쓰려면 노트북 첫 셀에 넣는다.

```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### 결과의 미세한 차이

이 책의 노트북 출력은 CUDA 환경에서 만든 것이다. MPS에서는 부동소수점 연산 순서가 달라 손실값이나 정확도의 마지막 자리가 다를 수 있다. 학습 경향과 정확도 수준 같은 결론은 같게 유지된다.

## 9. 장별 기준

장별로 어디까지 되는지는 저장소 최상위 [`README.md`](../../README.md)의 'CUDA / GPU 필요 여부' 표에 정리되어 있다. 요약하면 12-3절만 실행할 수 없고, 나머지는 시간이 더 걸릴 뿐 모두 동작한다.
