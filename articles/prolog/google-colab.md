# 구글 코랩에서 예제 노트북 실행하기

> **이 책의 예제 노트북을 코랩에서 돌리는 방법**
> 대상: `code_examples/chNN/NN-NN_example.ipynb`와 `exercises/` 아래의 노트북
> 로컬 환경 설정은 저장소 최상위 [`README.md`](../../README.md)를 참고하자.

GPU가 없는 컴퓨터를 쓰거나, 12-3절의 양자화처럼 **NVIDIA GPU가 반드시 필요한 절**을 실습할 때는 구글 코랩이 좋은 선택지다. 무료 티어로도 이 책의 예제 대부분을 돌릴 수 있다.

다만 **코랩에는 로컬과 다른 점이 하나 있어서, 노트북을 그냥 열면 첫 셀부터 실패한다.** 이 문서는 그 이유와 해결 방법, 그리고 장별로 미리 알아 둘 것을 정리한다.

---

## 1. 왜 그냥 열면 안 되는가

이 책의 모든 노트북은 첫 셀에서 공통 라이브러리를 불러온다.

```python
import sys
sys.path.append('../../')

from code_reference import common
from code_reference import visualize as viz
```

`../../`는 **노트북 파일이 있는 폴더를 기준으로 두 단계 위**, 곧 저장소 최상위를 가리킨다. 로컬에서 VS Code나 JupyterLab으로 열면 작업 디렉터리가 노트북이 있는 폴더(`code_examples/ch08/`)이므로 이 경로가 정확히 맞는다.

**코랩은 다르다. 노트북을 어디서 열든 작업 디렉터리가 항상 `/content`다.**

| | 작업 디렉터리 | `../../`가 가리키는 곳 |
|---|---|---|
| 로컬 | `…/dl-pytorch/code_examples/ch08` | `…/dl-pytorch` ✅ |
| **코랩** | **`/content`** | **`/`** ❌ |

그래서 이런 오류를 만난다.

```
ModuleNotFoundError: No module named 'code_reference'
FileNotFoundError: [Errno 2] No such file or directory: '../../data/ch3_spiral_data.csv'
```

**구글 드라이브에 노트북을 올려 두어도 마찬가지다.** 드라이브의 노트북을 코랩에서 열어도 작업 디렉터리는 여전히 `/content`이지 드라이브 폴더가 아니다.

해결은 간단하다. **저장소를 통째로 받아 놓고, 노트북이 있는 폴더로 이동하면 된다.**

---

## 2. 런타임 고르기 — TPU는 고르지 말 것

노트북을 열었으면 먼저 런타임을 정한다.

**메뉴 → 런타임 → 런타임 유형 변경 → 하드웨어 가속기**

| 선택 | 권장 여부 | 설명 |
|---|---|---|
| **T4 GPU** | **권장** | 무료 티어에서 쓸 수 있다. 9장 이후는 사실상 필수 |
| CPU | 1~7장에 한해 가능 | 느리지만 동작한다. 12-3절은 실행할 수 없다 |
| **TPU** | **고르지 말 것** | 아래 설명 참고 |

### TPU를 고르면 안 되는 이유

`code_reference/common.py`의 `get_device()`는 사용할 장치를 `cuda → mps → xla → cpu` 순으로 고른다. **TPU 런타임에는 `torch_xla`가 설치되어 있어 XLA 장치가 선택된다.**

그런데 이 책의 학습 루프는 전부 일반적인 파이토치 방식이다(`.to(device)`, `loss.backward()`, `optimizer.step()`, 표준 `DataLoader`). **TPU는 연산을 모아 두었다가 한꺼번에 실행하는 지연 실행 방식**이라, `xm.optimizer_step()`이나 `MpDeviceLoader` 같은 XLA 전용 장치를 써야 제대로 돈다.

그래서 TPU 런타임에서 이 책의 노트북을 돌리면 **오류 없이 극도로 느려지거나 멈춘 것처럼 보인다.** 원인을 찾기 어려운 종류의 문제다.

> GPU를 배정받지 못했다면 TPU 대신 **CPU로 두고 기다리는 편**이 낫다.

---

## 3. 표준 부트스트랩 셀

**노트북의 맨 첫 셀로 아래 코드를 넣는다.** 코랩인지 아닌지를 스스로 판단하므로, **로컬에서도 그대로 두고 쓸 수 있다.**

```python
# === 코랩 부트스트랩 (로컬에서는 아무 일도 하지 않는다) ===
import os
import sys

if 'google.colab' in sys.modules:
    CHAPTER = 'ch08'                      # ← 실습할 장으로 바꾼다
    if not os.path.exists('/content/dl-pytorch'):
        !git clone -q --depth 1 https://github.com/crapas/dl-pytorch.git /content/dl-pytorch
    %cd /content/dl-pytorch/code_examples/{CHAPTER}
```

고칠 곳은 **`CHAPTER` 한 줄뿐**이다. 연습 문제 노트북을 돌린다면 `code_examples`를 `exercises`로 바꾼다.

이 셀을 넣으면 그다음 셀부터는 **로컬과 코랩이 완전히 같게 동작한다.** 노트북 본문의 `../../`도, `../../data/...`도 손댈 필요가 없다.

| | 하는 일 |
|---|---|
| 로컬 | `if` 안으로 들어가지 않는다. 기존과 100% 동일 |
| 코랩 | 저장소를 `/content`에 받고 해당 장 폴더로 이동한다 |

### 왜 `/content`에 받는가

구글 드라이브에 받아 두고 싶을 수 있지만 **권하지 않는다.** 드라이브는 네트워크 파일 시스템이라 **작은 파일을 많이 읽고 쓸 때 매우 느리고, 하루 입출력 한도도 있다.**

이 책의 예제는 그런 작업이 많다.

- MNIST와 CIFAR-10은 압축을 풀면 파일이 수천 개다(5장, 8장)
- 허깅페이스 모델 캐시는 수 GB다(12장, 13장)
- 학습 중 체크포인트를 자주 쓴다

`/content`는 코랩 인스턴스의 로컬 디스크라 빠르다. **저장소와 데이터는 `/content`에 두고, 남기고 싶은 결과만 마지막에 드라이브로 복사하는 것**이 정석이다(→ 7절).

---

## 4. 첫 실행에서 만나는 두 가지

### 4.1 한글 폰트 — 한 번은 재시작해야 한다

그림에 한글을 쓰려면 나눔 고딕이 필요한데, 코랩에는 기본으로 깔려 있지 않다. `common.set_korean_plot_env()`가 자동으로 설치를 시도하지만, **설치한 폰트는 지금 실행 중인 세션에 바로 반영되지 않는다.**

그래서 첫 실행에서 이런 메시지를 만난다.

```
FileNotFoundError: 나눔 고딕 폰트를 설치했습니다.
노트북 커널을 다시 시작한 후 다시 시도하세요.
```

**오류가 아니라 안내다.** 시키는 대로 하면 된다.

1. **런타임 → 세션 다시 시작**
2. 첫 셀부터 다시 실행

재시작해도 `/content`에 받아 둔 저장소는 그대로 남아 있으므로, 부트스트랩 셀은 `git clone`을 건너뛰고 `%cd`만 한다. **두 번째 실행부터는 이 일이 없다.**

> 재시작이 번거롭다면 부트스트랩 셀 다음에 아래 한 줄을 미리 실행해 두고 한 번에 재시작해도 된다.
> ```python
> !apt-get install -qq fonts-nanum && fc-cache -f
> ```

### 4.2 추가 라이브러리 설치

코랩에는 파이토치와 torchvision이 이미 깔려 있다. **1장부터 11장까지는 따로 설치할 것이 없다.**

12장과 13장은 허깅페이스 생태계를 쓰므로 설치가 필요하다.

```python
# 12-1, 12-2, 13장
!pip install -q transformers datasets accelerate

# 12-3 (양자화, QLoRA)
!pip install -q transformers accelerate peft bitsandbytes

# 12-4 (검색 증강 생성)
!pip install -q transformers sentence-transformers groq
```

설치 후 **세션 재시작을 요구하는 경우**가 있다. 코랩이 안내하면 재시작하고 첫 셀부터 다시 실행한다.

> **버전 지정에 따옴표를 쓰자.** `!pip install accelerate>=1.1.0`처럼 쓰면 셸이 `>`를 **출력 리다이렉션**으로 해석해 버전 조건이 무시되고 `=1.1.0`이라는 빈 파일이 생긴다. `!pip install 'accelerate>=1.1.0'`처럼 따옴표로 감싸야 한다.

---

## 5. 장별로 알아 둘 것

| 장 | 가속기 | 내려받는 것 | 참고 |
|---|---|---|---|
| 1 ~ 4 | CPU로 충분 | 저장소의 CSV 파일 | 가볍다 |
| 5 | GPU 권장 | MNIST, CIFAR-10 | `download/`에 받는다 |
| 6, 7 | GPU 권장 | MNIST, 오즈의 마법사 텍스트 | |
| 8 | **GPU 권장** | CIFAR-10, ResNet-50·VGG16 가중치 | 8-4절은 **캐글 계정**이 필요하다(아래) |
| 9, 10 | **GPU 권장** | 없음(데이터를 코드로 생성) | 10-1절은 200 에포크라 시간이 걸린다 |
| 11 | **GPU 권장** | Fashion-MNIST | DCGAN 20 에포크 |
| 12-1, 12-2 | **GPU 필요** | Bllossom-3B(약 6GB), KoBART | 디스크 여유 확인 |
| 12-3 | **GPU 필수(CUDA)** | Bllossom-3B | `bitsandbytes`는 CPU·TPU에서 못 쓴다 |
| 12-4 | GPU 권장 | KR-SBERT | **Groq API 키**가 필요하다(아래) |
| 13-1 | GPU 권장 | CLIP, GPT-2, Flickr8k | 비교적 가볍다 |
| 13-2 | **GPU 필요** | **BLIP-2 약 15GB** | 디스크와 시간에 주의 |

### 8-4절 — 캐글 데이터셋

흉부 X선 데이터셋은 캐글에서 받는다. 캐글 계정에서 API 토큰(`kaggle.json`)을 발급받아 두자. `kagglehub`가 처음 실행될 때 인증을 요구한다.

### 12-4절 — Groq API 키

본문 예제는 Groq의 LLM API를 쓴다. [Groq 콘솔](https://console.groq.com)에서 무료 계정을 만들고 키를 발급받자. 코랩에서는 **왼쪽 열쇠 모양 아이콘(보안 비밀)**에 저장하는 방법이 안전하다.

```python
from google.colab import userdata
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
```

키 없이도 실습할 수 있도록 **예제 노트북에 로컬 LLM으로 대체하는 분기**를 넣어 두었다. 다만 3B 모델이라 본문의 8B 모델보다 답변 품질이 떨어지며, 특히 **"문서에 없으면 모른다고 답하라"는 지시를 잘 지키지 못한다.**

### 13-2절 — BLIP-2 15GB

코랩 무료 티어의 디스크로도 받을 수 있지만 **시간이 꽤 걸린다.** 세션이 끊기면 다시 받아야 하므로, 시간 여유가 있을 때 실행하자.

---

## 6. 세션이 끊기면 무엇이 사라지는가

코랩 세션은 **일정 시간 쓰지 않으면 끊기고, 끊기면 `/content` 아래가 모두 지워진다.** 무료 티어는 연속 사용 시간에도 제한이 있다.

| 사라지는 것 | 대응 |
|---|---|
| `/content/dl-pytorch` (저장소) | 부트스트랩 셀이 다시 받는다. 신경 쓸 것 없다 |
| 내려받은 데이터셋과 모델 캐시 | 다시 받아야 한다. 13-2절처럼 큰 모델은 시간이 든다 |
| 학습한 모델 파라미터 | **미리 저장해 두지 않으면 잃는다** |
| 노트북의 셀 출력 | 노트북 자체는 드라이브에 저장된다 |

**긴 학습을 돌릴 때는 브라우저 탭을 열어 두자.** 탭을 닫으면 유휴로 판정되어 더 빨리 끊긴다.

---

## 7. 결과를 드라이브에 남기기

학습 결과를 보존하려면 드라이브를 마운트해 **필요한 것만 복사한다.**

```python
from google.colab import drive
drive.mount('/content/drive')

# 학습은 빠른 로컬 디스크에서, 보관만 드라이브로
!mkdir -p /content/drive/MyDrive/dl-pytorch-backup
!cp -r /content/dl-pytorch/checkpoint /content/drive/MyDrive/dl-pytorch-backup/
```

**작업 디렉터리 자체를 드라이브로 옮기지는 말자.** 3절에서 설명한 대로 느려진다. 마운트는 **마지막에 결과를 복사할 때만** 쓰는 것이 좋다.

허깅페이스 캐시(`HF_HOME`)와 torchvision 다운로드 폴더도 **드라이브로 바꾸지 않는 것**이 좋다. 기본값 그대로 두면 `/content`와 `~/.cache`를 쓴다.

---

## 8. 자주 만나는 오류

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: No module named 'code_reference'` | 부트스트랩 셀이 없거나 `%cd`가 실행되지 않았다 | 3절의 셀을 첫 셀로 넣고 처음부터 실행 |
| `FileNotFoundError: '../../data/...'` | 같은 원인 | 같음 |
| `FileNotFoundError: 나눔 고딕 폰트를 설치했습니다…` | 폰트 설치 직후 | **세션 다시 시작** 후 처음부터 실행(4.1절) |
| 그림의 한글이 □로 보인다 | 재시작 전에 그림을 그렸다 | 같음 |
| 학습이 오류 없이 멈춘 듯 느리다 | **TPU 런타임**을 골랐다 | GPU 또는 CPU로 변경(2절) |
| `CUDA out of memory` | 이전 모델이 메모리에 남아 있다 | `del model` 후 `torch.cuda.empty_cache()`, 안 되면 세션 재시작 |
| `RuntimeError: ... bitsandbytes ... CUDA` | CPU/TPU 런타임에서 12-3절을 실행했다 | **T4 GPU 런타임**으로 변경 |
| `=1.1.0`이라는 이상한 파일이 생겼다 | `!pip install accelerate>=1.1.0`의 `>` 리다이렉션 | 따옴표로 감싼다(4.2절) |
| 디스크 부족 | 큰 모델을 여러 개 받았다 | 세션 재시작으로 `/content` 비우기 |
| `DataLoader worker ... Bus error` | `num_workers`를 올렸다 | 코랩에서는 **0~2**로 제한 |

> 이 책의 예제는 모든 `DataLoader`가 `num_workers=0`이라 기본 상태에서는 이 문제가 없다. 속도를 위해 값을 올릴 때만 주의하면 된다.

---

## 9. 요약

**처음 시작할 때 이 순서로 하면 된다.**

1. 깃허브에서 노트북을 연다 — 코랩의 **파일 → 노트북 열기 → GitHub** 탭에 `crapas/dl-pytorch`를 입력하면 바로 열 수 있다
2. **런타임 → 런타임 유형 변경 → T4 GPU** (TPU는 고르지 않는다)
3. 맨 위에 셀을 하나 추가하고 **3절의 부트스트랩 셀**을 붙여 넣는다. `CHAPTER`만 바꾼다
4. 위에서부터 실행한다
5. 폰트 안내가 나오면 **세션 다시 시작** 후 처음부터 다시 실행한다
6. 남기고 싶은 결과는 **드라이브로 복사**한다(7절)

**한 번만 기억하면 되는 것은 하나다.** 코랩의 작업 디렉터리는 언제나 `/content`이므로, **저장소를 받아 놓고 노트북이 있는 폴더로 들어가는 일**만 해 주면 나머지는 로컬과 똑같이 동작한다.
