# 자동·외부 다운로드 디렉터리

본 디렉터리는 노트북 실행 중 **자동으로 받거나 사용자가 별도로 받아 두는 외부 데이터**의 저장소다. 저자 제공 예제 데이터(`../data/`)와 분리되어 있으며, **본 디렉터리의 내용물은 깃 추적에서 제외**된다(이 README 만 추적).

## 디렉터리 분리 원칙

| 디렉터리 | 내용 | 깃 추적 |
|---------|------|--------|
| `data/`      | 저자가 제공하는 작은 예제 파일 (CSV, 이미지, 텍스트) | ✓ 추적 |
| `downloads/` | torchvision · HuggingFace 등이 자동으로 받는 데이터, 또는 사용자가 직접 받는 대형 데이터 | ✗ 제외 |

## 노트북에서의 경로 사용

```python
# 자동 다운로드 데이터 (예: MNIST, CIFAR-10)
datasets.MNIST(root='../../downloads', train=True, download=True, transform=...)

# 저자 제공 예제 데이터 (예: 회오리 CSV)
with open('../../data/ch3_spiral_data.csv') as f:
    ...
```

`download=True` 옵션으로 자동으로 받는 데이터는 모두 본 디렉터리에 떨어지도록 `root='../../downloads'` 를 지정한다.

## 자동 다운로드되는 데이터

다음은 노트북에서 `download=True` 로 자동으로 받는 데이터다. 첫 실행 시 한 번만 받아 본 디렉터리에 캐시되며, 두 번째 실행부터는 캐시를 사용한다.

| 데이터 | 사용 장 | 받는 라이브러리 | 대략 용량 |
|--------|---------|---------------|---------|
| MNIST | 1·3·4·5장 등 | `torchvision.datasets.MNIST` | ~50MB |
| FashionMNIST | 11장 | `torchvision.datasets.FashionMNIST` | ~40MB |
| CIFAR-10 | 5·8장 | `torchvision.datasets.CIFAR10` | ~170MB |

## 수동 다운로드가 필요한 데이터

다음은 라이브러리가 자동으로 받지 않으므로, 본 디렉터리에 직접 받아 둔다.

| 데이터 | 사용 장 | URL | 본 디렉터리 안 위치 |
|--------|---------|-----|-------------------|
| GloVe 6B 100d | 7장 | https://nlp.stanford.edu/data/glove.6B.zip | `glove.6B.100d.txt` (압축 해제 후 그대로) |

GloVe 는 압축 파일 안에 여러 차원의 임베딩이 들어 있다. 본 책은 100차원(`glove.6B.100d.txt`)만 사용한다.

## HuggingFace 캐시는 본 디렉터리 밖에 저장된다

12·13장에서 사용하는 HuggingFace `transformers`, `datasets`, `peft` 등은 **시스템 사용자 캐시(`~/.cache/huggingface/`)** 에 모델·데이터셋을 캐시한다. 본 디렉터리와는 무관하며, 환경 변수 `HF_HOME` 으로 위치를 바꿀 수 있다.

```bash
# 본 디렉터리로 HF 캐시를 옮기고 싶을 때 (선택)
export HF_HOME=$(pwd)/downloads/huggingface
```
