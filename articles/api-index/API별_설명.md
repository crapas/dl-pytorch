# 파이토치 API 찾아보기 - 접두사별

- <직접 구현하는 딥러닝 with 파이토치> 본문과 예제 코드에 등장하는 클래스, 함수, 메서드를 패키지별로 모은 표
- 각 항목에 대한 간단한 설명을 제시
- 이름 기준으로 정리한 목록은 [ABC순 목록](README.md)에 수록
- 바로가기
  - torch
    - [torch.* (최상위 함수와 자료형)](#torch--최상위-함수와-자료형)
    - [torch.nn.* (계층과 손실 함수 클래스)](#torchnn--계층과-손실-함수-클래스)
    - [torch.optim.*](#torchoptim)
    - [torch.cuda.* / torch.backends.* / torch.xpu.* / torch.linalg.*](#torchcuda--torchbackends--torchxpu--torchlinalg)
    - [torch.nn.functional.* (별칭 F)](#torchnnfunctional--별칭-f)
    - [torch.nn.init.*](#torchnninit)
    - [torch.nn.utils.*](#torchnnutils)
    - [torch.nn.Module 메서드](#torchnnmodule-메서드)
    - [torch.nn.Sequential 메서드](#torchnnsequential-메서드)
    - [torch.utils.data.*](#torchutilsdata)
    - [torch.Tensor 메서드](#torchtensor-메서드)
    - [torch.Tensor 제자리 연산 메서드](#torchtensor-제자리-연산-메서드)
  - torchvision
    - [torchvision.datasets.*](#torchvisiondatasets)
    - [torchvision.transforms.*](#torchvisiontransforms)
    - [torchvision.models.*](#torchvisionmodels)
  - 서드파티 라이브러리
    - [torchinfo](#torchinfo)
    - [timm](#timm)
    - [transformers (허깅페이스)](#transformers-허깅페이스)
    - [datasets (허깅페이스)](#datasets-허깅페이스)
    - [sentence_transformers](#sentence_transformers)
    - [groq](#groq)
    - [PIL (Pillow)](#pil-pillow)
    - [numpy](#numpy)
    - [requests](#requests)
  - [파이썬 표준 라이브러리](#파이썬-표준-라이브러리)

---

## torch.*  (최상위 함수와 자료형)

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `torch.Size` | 1-1 | 텐서의 형태를 담는 튜플형 객체. `shape` 속성과 `size()` 메서드가 반환한다. |
|  | `torch.no_grad` | 1-2, 5-3 | 연산 그래프 기록을 끄는 컨텍스트 매니저. 평가와 추론에서 쓴다. `@torch.no_grad()` 데코레이터로도 쓴다. |
|  | `torch.device` | 2-3, 5-3 | 연산을 수행할 장치를 나타내는 객체. `torch.device('cuda')`처럼 만든다. |
|  | `torch.Generator` | 4-2 | 난수 발생기 객체. `random_split()`처럼 무작위성이 있는 함수에 넘겨 시드를 고정한다. |
| 함수 | `torch.arange` | 1-1 | 시작값부터 끝값 직전까지 일정 간격으로 채운 1차원 텐서를 만든다. |
|  | `torch.cat` | 1-1, 10-1 | 지정한 축을 따라 여러 텐서를 이어 붙인다. 차원 수는 늘지 않는다. |
|  | `torch.linspace` | 1-1 | 시작값부터 끝값까지를 지정한 개수로 균등 분할한 1차원 텐서를 만든다. |
|  | `torch.ones` | 1-1 | 1로 채운 텐서를 만든다. |
|  | `torch.rand` | 1-1 | 0 이상 1 미만의 균등 분포 난수로 채운 텐서를 만든다. |
|  | `torch.randint` | 1-1 | 지정한 범위의 정수 난수로 채운 텐서를 만든다. |
|  | `torch.randn` | 1-1, 2-2 | 표준정규분포 난수로 채운 텐서를 만든다. 파라미터 초기화에 쓴다. |
|  | `torch.stack` | 1-1 | 새 차원을 만들어 여러 텐서를 쌓는다. `cat()`과 달리 차원 수가 하나 늘어난다. |
|  | `torch.tensor` | 1-1 | 리스트나 넘파이 배열로 텐서를 만드는 가장 기본적인 함수. |
|  | `torch.zeros` | 1-1 | 0으로 채운 텐서를 만든다. |
|  | `torch.max` | 1-2 | 최댓값을 반환한다. `dim`을 주면 최댓값과 그 인덱스를 함께 반환한다. |
|  | `torch.min` | 1-2 | 최솟값을 반환한다. `dim`을 주면 최솟값과 그 인덱스를 함께 반환한다. |
|  | `torch.manual_seed` | 1-3, 4-2 | 난수 발생 시드를 고정해 실행할 때마다 같은 난수가 나오게 한다. |
|  | `torch.mean` | 1-3 | 평균을 계산한다. 실수형 텐서에만 쓸 수 있다. |
|  | `torch.matmul` | 2-2, 9-3 | 행렬곱. `@` 연산자와 같다. |
|  | `torch.sigmoid` | 2-2 | 요소별 시그모이드 함수. 값을 0과 1 사이로 눌러 준다. |
|  | `torch.load` | 4-1 | `torch.save()`로 저장한 객체를 불러온다. |
|  | `torch.save` | 4-1 | 모델 파라미터나 텐서를 파일로 저장한다. |
|  | `torch.randn_like` | 8-3 | 인자로 받은 텐서와 같은 형태의 표준정규분포 난수 텐서를 만든다. |
|  | `torch.argmax` | 8-4, 10-3 | 가장 큰 값의 인덱스를 반환한다. 분류 모델의 로짓에서 예측 클래스를 뽑을 때 쓴다. |
|  | `torch.topk` | 8-4 | 가장 큰 k개의 값과 그 인덱스를 반환한다. |
|  | `torch.bmm` | 9-3 | 배치 단위 행렬곱. `(B, n, m)`과 `(B, m, p)`를 곱해 `(B, n, p)`를 만든다. |
|  | `torch.softmax` | 9-3 | 지정한 축의 값을 합이 1인 확률 분포로 바꾼다. |
|  | `torch.tanh` | 9-3 | 요소별 하이퍼볼릭 탄젠트. 값을 -1과 1 사이로 눌러 준다. |
|  | `torch.multinomial` | 10-3 | 주어진 확률 분포에서 표본을 뽑는다. 문장 생성의 확률적 샘플링에 쓴다. |
|  | `torch.exp` | 11-1 | 요소별 지수 함수. 자연상수 e의 거듭제곱을 계산한다. |
|  | `torch.sum` | 11-1 | 요소의 합을 구한다. `dim`으로 축을 지정할 수 있다. |
|  | `torch.full` | 13-1 | 지정한 값 하나로 전부 채운 텐서를 만든다. |
| 자료형 | `torch.bool` | 1-1 | 불리언 자료형. 마스크 텐서에 쓴다. |
|  | `torch.double` | 1-1 | `torch.float64`의 별칭. |
|  | `torch.float` | 1-1, 1-2 | `torch.float32`의 별칭. |
|  | `torch.float32` | 1-1 | 32비트 실수 자료형. 파이토치의 기본 실수형이다. |
|  | `torch.float64` | 1-1 | 64비트 실수 자료형. 배정밀도다. |
|  | `torch.long` | 1-1 | `torch.int64`의 별칭. 분류 문제의 정답 텐서에 쓴다. |
|  | `torch.int32` | 4-2 | 32비트 정수 자료형. |
|  | `torch.int64` | 4-2 | 64비트 정수 자료형. 파이토치의 기본 정수형이다. |
|  | `torch.bfloat16` | 12장 도입 | 16비트 실수 자료형. float16보다 지수부가 넓어 값의 범위가 크다. |
|  | `torch.float16` | 12장 도입 | 16비트 실수 자료형. 메모리를 절반으로 줄이는 반정밀도다. |
|  | `torch.int8` | 12장 도입 | 8비트 정수 자료형. 양자화에 쓴다. |

## torch.nn.*  (계층과 손실 함수 클래스)

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `nn.Sigmoid` | 2-2, 2-3 | 시그모이드 활성화 계층. 출력을 0과 1 사이로 눌러 준다. |
|  | `nn.AdaptiveMaxPool2d` | 2-3 | 출력 크기를 지정하면 그에 맞춰 커널을 정하는 최대 풀링 계층. |
|  | `nn.Linear` | 2-3 | 선형 계층. 가중치를 곱하고 편향을 더하는, 퍼셉트론의 뉴런에 해당하는 계층이다. |
|  | `nn.Module` | 2-3 | 모든 모델과 계층의 부모 클래스. 이를 상속해 모델 클래스를 정의한다. |
|  | `nn.MSELoss` | 2-3, 3-3 | 평균제곱오차 손실. 회귀 분석의 표준 손실 함수다. |
|  | `nn.Unflatten` | 2-3 | 평탄화된 텐서를 지정한 형태로 되돌린다. |
|  | `nn.CrossEntropyLoss` | 3-3 | 교차 엔트로피 손실. 다중 클래스 분류의 표준 손실 함수이며 소프트맥스를 내부에서 적용한다. |
|  | `nn.ReLU` | 3-3 | ReLU 활성화 계층. 음수를 0으로 만들어 기울기 소실을 완화한다. |
|  | `nn.Sequential` | 3-3 | 계층을 나열한 순서대로 데이터를 흘려보내는 컨테이너 클래스. |
|  | `nn.Softmax` | 3-3 | 소프트맥스 활성화 계층. 출력을 합이 1인 확률 분포로 바꾼다. |
|  | `nn.Flatten` | 4-3 | 텐서를 평탄화해 완전 연결 계층에 넣을 수 있는 형태로 바꾼다. |
|  | `nn.AvgPool2d` | 5-2 | 지정한 창 안의 평균으로 특징 지도를 줄이는 평균 풀링 계층. |
|  | `nn.Conv2d` | 5-2 | 2차원 합성곱 계층. 필터로 이미지에서 특징을 뽑는다. |
|  | `nn.MaxPool2d` | 5-2, 8-1 | 지정한 창 안의 최댓값으로 특징 지도를 줄이는 최대 풀링 계층. |
|  | `nn.Dropout` | 5-3 | 학습 중 일부 뉴런의 출력을 무작위로 0으로 만드는 규제 계층. |
|  | `nn.RNN` | 6-1, 6-2 | 단순 순환 신경망 계층. 이전 시점의 숨겨진 상태를 입력과 함께 받는다. |
|  | `nn.GRU` | 6-3 | 게이트 순환 유닛 계층. LSTM보다 가벼운 순환 신경망 계층이다. |
|  | `nn.LSTM` | 6-3 | 장단기 기억 계층. 게이트로 장기 의존성 문제를 완화한 순환 신경망 계층이다. |
|  | `nn.Embedding` | 7-2 | 토큰 고유 번호를 의미가 담긴 실수 벡터로 바꾸는 임베딩 계층. |
|  | `nn.ConvTranspose2d` | 7-3, 11-1, 11-2 | 전치 합성곱 계층. 특징 지도의 크기를 키운다. 생성 모델의 디코더에 쓴다. |
|  | `nn.L1Loss` | 7-3 | 평균절대오차 손실. 이상치가 많은 회귀 분석에 쓴다. |
|  | `nn.Upsample` | 7-3 | 보간으로 특징 지도의 크기를 키운다. |
|  | `nn.BatchNorm1d` | 8-2 | 선형 계층이 출력하는 텐서에 쓰는 배치 정규화 계층. |
|  | `nn.BatchNorm2d` | 8-2 | 합성곱 계층이 출력하는 특징 지도에 쓰는 배치 정규화 계층. |
|  | `nn.AdaptiveAvgPool2d` | 8-3 | 출력 크기를 지정하면 그에 맞춰 커널을 정하는 평균 풀링 계층. |
|  | `nn.Identity` | 8-3, 13-1 | 입력을 그대로 내보내는 계층. 사전 학습 모델의 분류기를 떼어낼 때 자리 채우기로 쓴다. |
|  | `nn.NLLLoss` | 9-2 | 음의 로그 가능도 손실. 로그 소프트맥스를 거친 출력과 함께 쓴다. |
|  | `nn.MultiheadAttention` | 9-3 | 멀티 헤드 어텐션 계층. |
|  | `nn.Tanh` | 10-1 | 하이퍼볼릭 탄젠트 활성화 계층. 출력을 -1과 1 사이로 눌러 준다. |
|  | `nn.Transformer` | 10-1 | 인코더와 디코더를 모두 갖춘 트랜스포머 모델 클래스. |
|  | `nn.TransformerDecoder` | 10-1 | 트랜스포머 디코더. 디코더 계층을 여러 겹 쌓는다. |
|  | `nn.TransformerEncoder` | 10-1, 10-2 | 트랜스포머 인코더. 인코더 계층을 여러 겹 쌓는다. |
|  | `nn.TransformerEncoderLayer` | 10-2 | 트랜스포머 인코더 한 겹. 셀프 어텐션과 피드포워드로 이루어진다. |
|  | `nn.BCELoss` | 11-2 | 이진 교차 엔트로피 손실. 정답이 두 가지인 이진 분류에 쓴다. |
|  | `nn.LeakyReLU` | 11-2 | 음수 입력에도 작은 기울기를 남기는 ReLU 변형. GAN의 판별자에 자주 쓴다. |
|  | `nn.GELU` | 13-1 | GELU 활성화 계층. 트랜스포머 계열에서 ReLU 대신 널리 쓴다. |
|  | `nn.LayerNorm` | 13-1 | 계층 정규화. 샘플 하나 안에서 정규화하며 트랜스포머 계열에서 쓴다. |
|  | `nn.Parameter` | 13-2 | 학습 대상 파라미터임을 표시하는 텐서 래퍼. 모델의 속성으로 두면 자동으로 등록된다. |

## torch.optim.*

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `optim.Adam` | 2-3, 3-2 | 적응형 모멘트 추정법 옵티마이저. 파라미터마다 학습률을 조정하며 가장 널리 쓰인다. |
|  | `optim.AdamW` | 2-3 | Adam의 가중치 감쇠 방식을 개선한 옵티마이저. 트랜스포머 계열 학습에 주로 쓴다. |
|  | `optim.SGD` | 2-3, 3-2 | 확률적 경사하강법 옵티마이저. 모든 파라미터에 같은 학습률을 적용한다. |
| 메서드 | `Optimizer.step` | 2-3 | 계산된 기울기로 파라미터를 갱신한다. |
|  | `Optimizer.zero_grad` | 2-3 | 이전 단계에서 누적된 기울기를 초기화한다. 역전파 전에 부른다. |

## torch.cuda.* / torch.backends.* / torch.xpu.* / torch.linalg.*

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 함수 | `torch.backends.mps.is_available` | 5-3 | 애플 실리콘의 MPS 장치를 쓸 수 있는지 확인한다. |
|  | `torch.cuda.is_available` | 5-3 | NVIDIA CUDA 장치를 쓸 수 있는지 확인한다. |
|  | `torch.xpu.is_available` | 5-3 | 인텔 XPU 장치를 쓸 수 있는지 확인한다. |
|  | `torch.linalg.norm` | 8-2 | 벡터나 행렬의 노름을 계산한다. |
|  | `torch.cuda.memory_allocated` | 12-3 | 현재 CUDA 장치에 할당된 메모리 크기를 바이트 단위로 반환한다. |

## torch.nn.functional.*  (별칭 F)

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 함수 | `F.one_hot` | 3-3 | 클래스 번호 텐서를 원-핫 인코딩 텐서로 바꾼다. |
|  | `F.sigmoid` | 3-3 | 시그모이드 함수의 함수형 API. |
|  | `F.softmax` | 3-3 | 소프트맥스 함수의 함수형 API. |
|  | `F.log_softmax` | 10-3 | 소프트맥스에 로그를 취한 값을 한 번에 계산한다. `nn.NLLLoss`와 함께 쓴다. |
|  | `F.normalize` | 12-4 | 벡터를 단위 길이로 정규화한다. 코사인 유사도 계산에 쓴다. |

## torch.nn.init.*

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 함수 | `nn.init.constant_` | 11-2 | 파라미터를 지정한 값 하나로 채우는 초기화 함수. |
|  | `nn.init.normal_` | 11-2 | 파라미터를 지정한 평균과 표준편차의 정규 분포로 초기화한다. |

## torch.nn.utils.*

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 함수 | `nn.utils.clip_grad_norm_` | 6-3 | 기울기의 노름이 기준값을 넘으면 잘라 기울기 폭주를 막는다. |
|  | `nn.utils.rnn.pack_padded_sequence` | 9-2 | 패딩이 섞인 배치를 순환 계층이 패딩을 건너뛰도록 묶어 준다. |
|  | `nn.utils.rnn.pad_packed_sequence` | 9-2 | `pack_padded_sequence()`로 묶은 결과를 다시 패딩된 텐서로 되돌린다. |
|  | `nn.utils.rnn.pad_sequence` | 9-2 | 길이가 다른 시퀀스들을 가장 긴 것에 맞춰 패딩해 하나의 텐서로 만든다. |

## torch.nn.Module 메서드

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 메서드 | `Module.to` | 1-1, 5-3 | 모델을 지정한 장치로 옮긴다. 텐서와 달리 제자리 연산이라 반환값을 받지 않아도 된다. |
|  | `Module.train` | 1-3, 2-3 | 모델을 학습 모드로 전환한다. 드롭아웃과 배치 정규화의 동작이 달라진다. |
|  | `Module.eval` | 2-3 | 모델을 평가 모드로 전환한다. 평가와 추론 전에 반드시 부른다. |
|  | `Module.forward` | 2-3 | 모델의 데이터 흐름을 정의하는 메서드. 직접 부르지 않고 `model(x)`로 호출한다. |
|  | `Module.parameters` | 2-3 | 모델에 포함된 모든 학습 파라미터를 제너레이터로 반환한다. 옵티마이저에 넘긴다. |
|  | `Module.zero_grad` | 2-3 | 모델 파라미터의 기울기를 초기화한다. |
|  | `Module.add_module` | 3-3 | 이름을 붙여 하위 계층을 추가한다. |
|  | `Module.load_state_dict` | 4-1, 8-4 | `state_dict()`로 얻은 파라미터를 모델에 되돌려 넣는다. |
|  | `Module.state_dict` | 4-1 | 모델의 파라미터를 담은 `OrderedDict`를 반환한다. 저장에 쓴다. |
|  | `Module.apply` | 11-2 | 콜백 함수를 자기 자신과 모든 하위 계층에 적용한다. 파라미터 초기화에 쓴다. |

## torch.nn.Sequential 메서드

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 메서드 | `Sequential.append` | 3-3 | `nn.Sequential`의 끝에 계층을 추가한다. |
|  | `Sequential.insert` | 3-3 | `nn.Sequential`의 지정한 위치에 계층을 끼워 넣는다. |
|  | `Sequential.pop` | 3-3 | `nn.Sequential`에서 지정한 위치의 계층을 빼내 반환한다. |

## torch.utils.data.*

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `ConcatDataset` | 4-2 | 여러 데이터셋을 순서대로 이어 붙여 하나로 만든다. |
|  | `DataLoader` | 4-2 | 데이터셋에서 미니배치를 꺼내 주는 반복 가능 객체. 배치 크기와 섞기를 지정한다. |
|  | `Dataset` | 4-2 | 데이터셋의 부모 클래스. `__len__()`과 `__getitem__()`을 구현해 사용자 데이터셋을 만든다. |
| 함수 | `random_split` | 4-2 | 데이터셋을 지정한 크기로 무작위 분리한다. 훈련과 검증 분리에 쓴다. |

## torch.Tensor 메서드

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 메서드 | `Tensor.clamp` | 1-1 | 요소의 값을 지정한 범위 안으로 자른다. |
|  | `Tensor.clone` | 1-1 | 값을 복사한 새 텐서를 만든다. 슬라이싱의 참조 공유를 끊을 때 쓴다. |
|  | `Tensor.dim` | 1-1 | 텐서의 차원 수를 반환한다. |
|  | `Tensor.flatten` | 1-1 | 지정한 차원 구간을 이어 붙여 평탄화한다. |
|  | `Tensor.numel` | 1-1 | 텐서에 담긴 요소의 총 개수를 반환한다. |
|  | `Tensor.permute` | 1-1 | 차원의 순서를 통째로 바꾼다. 모든 차원을 지정해야 한다. |
|  | `Tensor.reshape` | 1-1, 9-1 | 요소 수를 유지한 채 텐서의 형태를 바꾼다. |
|  | `Tensor.size` | 1-1, 11-1 | 텐서의 형태를 `torch.Size`로 반환한다. 인자로 차원을 지정할 수 있다. |
|  | `Tensor.squeeze` | 1-1 | 크기가 1인 차원을 없앤다. |
|  | `Tensor.t` | 1-1 | 2차원 텐서의 행과 열을 바꾼다. 전치행렬을 만든다. |
|  | `Tensor.to` | 1-1, 5-3 | 자료형이나 장치를 바꾼 새 텐서를 반환한다. 제자리 연산이 아니므로 반환값을 받아야 한다. |
|  | `Tensor.transpose` | 1-1 | 지정한 두 차원을 맞바꾼다. |
|  | `Tensor.unsqueeze` | 1-1, 9-3 | 지정한 위치에 크기가 1인 차원을 추가한다. |
|  | `Tensor.view` | 1-1 | 형태를 바꾼다. `reshape()`과 달리 메모리가 연속인 텐서에만 쓸 수 있다. |
|  | `Tensor.backward` | 1-2 | 스칼라 손실에서 역전파를 실행해 각 파라미터의 기울기를 구한다. |
|  | `Tensor.detach` | 1-2 | 값은 공유하되 연산 그래프에서 떼어 낸 새 텐서를 반환한다. |
|  | `Tensor.detach_` | 1-2 | `detach()`의 제자리 연산 버전. |
|  | `Tensor.item` | 1-2 | 요소가 하나인 텐서에서 파이썬 숫자를 꺼낸다. 손실값을 기록할 때 쓴다. |
|  | `Tensor.max` | 1-2, 3-3 | 최댓값을 반환한다. |
|  | `Tensor.min` | 1-2 | 최솟값을 반환한다. |
|  | `Tensor.numpy` | 1-2 | 텐서를 넘파이 배열로 바꾼다. 서드파티 라이브러리에 넘길 때 쓴다. |
|  | `Tensor.requires_grad_` | 1-2 | 텐서를 자동 미분 대상으로 지정한다. |
|  | `Tensor.std` | 1-2 | 표준편차를 계산한다. |
|  | `Tensor.sum` | 1-2, 11-1 | 요소의 합을 구한다. |
|  | `Tensor.tolist` | 1-2 | 텐서를 파이썬 리스트로 바꾼다. |
|  | `Tensor.mean` | 1-3 | 평균을 계산한다. |
|  | `Tensor.argmax` | 3-3 | 가장 큰 값의 인덱스를 반환한다. |
|  | `Tensor.float` | 3-3 | 요소의 자료형을 `torch.float32`로 바꾼다. |
|  | `Tensor.long` | 7-2 | 요소의 자료형을 `torch.int64`로 바꾼다. |
|  | `Tensor.norm` | 8-2 | 텐서의 노름을 계산한다. |
|  | `Tensor.topk` | 8-4 | 가장 큰 k개의 값과 인덱스를 반환한다. |
|  | `Tensor.cpu` | 9-2 | 텐서를 CPU로 옮긴다. |
|  | `Tensor.expand` | 9-3 | 크기가 1인 차원을 늘려 형태를 맞춘다. 메모리를 새로 쓰지 않는다. |
|  | `Tensor.masked_fill` | 9-3 | 마스크가 True인 위치를 지정한 값으로 채운다. 어텐션 마스킹에 쓴다. |
|  | `Tensor.repeat` | 9-3 | 텐서를 지정한 횟수만큼 반복해 이어 붙인다. |
|  | `Tensor.pow` | 11-1 | 요소별 거듭제곱을 계산한다. |

## torch.Tensor 제자리 연산 메서드

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 메서드 | `Tensor.zero_` | 2-2 | 텐서의 모든 요소를 0으로 바꾸는 제자리 연산. 기울기 초기화에 쓴다. |

---

## torchvision.datasets.*

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `datasets.MNIST` | 4-3 | 손글씨 숫자 이미지 내장 데이터셋. |
|  | `datasets.CIFAR10` | 5-3 | 10종류 객체의 32x32 컬러 이미지 내장 데이터셋. |
|  | `datasets.ImageFolder` | 8-4 | 클래스별 하위 디렉터리로 정리된 이미지 폴더를 데이터셋으로 만든다. |
|  | `datasets.FashionMNIST` | 11-1 | 의류 이미지 내장 데이터셋. MNIST와 형태가 같다. |

## torchvision.transforms.*

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `transforms.CenterCrop` | 4-3 | 이미지의 가운데를 지정한 크기로 잘라 낸다. |
|  | `transforms.ColorJitter` | 4-3 | 밝기, 대비, 채도를 무작위로 바꾼다. |
|  | `transforms.Compose` | 4-3 | 여러 데이터 변환 객체를 순서대로 묶어 하나로 만든다. |
|  | `transforms.Grayscale` | 4-3 | 이미지를 회색조로 바꾼다. |
|  | `transforms.Normalize` | 4-3, 7-3 | 지정한 평균과 표준편차로 텐서를 표준화한다. |
|  | `transforms.RandomCrop` | 4-3 | 이미지의 일부를 무작위 위치에서 잘라 낸다. |
|  | `transforms.Resize` | 4-3 | 이미지를 지정한 크기로 조정한다. |
|  | `transforms.ToTensor` | 4-3, 7-3 | Pillow 이미지나 넘파이 배열을 0~1 범위의 텐서로 바꾼다. |
|  | `transforms.RandomHorizontalFlip` | 5-3 | 지정한 확률로 이미지를 좌우 반전한다. 데이터 증강에 쓴다. |

## torchvision.models.*

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `models.ResNet50_Weights` | 13-1 | ResNet-50의 사전 학습 가중치를 고르는 열거형. |
| 함수 | `models.resnet50` | 8-4 | ImageNet으로 사전 학습된 ResNet-50 모델을 불러온다. |

---

## torchinfo

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 함수 | `torchinfo.summary` | 2-3, 5-2 | 모델의 계층별 출력 형태와 파라미터 수를 표로 보여 준다. `print(model)`과 달리 데이터 흐름 순서를 따른다. |

## timm

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `timm.layers.SelectAdaptivePool2d` | 8-4 | 풀링 방식을 고를 수 있는 적응형 풀링 계층. |
| 함수 | `timm.create_model` | 8-4 | 모델 이름으로 사전 학습 모델을 만든다. `pretrained=True`로 가중치까지 받는다. |
|  | `timm.data.create_transform` | 8-4 | `resolve_data_config()`가 찾은 설정으로 데이터 변환 객체를 만든다. |
|  | `timm.data.resolve_data_config` | 8-4 | 모델에 맞는 입력 전처리 설정을 찾아 준다. |

## transformers (허깅페이스)

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `AutoModelForCausalLM` | 12-1 | 다음 토큰을 예측하는 인과 언어 모델용 Auto 클래스. |
|  | `AutoModelForMaskedLM` | 12-1 | 가려진 토큰을 맞히는 마스크 언어 모델용 Auto 클래스. |
|  | `AutoModelForQuestionAnswering` | 12-1 | 질의응답 모델용 Auto 클래스. |
|  | `AutoModelForSequenceClassification` | 12-1 | 문장 분류 모델용 Auto 클래스. |
|  | `AutoModelForTokenClassification` | 12-1 | 개체명 인식처럼 토큰마다 레이블을 붙이는 모델용 Auto 클래스. |
|  | `AutoTokenizer` | 12-1 | 모델에 맞는 토크나이저를 자동으로 고르는 클래스. |
|  | `AutoModel` | 12-2 | 모델 이름으로 알맞은 모델 클래스를 자동으로 고르는 클래스. |
|  | `AutoModelForSeq2SeqLM` | 12-2 | 입력을 받아 새 문장을 만드는 인코더-디코더 모델용 Auto 클래스. |
|  | `DataCollatorForSeq2Seq` | 12-2 | 배치를 만들 때 길이가 다른 샘플을 패딩해 묶어 준다. |
|  | `Seq2SeqTrainer` | 12-2 | 인코더-디코더 모델의 미세 조정을 맡는 학습 엔진. |
|  | `Seq2SeqTrainingArguments` | 12-2 | `Seq2SeqTrainer`의 학습 설정을 담는 객체. |
|  | `Trainer` | 12-2 | transformers의 기본 학습 엔진. 학습 루프를 대신 돌려 준다. |
|  | `BitsAndBytesConfig` | 12-3 | 4비트나 8비트 양자화 설정을 담는 객체. `from_pretrained()`에 넘긴다. |
|  | `CLIPModel` | 13-1 | 이미지와 텍스트를 같은 공간에 정렬한 CLIP 모델. |
|  | `CLIPProcessor` | 13-1 | CLIP의 이미지 전처리와 텍스트 토크나이징을 함께 처리한다. |
|  | `GPT2LMHeadModel` | 13-1 | 언어 모델 헤드가 붙은 GPT-2 모델. |
|  | `GPT2Tokenizer` | 13-1 | GPT-2용 토크나이저. |
|  | `Blip2ForConditionalGeneration` | 13-2 | 이미지를 보고 문장을 생성하는 BLIP-2 모델. |
|  | `Blip2Processor` | 13-2 | BLIP-2의 이미지와 텍스트 전처리를 담당한다. |
| 함수 | `pipeline` | 12-1 | 작업 이름과 모델 이름만 주면 토크나이저 준비부터 추론까지 한 번에 처리한다. |
| 메서드 | `*.from_pretrained` (클래스 메서드) | 12-1 | 허깅페이스 허브나 로컬 경로에서 사전 학습 가중치를 불러오는 클래스 메서드. |
|  | `model.generate` | 12-1 | 언어 모델이 토큰을 차례로 생성하게 한다. 빔 서치와 샘플링 방식을 인자로 고른다. |
|  | `tokenizer.apply_chat_template` | 12-1 | 대화 형식의 메시지 목록을 모델이 기대하는 프롬프트 문자열로 바꾼다. |
|  | `tokenizer.convert_tokens_to_ids` | 12-1 | 토큰 목록을 고유 번호 목록으로 바꾼다. |
|  | `tokenizer.decode` | 12-1 | 토큰 고유 번호 목록을 다시 문자열로 되돌린다. |
|  | `tokenizer.tokenize` | 12-1 | 문자열을 토큰 목록으로 쪼갠다. |
|  | `Trainer.train` | 12-2 | 미세 조정 학습을 실행한다. |
|  | `model.get_image_features` (CLIP) | 13-1 | CLIP 모델에서 이미지 임베딩 벡터를 뽑는다. |

## datasets (허깅페이스)

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `Dataset` (HF) | 12-2 | 허깅페이스 datasets 라이브러리의 데이터셋 객체. 파이토치 `Dataset`과 다르다. |
|  | `DatasetDict` | 12-2 | `train`, `validation` 같은 분할을 담은 딕셔너리형 데이터셋 묶음. |
| 함수 | `load_dataset` | 12-2, 13-1 | 허깅페이스 허브의 데이터셋을 불러온다. |
| 메서드 | `Dataset.map` | 12-2 | 데이터셋의 모든 샘플에 함수를 적용한다. 토크나이징에 쓴다. |

## sentence_transformers

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `SentenceTransformer` | 12-4 | 문장을 통째로 임베딩 벡터로 바꾸는 모델 클래스. |
| 메서드 | `SentenceTransformer.encode` | 12-4 | 문장이나 문장 목록을 임베딩 벡터로 바꾼다. |

## groq

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `Groq` | 12-4 | Groq LLM API 클라이언트 객체. |
| 메서드 | `client.chat.completions.create` | 12-4 | Groq API에 대화 메시지를 보내 답변을 받는다. |

## PIL (Pillow)

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 함수 | `Image.open` | 5-3 | 이미지 파일을 Pillow 객체로 불러온다. |
| 자료형 | `Image.Image` (자료형) | 4-3 | Pillow의 이미지 객체 자료형. |

## numpy

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 함수 | `np.array` | 1-1 | 리스트로 넘파이 배열을 만든다. |
| 자료형 | `np.int64` | 1-1 | 넘파이의 64비트 정수 자료형. |

## requests

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 함수 | `requests.get` | 8-4 | HTTP GET 요청을 보낸다. 레이블 파일을 내려받을 때 쓴다. |

---

## 파이썬 표준 라이브러리

| 종류 | 이름 | 절 | 설명 |
|---|---|---|---|
| 클래스 | `collections.OrderedDict` | 3-3 | 순서를 기억하는 딕셔너리. 계층에 이름을 붙여 모델을 만들 때 쓴다. |
|  | `csv.DictReader` | 3-3 | CSV 파일을 행마다 딕셔너리로 읽는다. |
| 함수 | `copy.deepcopy` | 4-1, 7-4 | 객체를 깊은 복사한다. 최적 모델 파라미터를 보관할 때 쓴다. |

