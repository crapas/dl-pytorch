# 케라스 API로 다시 쓰는 합성곱 신경망

> **5장 5-3절 추가 학습 자료**
> 예제 노트북: [`keras_api_cnn_example.ipynb`](keras_api_cnn_example.ipynb)
> 선수 지식: 4장(학습 루프와 조기 종료), 5-2절(합성곱 블록), 5-3절(CIFAR-10 분류기, 데이터 증강, 드롭아웃)
> 이어지는 글: [케라스 API로 다시 쓰는 순환 신경망](../keras-api-2/) - 6-3절

이 책은 내부 구현과 API 모두 파이토치를 기준으로 한다. 

하지만 현업에서 마주치는 코드가 모두 그렇게 쓰여 있지는 않다. 특히 **케라스**(Keras)로 쓰인 코드는
이 책의 예제와 겉모습이 꽤 다르다. 이 글은 그 코드를 읽을 수 있게 해 주는 것을 목표로 한다.
5-3절의 CIFAR-10 분류기를 케라스로 다시 만들면서, 이 책에서 직접 만들었던 것들이 케라스에서는
어디로 갔는지 하나씩 짚는다.

케라스를 새 프레임워크로 배우는 것이 아니다. **이미 학습한 것을 다르게 읽고 쓰는 법**을 배우는 것이다.

---

## 1. 케라스는 무엇인가

케라스는 2015년에 나온 딥러닝 라이브러리로, 오랫동안 텐서플로의 고수준 API 자리를 지켰다.
2023년에 나온 **케라스 3**부터는 성격이 달라졌다. 텐서플로에 매이지 않고
**파이토치, 텐서플로, JAX 중 어느 것 위에서든 동작한다.** 이것을 백엔드(backend)라고 부른다.

우리에게 중요한 것은 그 다음이다. 파이토치를 백엔드로 고르면,

- 실제 연산은 파이토치가 한다. 텐서는 `torch.Tensor`이고, GPU도 파이토치가 쓰던 그대로 쓴다.
- 이 책에서 만든 **`Dataset`과 `DataLoader`를 그대로 쓸 수 있다.**
- 파이토치 생태계의 다른 도구(`torchvision` 등)도 함께 쓸 수 있다.

즉 케라스 3 + 파이토치 백엔드는 **파이토치를 버리는 선택이 아니라, 파이토치 위에 얇은 껍질을
하나 씌우는 선택**이다. 그래서 이 책을 끝낸 다음에 가장 부담 없이 시도해 볼 수 있다.

### 설치

```bash
pip install keras
```

파이토치가 이미 설치되어 있다면 이것으로 끝이다. 텐서플로도 JAX도 필요 없다.

### 백엔드 지정

백엔드는 `keras`를 **임포트하기 전에** 환경 변수로 정한다.

```python
import os
os.environ['KERAS_BACKEND'] = 'torch'   # 반드시 import keras 앞에

import keras
print(keras.backend.backend())          # torch
```

임포트한 뒤에 환경 변수를 바꿔도 반영되지 않는다. 케라스가 임포트 시점에 백엔드를 확정하기
때문이다. 노트북에서 이 줄을 뒤늦게 고쳤다면 커널을 다시 시작해야 한다.

환경 변수 대신 홈 디렉터리의 `~/.keras/keras.json` 파일에 적어 둘 수도 있다.
다만 예제 노트북처럼 다른 환경에서도 그대로 돌아가야 하는 코드라면 환경 변수 쪽이 안전하다.

### 이미지 텐서의 채널 위치

한 가지를 더 맞춰야 한다. 5-1절에서 본 대로 `torchvision`이 만드는 이미지 텐서는
채널이 앞에 오는 `(B, C, H, W)` 형태다. 반면 케라스의 기본값은 채널이 뒤에 오는
`(B, H, W, C)`다. 텐서플로에서 온 관습이다.

```python
keras.config.set_image_data_format('channels_first')
```

이 한 줄을 프로그램 시작 부분에 넣어 두면, 이 책의 데이터로더를 **변환 없이 그대로** 넘길 수 있다.
이 설정을 빠뜨리면 합성곱 계층이 32를 채널 수로 착각해 엉뚱한 형태의 오류를 낸다.
케라스 코드에서 `permute()`나 `transpose()`가 보인다면 이 설정을 하지 않았기 때문일 가능성이 높다.

---

## 2. 무엇이 어떻게 바뀌나?

5-3절 코드 예제에 사용한 요소가 케라스의 무엇에 대응하는지 정리해 보자.

| 본문 5-3절 | 케라스 |
|---|---|
| `nn.Module` 상속 클래스 | `keras.Sequential` 또는 `keras.Model` 상속 클래스 |
| `nn.Conv2d(3, 16, 5, 1, 2)` | `layers.Conv2D(16, 5, padding='same')` |
| `nn.ReLU()`, `nn.MaxPool2d(2)` | `layers.ReLU()`, `layers.MaxPooling2D(2)` |
| `nn.Flatten()`, `nn.Linear(2048, 256)` | `layers.Flatten()`, `layers.Dense(256)` |
| `nn.Dropout(p=0.5)` | `layers.Dropout(0.5)` |
| `print(model)` | `model.summary()` |
| `nn.CrossEntropyLoss()` | `compile(loss=SparseCategoricalCrossentropy(from_logits=True))` |
| `optim.Adam(model.parameters(), lr=LR)` | `compile(optimizer=Adam(learning_rate=LR))` |
| `train_batch()`, `validate()`, `train_with_early_stopping()` | `fit()` |
| 조기 종료 판단과 `best_model_params` | `EarlyStopping` 콜백 |
| `common.EpochLogger` | `fit()`의 진행 표시와 반환값 `History` |
| `common.get_accuracy()` | `model.evaluate()` |
| `torch.device()`와 `.to(device)` | (없음 - 케라스가 처리) |
| `model.train()` / `model.eval()` | (없음 - `fit()`과 `evaluate()`가 전환) |
| `Dataset`, `DataLoader` | **그대로 사용** |
| `transforms.Compose` | **그대로 사용** (증강은 예외, 5절 참고) |

오른쪽 열에서 '없음'과 '그대로 사용'이 핵심이다.
**데이터를 다루는 층은 바뀌지 않고, 모델을 정의하고 학습시키는 층만 바뀐다.**

---

## 3. 모델 정의 - 입력 채널을 세지 않는다

본문 [코드 5-14]의 `CIFAR10ConvClassifier`를 케라스로 옮기면 이렇게 된다.

```python
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(3, 32, 32)),
    layers.Conv2D(16, 5, padding='same'), layers.ReLU(), layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, padding='same'), layers.ReLU(), layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(256), layers.ReLU(),
    layers.Dense(10),
])
```

본문과 다른 점이 두 가지 있다.

**입력 채널 수를 쓰지 않는다.** 본문은 `nn.Conv2d(in_channels=3, out_channels=16, ...)`처럼
입력과 출력을 모두 적었고, 두 번째 블록의 `in_channels=16`은 앞 블록의 출력을 보고 직접 맞춰야 했다.
`nn.Linear(32 * 8 * 8, 256)`의 2048도 두 번의 풀링을 거친 크기를 손으로 계산한 값이다.
케라스는 `Input`에 준 형태에서 시작해 각 계층의 입력 크기를 스스로 알아낸다.
계층을 하나 끼워 넣어도 뒤쪽 숫자를 고칠 일이 없다.

**패딩을 숫자가 아니라 이름으로 준다.** 본문의 `padding=2`(5x5 필터)와 `padding=1`(3x3 필터)은
모두 "출력 크기를 입력과 같게 유지"하려는 것이었다. 5-1절에서 본 대로 필터 크기가 홀수 `k`일 때
그 값은 `(k-1)/2`다. 케라스는 이 의도를 `padding='same'`이라는 이름으로 받는다.
패딩을 아예 주지 않으려면 `padding='valid'`다.

두 모델의 파라미터 수는 **532,970개로 정확히 같다.** 계층별 출력 형태도 같다.
같은 모델을 다른 표기법으로 적은 것이 맞다는 뜻이다.

### 같은 구조라고 결과까지 같지는 않다

파라미터 수가 같아도 학습 결과의 숫자는 일치하지 않는다. 가중치의 **초깃값을 정하는 방식**이
다르기 때문이다.

| | 가중치 | 편향 |
|---|---|---|
| 파이토치 `nn.Conv2d` | 카이밍 균등 분포 | 카이밍 균등 분포 |
| 케라스 `layers.Conv2D` | 글로럿 균등 분포 | **0** |

초깃값이 다르면 경사 하강이 지나는 경로가 달라지고, 도착점도 달라진다.
그러니 두 구현을 비교할 때 볼 것은 소수점 아래 숫자가 아니라 **경향**이다.

예제 노트북의 결과를 본문 [표 5-5]와 나란히 놓아 보자.

| 구분 | 최적 에포크 (본문 / 케라스) | 평가 정확도 (본문 / 케라스) |
|---|---|---|
| 기본 모델 | 7 / 7 | 69.33% / 67.62% |
| 증강 적용 | 11 / 8 | 72.26% / 72.74% |
| 증강 + 드롭아웃 | 17 / 18 | 72.68% / 74.09% |

숫자는 조금씩 어긋나지만 **읽어 내는 바는 같다.** 증강을 넣으면 과적합이 늦춰져 최적 에포크가
뒤로 밀리고, 드롭아웃을 더하면 더 밀린다. 평가 정확도는 두 처방을 더할 때마다 올라간다.
5-3절이 말하려던 것이 그대로 재현된다.

---

## 4. 학습 - 60줄이 두 줄로

이 글에서 차이가 가장 큰 곳이다.

본문 4-2절부터 5-3절까지 우리는 학습 루프를 직접 만들었다. 에포크를 반복하고, 미니배치를 꺼내고,
기울기를 초기화하고, 순전파와 역전파를 부르고, 손실을 샘플 수로 가중해 누적하고,
검증 손실의 최솟값을 추적해 조기 종료를 판단하고, 최적 파라미터를 복사해 두었다가 되돌렸다.
`train_batch()`, `validate()`, `train_with_early_stopping()` 세 함수에 걸쳐 60줄 남짓이다.

케라스에서는 이렇게 쓴다.

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

history = model.fit(
    train_loader,                       # 파이토치 DataLoader 를 그대로
    validation_data=valid_loader,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, restore_best_weights=True)],
)
```

`compile()`은 **무엇으로 학습할지**(손실 함수, 옵티마이저, 지켜볼 지표)를 정하고,
`fit()`은 **학습을 실행한다.** 두 단계로 나뉘어 있는 것이 처음에는 어색하지만,
본문에서 `criterion`과 `optimizer`를 먼저 만들고 그다음에 학습 루프를 돌렸던 것과 같은 순서다.

### 손실 함수의 긴 이름

`SparseCategoricalCrossentropy(from_logits=True)`는 본문의 `nn.CrossEntropyLoss()`와
**같은 손실 함수다.** 이름이 긴 것은 파이토치가 기본값으로 정해 둔 두 가지를 겉으로 드러내기
때문이다.

- **`Sparse`** - 정답을 원-핫 벡터가 아니라 클래스 번호 하나로 준다는 뜻이다.
  본문의 정답 텐서가 그렇다. 원-핫 벡터라면 `Sparse`를 뺀 `CategoricalCrossentropy`를 쓴다.
- **`from_logits=True`** - 모델의 출력이 확률이 아니라 로짓이라는 뜻이다.
  3-3절에서 본 대로 본문의 모델도 마지막에 소프트맥스를 두지 않고 로짓을 그대로 내보낸다.

`from_logits`를 빠뜨리는 것이 케라스에서 가장 흔한 실수다.
기본값이 `False`라서, 모델이 로짓을 내보내는데도 케라스는 그것을 확률로 믿고 손실을 계산한다.
오류가 나지 않고 **조용히 잘못된 손실로 학습된다.** 정확도가 이유 없이 낮다면 여기를 먼저 보자.

### 콜백

`fit()`이 학습 루프를 가져간 대신, 루프 중간에 끼어들 자리를 **콜백**(callback)으로 열어 둔다.
본문의 조기 종료 코드는 `EarlyStopping` 콜백 하나에 대응한다.

| 본문의 변수 | `EarlyStopping` 인자 |
|---|---|
| `best_valid_loss` 추적 | `monitor='val_loss'` |
| `patience_counter` | `patience=PATIENCE` |
| `best_model_params` 복사와 복원 | `restore_best_weights=True` |

`restore_best_weights`의 기본값은 `False`다. 이때는 학습이 멈춘 시점,
즉 검증 손실이 이미 `patience` 에포크만큼 나빠진 상태의 파라미터가 남는다.
본문이 최적 파라미터를 따로 복사해 둔 이유가 여기서 다시 드러난다.

자주 쓰는 콜백이 몇 가지 더 있다.

- `ModelCheckpoint` - 에포크마다 모델을 파일로 저장한다.
- `ReduceLROnPlateau` - 검증 손실이 나아지지 않으면 학습률을 줄인다.
- `CSVLogger` - 에포크별 손실과 지표를 CSV로 남긴다.

---

## 5. 데이터 증강 - 변환이 아니라 계층으로

본문 [코드 5-17]은 조금 번거로운 코드였다. `transforms.RandomHorizontalFlip()`은
**데이터셋에 붙는 변환**이라서 지금이 훈련 중인지 검증 중인지 알지 못한다.
그래서 증강을 적용한 데이터셋과 적용하지 않은 데이터셋을 각각 만들고,
**같은 시드로** 분리해 훈련 쪽 40,000개와 검증 쪽 10,000개가 겹치지 않도록 맞춰야 했다.

케라스의 `layers.RandomFlip`은 **모델 안에 들어가는 계층**이다.

```python
model = keras.Sequential([
    layers.Input(shape=(3, 32, 32)),
    layers.RandomFlip('horizontal'),    # 학습할 때만 동작
    ...
])
```

드롭아웃과 마찬가지로 학습할 때만 동작하고, 평가할 때는 입력을 그대로 흘려보낸다.
훈련과 검증이 같은 데이터셋을 공유해도 문제가 없으므로 **데이터셋을 두 벌 만들 일이 없다.**
[코드 5-16]과 [코드 5-17]이 통째로 사라지는 셈이다.

이것이 케라스가 더 낫다는 뜻은 아니다. 두 방식은 증강을 **어디에서** 하느냐가 다르다.
`transforms` 쪽은 데이터로더의 워커 프로세스가 CPU에서 증강하고,
케라스의 전처리 계층은 모델의 일부이므로 **GPU에서** 증강한다.
어느 쪽이 유리한지는 증강의 무게와 데이터 적재가 병목인지에 따라 달라진다.

덧붙여, 증강 계층이 모델 안에 있으면 모델을 저장할 때 증강 설정도 함께 저장된다.
배포한 모델이 학습 때와 다른 증강을 쓰는 사고를 막아 준다는 이점이 있다.

---

## 6. 가속기와 학습 모드 - 쓰지 않는 코드

본문 [코드 5-20]부터 [코드 5-22]까지는 하드웨어 가속기를 다뤘다.
장치 객체를 만들고, 모델과 입력 텐서를 `.to(device)`로 옮기고,
서드파티 라이브러리에 넘기기 전에 다시 CPU로 되돌렸다.

케라스에서는 이 코드가 모두 사라진다. 사용 가능한 가속기를 찾아 모델과 데이터를 옮기는 일을
케라스가 알아서 한다. 예제 노트북에서 모델 세 개를 학습시키는 동안 `.to(device)`가
한 번도 나오지 않는다는 점을 확인해 보자.

`model.train()`과 `model.eval()`도 없다. `fit()`은 학습 모드로, `evaluate()`와 `predict()`는
평가 모드로 자동 전환한다. 4-3절에서 강조했던 "평가 전에 `eval()`을 부르지 않으면 드롭아웃이
평가에서도 동작한다"는 함정이 원천적으로 막힌다.

**편해진 만큼 보이지 않게 된 것도 있다.** 모드가 언제 바뀌는지, 텐서가 언제 어디로 옮겨지는지가
코드에 드러나지 않는다. 뜻대로 동작하지 않을 때 들여다볼 곳이 줄어든다는 뜻이기도 하다.
이 책이 학습 루프를 직접 만들어 본 이유가 여기에 있다. **한 번 만들어 본 사람에게만
`fit()` 한 줄이 무엇을 줄여 준 것인지 보인다.**

---

## 7. 속도는 어떤가

케라스는 배치마다 파이썬 수준의 일을 조금 더 한다. 학습 단계를 감싸고, 지표를 모으고,
콜백을 호출한다. 그만큼 순수 파이토치보다 느리다.

같은 장비(RTX 4070 SUPER)에서 이 글의 CIFAR-10 분류기를 같은 설정으로 학습시켜 재 보면,
**파이토치 111초, 케라스 148초로 1.3배가량** 차이가 난다.
모델이 훨씬 가벼우면 이 비율은 커지고(연산 대비 오버헤드의 몫이 커지므로),
모델이 무거우면 1에 가까워진다.

정리하면 이렇다.

- 배치 크기가 작고 모델이 가벼운 실험에서는 오버헤드가 눈에 띈다.
- 실무 규모의 모델에서는 대체로 무시할 만하다.
- GPU 지원 자체에는 아무 문제가 없다. 파이토치가 쓰던 CUDA를 그대로 쓴다.

다만 이것이 전부는 아니다. 여기 1.3배는 **배치마다 붙는 고정 비용**이라 예측 가능한 값이지만,
계층에 따라서는 훨씬 크게 벌어지기도 한다. 이어지는 [keras-api-2](../keras-api-2/) 7절에서
같은 장비에서 10배 넘게 차이 나는 경우를 다룬다. 고수준 API에는 **잘 닦인 길과 그렇지 않은 길**이
있고, 어느 쪽인지는 직접 재 봐야 알 수 있다.

---

## 8. 이 글이 다루지 않는 것

케라스로 이 책 전체를 다시 쓸 수는 없다. 몇 가지 예를 들면,

- **트랜스포머**(10장). 케라스 본체에는 `MultiHeadAttention`과 `LayerNormalization`은 있지만
  `nn.TransformerEncoderLayer`에 해당하는 것이 없다. 별도 패키지인 KerasHub를 써야 한다.
- **훅**(8장 심화 학습 자료). 파이토치의 `register_forward_hook()`에 해당하는 기능이 없다.
- **`pack_padded_sequence`**(9장). 가변 길이 시퀀스를 다루는 방식이 다르다.
- **파라미터 그룹별 학습률**(7장), **양자화와 LoRA**(12장). 파이토치 생태계 쪽 도구가 필요하다.

케라스는 흔한 구조를 짧게 쓰는 데 강하고, 흔하지 않은 것을 만들 때는 파이토치가 강하다.
이 책이 파이토치를 고른 이유이기도 하다.

이어지는 [keras-api-2](../keras-api-2/)에서는 6-3절의 순환 신경망을 케라스로 옮기면서,
두 프레임워크가 **같은 수식을 다르게 구현하는** 지점을 본다.

---

## 연습 문제

**1.** `build_classifier()`가 만든 케라스 모델과 `CIFAR10ConvClassifier()`가 만든 파이토치 모델의
계층별 파라미터 수를 각각 출력해 나란히 비교해 보자.

> 힌트: 파이토치 모델은 `named_children()`으로 하위 모듈을 순회할 수 있다.

**2.** 3절의 `padding='same'`을 `padding='valid'`로 바꾸면 `Flatten()` 뒤의 텐서 크기가 어떻게
달라지는지 `summary()`로 확인해 보자. 그리고 본문이 왜 `padding`을 2와 1로 지정했는지 설명해 보자.

**3.** 4절의 `compile()`에서 `from_logits=True`를 빼면 어떤 일이 벌어질까?
먼저 예상해 본 다음 실제로 학습시켜 정확도를 확인하고, 왜 그런 결과가 나오는지 설명해 보자.

**4.** `EarlyStopping`에서 `restore_best_weights=False`(기본값)로 두고 학습시킨 뒤 평가 정확도를
측정해 보자. `True`일 때와 얼마나 차이가 나는지 확인하고, 본문의 `best_model_params` 변수가
왜 필요했는지 설명해 보자.

**5.** 5절의 `layers.RandomFlip('horizontal')`을 `'horizontal_and_vertical'`로 바꿔 학습해 보자.
평가 정확도가 어떻게 달라지는지 확인하고, CIFAR-10 이미지의 성격과 연결해 그 이유를 설명해 보자.

**6.** [도전 문제] `keras.callbacks.ReduceLROnPlateau`를 추가해 검증 손실이 나아지지 않을 때
학습률을 줄이도록 하고, 학습 곡선이 어떻게 달라지는지 관찰해 보자.

**7.** [도전 문제] 이 노트북은 `keras.Sequential`로 모델을 만들었다.
`keras.Model`을 상속한 클래스로 같은 모델을 다시 만들어 보고, 본문의 `nn.Module` 상속 방식과
무엇이 같고 무엇이 다른지 정리해 보자.

> 힌트: `__init__()`에서 계층을 만들고 `call()`에서 이어 붙인다. 파이토치의 `forward()`에 해당하는 것이 `call()`이다.

연습 문제의 풀이는 따로 싣지 않았다. 모두 예제 노트북의 코드를 조금씩 고쳐 직접 돌려 보는 문제이고,
**결과를 예상한 다음 확인하는 것**이 이 문제들의 요점이기 때문이다.
다시 학습시켜야 하는 문제는 시간이 걸리므로 에포크 수를 줄여 경향만 확인해도 된다.

---

## 더 찾아보면 좋은 것

- 케라스 공식 문서의 [Getting started with Keras](https://keras.io/getting_started/)
- [Keras 3 백엔드별 안내](https://keras.io/keras_3/) - 백엔드를 바꿔 가며 같은 코드를 돌리는 예
- [KerasHub](https://keras.io/keras_hub/) - 트랜스포머 계열 사전 학습 모델을 제공하는 별도 패키지
- `keras.callbacks` 문서 - `fit()`이 가져간 학습 루프에 끼어드는 모든 방법
