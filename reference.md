# 본문에서 언급하는 연구의 관련 문헌

> AI가 자동 생성한 후, 저자가 검토, 수정한 파일임을 밝혀 둡니다.

- <멀티모달까지 직접 구현하는 딥러닝 with 파이토치> 본문에서 소개한 연구 논문과 서적의 출처
- 각 항목의 **공개 여부**에 관하여
    - **공개**는 인터넷에서 전문을 무료로 볼 수 있음을 뜻하며
    - **비공개**인 경우 원칙적으로 유료로 제공된다는 뜻이다.
    - 다만 비공개라도 저자가 개인 홈페이지등을 통해 따로 공개한 경우, 해당 링크를 함께 기록해 둔다.
- 각 항목 앞의 인용문은, 책 본문에서 해당 연구를 소개한 대목을 옮긴 것이다.

---

## 1장 텐서와 자동 미분

### **Computing Machinery and Intelligence(계산 기계와 지능)**

> 1950년 철학 저널인 ‘마인드’ 제59권 236호에 영국 수학자 앨런 튜링Alan Turing의 논문 한 편이 실렸다. <계산 기계와 지능Computing Machinery and Intelligence>이라는 제목의 논문에서 튜링은 ‘기계는 생각할 수 있는가?’라는 질문을 던지고, 이를 검증하는 방법으로 튜링 테스트를 제안했다. 오늘날, 이 논문을 인공지능의 시작으로 보는 데는 큰 이견이 없다.
>
> — 본문 2쪽

- 저자: A. M. Turing
- 출처: Mind, Vol. LIX, No. 236, pp. 433–460 (1950)
- DOI: <https://doi.org/10.1093/mind/LIX.236.433>
- 공개 여부: **공개**

---

## 2장 똑똑한 선 긋기, 퍼셉트론

### **A Logical Calculus of the Ideas Immanent in Nervous Activity**

> 1943년 신경생리학자 워런 맥컬럭Warren McCulloch과 수학자 월터 피츠Walter Pitts가 함께 인간의 뇌세포가 작동하는 방식을 수학적으로 모사한 모델을 발표했다.
>
> — 본문 2쪽

- 저자: Warren S. McCulloch, Walter Pitts
- 출처: Bulletin of Mathematical Biophysics, Vol. 5, No. 4, pp. 115–133 (1943)
- DOI: <https://doi.org/10.1007/BF02478259>
- 공개 여부: 비공개

### **The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain(퍼셉트론: 뇌의 정보 저장과 조직화에 관한 확률 모델)**

> 이 논문은 많은 사람에게 영감을 주었고, 코넬 항공 연구소의 프랭크 로젠블랫Frank Rosenblatt도 그중 한 사람이었다. 로젠블랫은 맥컬럭과 피츠가 고안한 인공 뉴런의 한계, 즉 가중치 값이 고정되어 있다는 점에 주목했다. 그는 인간의 뇌처럼 필요에 따라 스스로 지식을 습득하고 가중치를 조정할 수 있는 모델을 연구했고, 그 결과 1957년 최초의 학습 가능한 인공 뉴런인 퍼셉트론을 제안했다.
>
> — 본문 45쪽

- 저자: Frank Rosenblatt
- 출처: Psychological Review, Vol. 65, No. 6, pp. 386–408 (1958)
- DOI: <https://doi.org/10.1037/h0042519>
- 공개 여부: 비공개
- 특이사항: 퍼셉트론은 1957년 로젠블랫이 재직하던 코넬 항공 연구소의 보고서를 통해 최초로 공개되었으며, 이후 1958년 논문으로 발표되었다. 로젠블랫의 퍼셉트론 논문은 보통 후자를 말한다.

---

## 3장 선을 넘는 신경망, 다층 퍼셉트론

### **Perceptrons: An Introduction to Computational Geometry(퍼셉트론: 계산 기하학 입문)**

> 1969년 마빈 민스키Marvin Minsky와 시모어 페퍼트Seymour Papert는 그들의 저서 <퍼셉트론>에서 퍼셉트론만으로는 하나의 직선으로 나눌 수 없는 문제를 풀 수 없음을 수학적으로 증명했다.
>
> — 본문 75쪽

- 저자: Marvin Minsky, Seymour Papert
- 출처: MIT Press, Cambridge, MA (1969)
- ISBN: 0-262-13043-2
- 공개 여부: 비공개 — Internet Archive에 접근 제한 대출 형태로 등록되어 있음, <https://archive.org/details/perceptronsintro00mins>

### **Learning Representations by Back-Propagating Errors(오차 역전파를 통한 표현 학습)**

> 1986년 데이비드 럼멜하트David Rumelhart, 제프리 힌턴Geoffrey Hinton, 로널드 윌리엄스Ronald Williams가 역전파 알고리즘으로 다층 퍼셉트론을 효과적으로 학습할 수 있음을 보이면서 돌파구가 열렸다.
>
> — 본문 75쪽

- 저자: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
- 출처: Nature, Vol. 323, No. 6088, pp. 533–536 (1986)
- DOI: <https://doi.org/10.1038/323533a0>
- 공개 여부: 비공개

---

## 4장 딥.완.데, 딥러닝의 완성은 데이터

### **ImageNet: A Large-Scale Hierarchical Image Database(이미지넷: 대규모 계층적 이미지 데이터베이스)**

> 2006년 당시 AI 연구자들의 관심은 더 좋은 모델, 더 좋은 학습 방법에 쏠려 있었지만, 페이페이 리Fei-Fei Li는 다른 선택을 했다. 리는 알고리즘보다 데이터가 더 중요하다 확신했고, 대규모 데이터셋 구축 프로젝트를 시작했다. 이후 수년간 수만 명이 참여해 1,400만 장의 이미지로 구성된 ImageNet 데이터셋이 완성되었다.
>
> — 본문 119쪽

- 저자: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei
- 출처: 2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 248–255 (2009)
- DOI: <https://doi.org/10.1109/CVPR.2009.5206848>
- 공개 여부: 비공개 — IEEE Xplore 유료 접근이 원칙이나, ImageNet 공식 사이트 image-net.org에서 무료 PDF를 함께 제공하고 있어 이쪽도 참고용으로 병기 가능

### **ImageNet Classification with Deep Convolutional Neural Networks(깊은 합성곱 신경망을 이용한 이미지넷 분류)**

> 2012년 ImageNet 데이터셋을 기반으로 한 이미지 인식 대회 ILSVRC에서 AlexNet알렉스넷이 압도적인 성능 차이로 우승하며 딥러닝 시대를 열었다.
>
> — 본문 119쪽

- 저자: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- 출처: Advances in Neural Information Processing Systems (NeurIPS), Vol. 25, pp. 1097–1105 (2012)
- DOI: 없음 (NeurIPS 2012 proceedings 원본 논문에는 DOI가 부여되어 있지 않음)
- 링크: <https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>
- 공개 여부: **공개** — NeurIPS 공식 proceedings 페이지에서 전문 PDF를 무료로 제공

### **Gradient-Based Learning Applied to Document Recognition(문서 인식에 적용된 그래디언트 기반 학습)**

> 여기에 포함된 MNIST 데이터셋은 얀 르쿤Yann LeCun이 LeNet-5 신경망을 개발하기 위해 미국 국립표준기술연구소(NIST)의 손 글씨 문서를 정제해 만든 숫자 데이터셋으로, 손 글씨 숫자 인식 모델 개발의 기준 데이터로 지금까지도 널리 사용되고 있다.
>
> — 본문 148쪽

- 저자: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
- 출처: Proceedings of the IEEE, Vol. 86, No. 11, pp. 2278–2324 (1998)
- DOI: <https://doi.org/10.1109/5.726791>
- 공개 여부: 비공개 — IEEE Xplore 기준. 다만 저자인 얀 르쿤의 개인 홈페이지에서 논문 PDF를 자체적으로 공개하고 있어 그쪽으로는 무료 열람 가능, <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>

---

## 5장 이미지를 보는 눈, 합성곱 신경망

### **Receptive Fields of Single Neurones in the Cat's Striate Cortex(고양이 선조 피질 단일 뉴런의 수용장)**

> 1959년 신경생리학자 데이비드 허블David Hubel과 토르스텐 비셀Torsten Wiesel은 고양이의 시각 피질에 전극을 꽂아 뇌세포의 반응을 기록하는 실험을 했다. 그 결과 고양이의 시야에 들어온 사물의 경계선 방향에 따라 뇌 속의 서로 다른 뉴런이 반응한다는 사실을 발견했다.
>
> — 본문 163쪽

- 저자: David H. Hubel, Torsten N. Wiesel
- 출처: The Journal of Physiology, Vol. 148, No. 3, pp. 574–591 (1959)
- DOI: <https://doi.org/10.1113/jphysiol.1959.sp006308>
- 공개 여부: **공개** — PubMed Central, PMCID: PMC1363130

### **Neocognitron: A Self-Organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position(네오코그니트론: 위치 이동에 영향받지 않는 패턴 인식 메커니즘을 위한 자기 조직화 신경망 모델)**

> ‘시각 피질이 하는 일을 인공 신경망도 할 수 있을까?’ 이 실험 결과가 먼 훗날 딥러닝에 던진 질문은 단순했다. 1980년 후쿠시마 구니히코福島 邦彦가 제안한 네오코그니트론Neocognitron은 이 질문에 대한 첫 번째 시도였다.
>
> — 본문 163쪽

- 저자: Kunihiko Fukushima(후쿠시마 구니히코)
- 출처: Biological Cybernetics, Vol. 36, No. 4, pp. 193–202 (1980)
- DOI: <https://doi.org/10.1007/BF00344251>
- 공개 여부: 비공개 — Springer 유료 접근

### **Backpropagation Applied to Handwritten Zip Code Recognition(손 글씨 우편번호 인식에 적용된 역전파)**

> 이 질문에 답한 연구자는 딥러닝의 살아 있는 전설 중 한 사람인 얀 르쿤으로, 1989년 역전파 알고리즘을 결합해 만들어 낸 합성곱 신경망이 바로 그 답이었다. 그리고 1998년에 발표한 LeNet-5는 그 답이 실용적이기까지 하다는 사실을 증명한 셈이었다.
>
> — 본문 164쪽

- 저자: Yann LeCun, Bernhard Boser, John S. Denker, Donnie Henderson, Richard E. Howard, Wayne Hubbard, Lawrence D. Jackel
- 출처: Neural Computation, Vol. 1, No. 4, pp. 541–551 (1989)
- DOI: <https://doi.org/10.1162/neco.1989.1.4.541>
- 공개 여부: 비공개
- 특이사항: 인용문의 앞부분, 1989년 합성곱 신경망에 해당하는 논문이다. 뒷부분에 언급된 1998년 LeNet-5 논문은 바로 아래의 <Gradient-Based Learning Applied to Document Recognition> 항목을 참고한다.

### **Gradient-Based Learning Applied to Document Recognition(문서 인식에 적용된 그래디언트 기반 학습)**

→ 4장의 같은 항목을 참고한다.

### **Learning Multiple Layers of Features from Tiny Images(작은 이미지의 다층적 특징 학습)**

> 2008년 MIT의 안토니오 토랄바Antonio Torralba 연구팀이 <8,000만 개의 작은 이미지80 Million Tiny Images>라는 데이터셋을 공개했다. 그리고 이듬해인 2009년 제프리 힌턴 연구실의 알렉스 크리제프스키Alex Krizhevsky가 기술 보고서를 통해 이 데이터셋의 일부를 정제하고 레이블을 붙인 CIFAR-10/CIFAR-100 데이터셋을 발표했다.
>
> — 본문 191쪽

- 저자: Alex Krizhevsky
- 출처: Technical Report, University of Toronto, Toronto, Ontario (2009)
- DOI: 없음 (저널 논문이 아닌 대학 기술 보고서라 DOI가 부여되지 않음)
- 링크: <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>
- 공개 여부: **공개** — 저자 개인 홈페이지에서 PDF 무료 제공

---

## 6장 순서를 기억하는 방법, 순환 신경망

### **Neural Networks and Physical Systems with Emergent Collective Computational Abilities(창발적 집단 연산 능력을 갖는 신경망과 물리계)**

> 홉필드의 업적 중 하나는 1982년으로 거슬러 올라간다. 그는 뉴런이 서로의 출력을 다시 입력으로 받아들이는 홉필드 네트워크Hopfield network를 제안했다.
>
> — 본문 209쪽

- 저자: John J. Hopfield
- 출처: Proceedings of the National Academy of Sciences, Vol. 79, No. 8, pp. 2554–2558 (1982)
- DOI: <https://doi.org/10.1073/pnas.79.8.2554>
- 공개 여부: **공개** — PNAS는 오래된 논문을 무료로 공개하는 정책을 따르고 있어 공식 페이지에서 전문 열람 가능, <https://www.pnas.org/doi/10.1073/pnas.79.8.2554>

### **Finding Structure in Time(시간 속 구조 찾기)**

> 1990년 제프리 엘먼Jeffrey Elman이 제안한 순환 신경망은 영어 문장에서 단어의 경계와 품사 규칙을 스스로 찾아내며 순서를 학습할 수 있음을 보였다.
>
> — 본문 210쪽

- 저자: Jeffrey L. Elman
- 출처: Cognitive Science, Vol. 14, No. 2, pp. 179–211 (1990)
- DOI: <https://doi.org/10.1207/s15516709cog1402_1>
- 공개 여부: **공개**

### **Long Short-Term Memory(장단기 기억)**

> 다만 입력 데이터가 길어지면 성능이 떨어지는 문제가 드러났는데, 1997년 제프 호흐라이터Sepp Hochreiter와 위르겐 슈미트후버Jürgen Schmidhuber가 장단기 기억 신경망을 발표해 돌파구를 마련했다.
>
> — 본문 210쪽

- 저자: Sepp Hochreiter, Jürgen Schmidhuber
- 출처: Neural Computation, Vol. 9, No. 8, pp. 1735–1780 (1997)
- DOI: <https://doi.org/10.1162/neco.1997.9.8.1735>
- 공개 여부: 비공개

### **Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation(RNN 인코더-디코더를 이용한 통계적 기계 번역의 구절 표현 학습)**

> 2014년 한국인 연구자인 조경현 교수가 동료들과 함께 제안한 GRU(gated recurrent unit)는 단순 RNN의 장기 의존성 문제를 개선한 또 하나의 순환 신경망 계층으로, LSTM과 함께 자주 언급된다.
>
> — 본문 248쪽

- 저자: Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
- 출처: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), Doha, Qatar, pp. 1724–1734 (2014)
- DOI: <https://doi.org/10.3115/v1/D14-1179>
- 공개 여부: **공개** — ACL Anthology에서 전문 PDF 무료 제공, <https://aclanthology.org/D14-1179/>, arXiv 프리프린트도 존재, arXiv:1406.1078

### **Bidirectional Recurrent Neural Networks(양방향 순환 신경망)**

> 파이토치에서는 nn.RNN, nn.LSTM, nn.GRU 생성자에 bidirectional=True 인자만 추가하면 양방향 구조가 된다.
>
> — 본문 251쪽

- 저자: Mike Schuster, Kuldip K. Paliwal
- 출처: IEEE Transactions on Signal Processing, Vol. 45, No. 11, pp. 2673–2681 (1997)
- DOI: <https://doi.org/10.1109/78.650093>
- 공개 여부: 비공개 — IEEE Xplore 유료 접근, <https://ieeexplore.ieee.org/document/650093>

---

## 7장 데이터 속의 의미

### **A Synopsis of Linguistic Theory, 1930–1955(언어학 이론 개관, 1930–1955)**

> 1957년 영국의 언어학자 존 루퍼트 퍼스John Rupert Firth는 “단어의 의미는 그것이 함께 등장하는 단어들로 알 수 있다”라는 유명한 말을 남겼다.
>
> — 본문 255쪽

- 저자: J. R. Firth
- 출처: Studies in Linguistic Analysis, pp. 1–32, Blackwell, Oxford (1957)
- DOI: 없음 (저널 논문이 아닌 단행본 논문집(edited volume)의 한 챕터라 DOI가 부여되지 않음)
- 공개 여부: 비공개

### **A Neural Probabilistic Language Model(신경망 확률적 언어 모델)**

> 분포 가설은 2003년 요슈아 벤지오Yoshua Bengio와 동료들이 발표한 논문 <신경망 확률론적 언어 모델>의 이론적 토대를 구성했다. 이 논문은 토큰을 의미 기반의 작은 크기의 실수 벡터로 인코딩하는 임베딩의 기초 원리를 담고 있었다.
>
> — 본문 255쪽

- 저자: Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin
- 출처: Journal of Machine Learning Research, Vol. 3, pp. 1137–1155 (2003)
- DOI: 없음 (JMLR은 오픈 액세스 저널로 대부분의 초기 논문에 DOI가 부여되지 않음)
- 공개 여부: **공개** — JMLR 공식 사이트에서 전문 PDF 무료 제공, <https://www.jmlr.org/papers/v3/bengio03a.html>

### **Efficient Estimation of Word Representations in Vector Space(벡터 공간에서의 단어 표현의 효율적 추정)**

> 시간이 흘러 2013년과 2014년, 구글의 Word2Vec과 스탠퍼드 대학교 연구팀의 GloVe 등의 임베딩 벡터가 잇따라 발표되면서 임베딩은 단어의 의미와 관계를 벡터로 다루는 표준 도구로 자리 잡았다.
>
> — 본문 255쪽

- 저자: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
- 출처: arXiv preprint, arXiv:1301.3781 (2013)
- DOI: 없음 (동료 심사를 거친 정식 학술지·학회 논문이 아닌 arXiv 프리프린트)
- 공개 여부: **공개** — arXiv에서 전문 무료 제공, <https://arxiv.org/abs/1301.3781>

### **GloVe: Global Vectors for Word Representation(글로브: 단어 표현을 위한 전역 벡터)**

> 2014년 스탠퍼드 대학교의 자연어 처리 연구팀은 GloVe라는 이름의 임베딩 모델을 발표했다.
>
> — 본문 261쪽

- 저자: Jeffrey Pennington, Richard Socher, Christopher D. Manning
- 출처: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1532–1543 (2014)
- DOI: <https://doi.org/10.3115/v1/D14-1162>
- 공개 여부: **공개** — ACL Anthology에서 전문 PDF 무료 제공, <https://aclanthology.org/D14-1162/>

### **Reducing the Dimensionality of Data with Neural Networks(신경망을 이용한 데이터 차원 축소)**

> 한동안 묻혀 있던 이 아이디어는 2006년 발표된 논문 <신경망을 활용한 데이터의 차원 축소>에서 오토인코더라는 이름으로 되살아났다.
>
> — 본문 256쪽

- 저자: Geoffrey E. Hinton, Ruslan R. Salakhutdinov
- 출처: Science, Vol. 313, No. 5786, pp. 504–507 (2006)
- DOI: <https://doi.org/10.1126/science.1127647>
- 공개 여부: **공개** — Science 측이 커뮤니티 서비스 차원에서 이 논문을 무료로 공개하고 있음, <https://www.science.org/doi/10.1126/science.1127647>

---

## 8장 더 깊은 딥러닝

### **ImageNet Classification with Deep Convolutional Neural Networks(깊은 합성곱 신경망을 이용한 이미지넷 분류)**

> 5장에서 합성곱 신경망이 계층을 거듭할수록 저수준 특징에서 고수준 특징으로 점점 더 깊이 있는 특징을 뽑아낸다는 점을 살펴봤다. 연구자들은 계층을 더 깊게 쌓을수록 모델의 성능이 비약적으로 좋아질 것으로 추정했다.
>
> — 본문 299쪽

- 저자: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- 출처: Advances in Neural Information Processing Systems (NeurIPS), Vol. 25, pp. 1097–1105 (2012)
- DOI: 없음 (NeurIPS 2012 proceedings 원본 논문에는 DOI가 부여되어 있지 않음)
- 링크: <https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>
- 공개 여부: **공개** — NeurIPS 공식 proceedings 페이지에서 전문 PDF 무료 제공

### **Very Deep Convolutional Networks for Large-Scale Image Recognition(대규모 이미지 인식을 위한 매우 깊은 합성곱 신경망)**

> 이후 영국 옥스퍼드 대학교의 시각 기하학 연구 그룹Visual Geometry Group의 카렌 시모니안Karen Simonyan과 앤드루 지서먼Andrew Zisserman은 2014년 VGGNet을 개발했다. 그해 이미지 인식 대회 1위는 구글의 GoogLeNet이 차지했지만, 이 대회에서 2위에 오른 VGGNet은 최대 19층의 깊은 신경망이면서도, 더 깊게 확장하기 쉬운 단순한 구조 덕에 초기 심층 신경망을 대표하는 모델이 되었다.
>
> — 본문 300쪽

- 저자: Karen Simonyan, Andrew Zisserman
- 출처: International Conference on Learning Representations (ICLR) 2015 (arXiv 최초 게재: 2014)
- DOI: 없음 (ICLR은 오픈리뷰 기반 학회로 정식 proceedings에 DOI를 부여하지 않음)
- 공개 여부: **공개** — arXiv 프리프린트로 무료 열람 가능, arXiv:1409.1556, <https://arxiv.org/abs/1409.1556>

### **Deep Residual Learning for Image Recognition(이미지 인식을 위한 깊은 잔차 학습)**

> 그런데 연구자들이 VGGNet 구조를 더 깊게 확장하자 새로운 문제가 드러났다. 20층 모델과 56층 모델을 비교했더니 오히려 56층 모델의 성능이 훈련 데이터셋에서조차 더 낮게 나타난 것이다. 단순히 신경망을 깊게 쌓는 것만으로는 깊이의 이점을 살릴 수 없기 때문이었다. 이 문제는 2015년 마이크로소프트 연구소의 카이밍 허Kaiming He 등이 개발한 ResNet이 해결해 그해 이미지 인식 대회를 휩쓸었다.
>
> — 본문 300쪽

- 저자: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- 출처: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778 (2016)
- DOI: <https://doi.org/10.1109/CVPR.2016.90>
- 공개 여부: 비공개 — IEEE Xplore 유료 접근. 다만 arXiv 프리프린트로는 무료 열람 가능, arXiv:1512.03385, <https://arxiv.org/abs/1512.03385>

### **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift(배치 정규화: 내부 공변량 이동을 줄여 깊은 신경망 학습 가속화)**

> 배치 정규화는 2015년 구글의 세르게이 이오페Sergey Ioffe와 크리스천 세게디Christian Szegedy가 발표했다.
>
> — 본문 311쪽

- 저자: Sergey Ioffe, Christian Szegedy
- 출처: Proceedings of the 32nd International Conference on Machine Learning (ICML), PMLR, Vol. 37, pp. 448–456 (2015)
- DOI: 없음 (PMLR 게재 논문으로 DOI가 부여되지 않음)
- 공개 여부: **공개** — PMLR 공식 사이트에서 전문 PDF 무료 제공, <https://proceedings.mlr.press/v37/ioffe15.html>

---

## 9장 가변 길이를 다루는 Seq2Seq와 어텐션

### **Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation(RNN 인코더-디코더를 이용한 통계적 기계 번역의 구절 표현 학습)**

→ 6장의 같은 항목을 참고한다.

### **Sequence to Sequence Learning with Neural Networks(신경망을 이용한 시퀀스 투 시퀀스 학습)**

> 2014년 6월, 몬트리올 대학의 조경현 등은 단순 RNN으로 만든 두 순환 신경망을 각각 인코더와 디코더로 나란히 배치한 모델을 발표했고, 9월에 서츠케버 등은 거의 같은 구조의 LSTM을 사용하는 모델을 발표했다.
>
> — 본문 341쪽

- 저자: Ilya Sutskever, Oriol Vinyals, Quoc V. Le
- 출처: Advances in Neural Information Processing Systems (NeurIPS), Vol. 27, pp. 3104–3112 (2014)
- DOI: 없음 (NeurIPS proceedings 원본 논문에는 DOI가 부여되어 있지 않음)
- 공개 여부: **공개** — NeurIPS 공식 proceedings 페이지에서 전문 PDF 무료 제공, <https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf>

### **Neural Machine Translation by Jointly Learning to Align and Translate(정렬과 번역을 함께 학습하는 신경망 기계 번역)**

> 곧이어 드미트리 바다나우Dzmitry Bahdanau와 조경현 등은 이 한계를 해결하는 새로운 방법인 어텐션 메커니즘을 고안했다. 이 방법은 매 출력 토큰을 생성할 때마다 입력 순차 데이터의 어느 부분에 집중할지를 모델이 스스로 결정하게 만드는 방식이었다.
>
> — 본문 342쪽

- 저자: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- 출처: International Conference on Learning Representations (ICLR) 2015
- DOI: 없음 (ICLR은 오픈 리뷰(OpenReview) 기반 학회로 정식 proceedings에 DOI를 부여하지 않음)
- 공개 여부: **공개** — arXiv 프리프린트로 무료 열람 가능, arXiv:1409.0473, <https://arxiv.org/abs/1409.0473>

### **Attention Is All You Need(어텐션이 필요한 전부다)**

> 9-3절에서 소개한 바다나우 어텐션 외에도 다양한 어텐션 메커니즘이 있다. 그중 크기 조정 내적 어텐션(scaled dot-product attention)은 바다나우 어텐션과 함께 가장 널리 사용되는 방식으로, 대부분의 언어 모델의 핵심 구조인 트랜스포머는 크기 조정 내적 어텐션을 사용해 동작한다.
>
> — 본문 386쪽

- 저자: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- 출처: Advances in Neural Information Processing Systems (NeurIPS), Vol. 30, pp. 5998–6008 (2017)
- DOI: 없음 (NeurIPS proceedings 원본 논문에는 DOI가 부여되어 있지 않음)
- 공개 여부: **공개** — NeurIPS 공식 proceedings 페이지 및 arXiv에서 무료 열람 가능, arXiv:1706.03762, <https://arxiv.org/abs/1706.03762>

### **Attention Is Not Explanation(어텐션은 설명이 아니다)**

> 대표적인 회의론은 Jain & Wallace (2019)의 <Attention is not Explanation(어텐션은 설명이 아니다)>이며, 이에 대한 반론으로 Wiegreffe & Pinter (2019)의 <Attention is not not Explanation(어텐션은 설명이 아닌 게 아니다)>도 함께 읽어 볼 만하다.
>
> — 본문 388쪽(학습노트)

- 저자: Sarthak Jain, Byron C. Wallace
- 출처: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Vol. 1, pp. 3543–3556 (2019)
- DOI: <https://doi.org/10.18653/v1/N19-1357>
- 공개 여부: **공개** — ACL Anthology, <https://aclanthology.org/N19-1357/>

### **Attention Is Not Not Explanation(어텐션은 설명이 아닌 게 아니다)**

> 대표적인 회의론은 Jain & Wallace (2019)의 <Attention is not Explanation(어텐션은 설명이 아니다)>이며, 이에 대한 반론으로 Wiegreffe & Pinter (2019)의 <Attention is not not Explanation(어텐션은 설명이 아닌 게 아니다)>도 함께 읽어 볼 만하다.
>
> — 본문 388쪽(학습노트)

- 저자: Sarah Wiegreffe, Yuval Pinter
- 출처: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 11–20 (2019)
- DOI: <https://doi.org/10.18653/v1/D19-1002>
- 공개 여부: **공개** — ACL Anthology, <https://aclanthology.org/D19-1002/>

---

## 10장 어텐션만으로 만든 신경망, 트랜스포머

### **Attention Is All You Need(어텐션이 필요한 전부다)**

> 2017년 구글 브레인의 아시시 바스와니Ashish Vaswani와 동료 일곱 명은 <Attention Is All You Need(어텐션만으로
> 충분하다)>라는 다소 도발적인 제목의 논문을 발표했다.
>
> — 본문 389쪽

- 저자: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- 출처: Advances in Neural Information Processing Systems (NeurIPS), Vol. 30, pp. 5998–6008 (2017)
- DOI: 없음 (NeurIPS proceedings 원본 논문에는 DOI가 부여되어 있지 않음)
- 공개 여부: **공개** — NeurIPS 공식 proceedings 페이지 및 arXiv에서 무료 열람 가능, arXiv:1706.03762, <https://arxiv.org/abs/1706.03762>

### **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(BERT: 언어 이해를 위한 깊은 양방향 트랜스포머의 사전 학습)**

> 인코더만 사용하는 모델의 대표 사례는 2018년 구글이 발표한 BERTBidirectional Encoder Representations from Transformers이다.
>
> — 본문 410쪽

- 저자: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- 출처: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Vol. 1, pp. 4171–4186 (2019)
- DOI: <https://doi.org/10.18653/v1/N19-1423>
- 공개 여부: **공개** — ACL Anthology, <https://aclanthology.org/N19-1423/>

### **Improving Language Understanding by Generative Pre-Training(생성적 사전 학습을 통한 언어 이해 개선)**

> 2018년 등장한 GPT-1 모델은 두 가지 사실을 입증했다.
>
> — 본문 420쪽

- 저자: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
- 출처: OpenAI Technical Report (2018)
- DOI: 없음 (정식 학술지·학회 논문이 아닌 OpenAI 자체 기술 보고서)
- 공개 여부: **공개** — OpenAI 공식 사이트에서 PDF 무료 제공, <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>

---

## 11장 이미지를 만드는 생성 전략

### **Auto-Encoding Variational Bayes(변분 베이즈 오토인코딩)**

> 2013년 암스테르담 대학교의 디더릭 킹마Diederik Kingma와 막스 벨링Max Welling은 이 문제를 해결하기 위한 새로운 아이디어를 제안했다. 핵심은 잠재 벡터를 ‘고정된 한 점’이 아닌 ‘일정한 범위 안의 한 점’으로 인코딩하자는 것이다.
>
> — 본문 436쪽

- 저자: Diederik P. Kingma, Max Welling
- 출처: International Conference on Learning Representations (ICLR) 2014 (arXiv 최초 게재: 2013)
- DOI: 없음 (ICLR은 오픈리뷰 기반 학회로 정식 proceedings에 DOI를 부여하지 않음)
- 공개 여부: **공개** — arXiv 프리프린트로 무료 열람 가능, arXiv:1312.6114, <https://arxiv.org/abs/1312.6114>

### **Generative Adversarial Networks(생성적 적대 신경망)**

> 2014년 몬트리올 대학교의 연구자 이안 굿펠로Ian Goodfellow는 가짜 이미지를 만드는 생성자generator와 이를 식별하는 판별자discriminator라는 두 신경망을 서로 경쟁시켜 보자는 아이디어를 제안했다.
>
> — 본문 449쪽

- 저자: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- 출처: Advances in Neural Information Processing Systems (NeurIPS), Vol. 27, pp. 2672–2680 (2014)
- DOI: 없음 (NeurIPS proceedings 원본 논문에는 DOI가 부여되어 있지 않음)
- 공개 여부: **공개** — NeurIPS 공식 proceedings 페이지 및 arXiv에서 무료 열람 가능, arXiv:1406.2661, <https://arxiv.org/abs/1406.2661>

### **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks(깊은 합성곱 생성적 적대 신경망을 이용한 비지도 표현 학습)**

> 이번에는 11-1절의 의류 이미지 생성 모델(모델 14. 의류 이미지 생성 모델)을 VAE 대신 DCGAN으로 만들어 본다.
>
> — 본문 452쪽

- 저자: Alec Radford, Luke Metz, Soumith Chintala
- 출처: International Conference on Learning Representations (ICLR) 2016 (arXiv 최초 게재: 2015)
- DOI: 없음 (ICLR은 오픈리뷰 기반 학회로 정식 proceedings에 DOI를 부여하지 않음)
- 공개 여부: **공개** — arXiv 프리프린트로 무료 열람 가능, arXiv:1511.06434, <https://arxiv.org/abs/1511.06434>

### **Image Style Transfer Using Convolutional Neural Networks(합성곱 신경망을 이용한 이미지 스타일 전이)**

> 독일 튀빙겐 거리 풍경에 <별이 빛나는 밤>의 화풍을 합성한, 딥러닝 역사에서 가장 유명한 이미지 중 하나가 바로 이 방법으로 만들어졌다.
>
> — 본문 463쪽

- 저자: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
- 출처: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2414–2423 (2016)
- DOI: <https://doi.org/10.1109/CVPR.2016.265>
- 공개 여부: 비공개 — IEEE Xplore 유료 접근. 다만 CVF Open Access 페이지에서 전문 무료 제공, <https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html>

### **Denoising Diffusion Probabilistic Models(디노이징 확산 확률 모델)**

> 2020년 이후의 이미지 생성 연구를 살펴보면 또 하나의 거대한 흐름이 등장한다. 바로 확산 모델(diffusion model)이다. 확산 모델은 텍스트만 입력하면 사진처럼 정교한 이미지를 만들어 내는 DALL·E 2, Imagen, Stable Diffusion 등의 핵심 구성 요소로 사용된다.
>
> — 본문 462쪽

- 저자: Jonathan Ho, Ajay Jain, Pieter Abbeel
- 출처: Advances in Neural Information Processing Systems (NeurIPS), Vol. 33, pp. 6840–6851 (2020)
- DOI: 없음 (NeurIPS proceedings 원본 논문에는 DOI가 부여되어 있지 않음)
- 공개 여부: **공개** — NeurIPS 공식 proceedings 페이지 및 arXiv에서 무료 열람 가능, arXiv:2006.11239, <https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html>

### **High-Resolution Image Synthesis with Latent Diffusion Models(잠재 확산 모델을 이용한 고해상도 이미지 합성)**

> 2020년 이후의 이미지 생성 연구를 살펴보면 또 하나의 거대한 흐름이 등장한다. 바로 확산 모델(diffusion model)이다. 확산 모델은 텍스트만 입력하면 사진처럼 정교한 이미지를 만들어 내는 DALL·E 2, Imagen, Stable Diffusion 등의 핵심 구성 요소로 사용된다.
>
> — 본문 462쪽

- 저자: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
- 출처: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695 (2022)
- DOI: 없음 (CVF Open Access 페이지에서 정식 DOI 없이 게재)
- 공개 여부: **공개** — CVF Open Access, <https://openaccess.thecvf.com>, arXiv:2112.10752
- 특이사항: 본문에 논문 제목이 직접 언급되지는 않는다. 인용문에서 확산 모델의 사례로 든 Stable Diffusion이 이 논문에서 제안한 잠재 확산 모델을 바탕으로 만들어진 모델이다. 같은 인용문을 근거로 하는 <Denoising Diffusion Probabilistic Models> 항목과 함께 보면 좋다.

---

## 13장 언어 밖 진짜 세계의 모델, 멀티모달

### **Learning Transferable Visual Models From Natural Language Supervision(자연어 지도를 통한 전이 가능한 시각 모델 학습)**

> 하나는 두 모달의 임베딩 공간을 사전에 정렬해 두는 방법으로, OpenAI의 CLIP 모델이 대표적이다. CLIP 모델은 이미지 인코더와 텍스트 인코더로 구성된다. 하지만 두 인코더를 따로 학습하는 대신, 이미지와 텍스트 4억 쌍을 대조 학습contrastive learning 방식으로 학습했다.
>
> — 본문 514쪽

- 저자: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
- 출처: Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR, Vol. 139, pp. 8748–8763 (2021)
- DOI: 없음 (PMLR 게재 논문으로 DOI가 부여되지 않음)
- 공개 여부: **공개** — PMLR 공식 사이트에서 전문 PDF 무료 제공, <https://proceedings.mlr.press/v139/radford21a.html>, arXiv:2103.00020

### **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models(BLIP-2: 고정된 이미지 인코더와 대규모 언어 모델을 이용한 언어-이미지 사전 학습의 부트스트래핑)**

> 이번 절에서는 두 번째 방법인 ‘똑똑한 프로젝션 계층’을 BLIP-2 모델로 소개한다.
>
> — 본문 527쪽

- 저자: Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi
- 출처: Proceedings of the 40th International Conference on Machine Learning (ICML), PMLR, Vol. 202, pp. 19730–19742 (2023)
- DOI: 없음 (PMLR 게재 논문으로 DOI가 부여되지 않음)
- 공개 여부: **공개** — PMLR 공식 사이트에서 전문 PDF 무료 제공, <https://proceedings.mlr.press/v202/li23q.html>, arXiv:2301.12597
