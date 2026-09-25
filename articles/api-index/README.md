# 파이토치 API 찾아보기 — ABC순

- <직접 구현하는 딥러닝 with 파이토치> 본문과 예제 코드에 등장하는 클래스, 함수, 메서드를
이름순으로 모은 색인표
- 정렬 기준은 실제 이름 기준
  - 예를 들어 `optim.SGD`는 S, `nn.Linear`는 L에 위치함
- 패키지 접두사 기준으로 정리한 표는 [접두사별 목록](API별_설명.md)에 수록


---

| | API | 절 |
|---|---|---|
| **A** | `optim.Adam` | 2-3, 3-2 |
|  | `optim.AdamW` | 2-3 |
|  | `nn.AdaptiveAvgPool2d` | 8-3 |
|  | `nn.AdaptiveMaxPool2d` | 2-3 |
|  | `Module.add_module` | 3-3 |
|  | `Sequential.append` | 3-3 |
|  | `Module.apply` | 11-2 |
|  | `tokenizer.apply_chat_template` | 12-1 |
|  | `torch.arange` | 1-1 |
|  | `Tensor.argmax` | 3-3 |
|  | `torch.argmax` | 8-4, 10-3 |
|  | `np.array` | 1-1 |
|  | `AutoModel` | 12-2 |
|  | `AutoModelForCausalLM` | 12-1 |
|  | `AutoModelForMaskedLM` | 12-1 |
|  | `AutoModelForQuestionAnswering` | 12-1 |
|  | `AutoModelForSeq2SeqLM` | 12-2 |
|  | `AutoModelForSequenceClassification` | 12-1 |
|  | `AutoModelForTokenClassification` | 12-1 |
|  | `AutoTokenizer` | 12-1 |
|  | `nn.AvgPool2d` | 5-2 |
| **B** | `Tensor.backward` | 1-2 |
|  | `nn.BatchNorm1d` | 8-2 |
|  | `nn.BatchNorm2d` | 8-2 |
|  | `nn.BCELoss` | 11-2 |
|  | `torch.bfloat16` | 12장 도입 |
|  | `BitsAndBytesConfig` | 12-3 |
|  | `Blip2ForConditionalGeneration` | 13-2 |
|  | `Blip2Processor` | 13-2 |
|  | `torch.bmm` | 9-3 |
|  | `torch.bool` | 1-1 |
| **C** | `torch.cat` | 1-1, 10-1 |
|  | `transforms.CenterCrop` | 4-3 |
|  | `datasets.CIFAR10` | 5-3 |
|  | `Tensor.clamp` | 1-1 |
|  | `nn.utils.clip_grad_norm_` | 6-3 |
|  | `CLIPModel` | 13-1 |
|  | `CLIPProcessor` | 13-1 |
|  | `Tensor.clone` | 1-1 |
|  | `transforms.ColorJitter` | 4-3 |
|  | `transforms.Compose` | 4-3 |
|  | `ConcatDataset` | 4-2 |
|  | `nn.init.constant_` | 11-2 |
|  | `nn.Conv2d` | 5-2 |
|  | `tokenizer.convert_tokens_to_ids` | 12-1 |
|  | `nn.ConvTranspose2d` | 7-3, 11-1, 11-2 |
|  | `Tensor.cpu` | 9-2 |
|  | `client.chat.completions.create` | 12-4 |
|  | `timm.create_model` | 8-4 |
|  | `timm.data.create_transform` | 8-4 |
|  | `nn.CrossEntropyLoss` | 3-3 |
| **D** | `DataCollatorForSeq2Seq` | 12-2 |
|  | `DataLoader` | 4-2 |
|  | `Dataset` | 4-2 |
|  | `Dataset` (HF) | 12-2 |
|  | `DatasetDict` | 12-2 |
|  | `tokenizer.decode` | 12-1 |
|  | `copy.deepcopy` | 4-1, 7-4 |
|  | `Tensor.detach` | 1-2 |
|  | `Tensor.detach_` | 1-2 |
|  | `torch.device` | 2-3, 5-3 |
|  | `csv.DictReader` | 3-3 |
|  | `Tensor.dim` | 1-1 |
|  | `torch.double` | 1-1 |
|  | `nn.Dropout` | 5-3 |
| **E** | `nn.Embedding` | 7-2 |
|  | `SentenceTransformer.encode` | 12-4 |
|  | `Module.eval` | 2-3 |
|  | `torch.exp` | 11-1 |
|  | `Tensor.expand` | 9-3 |
| **F** | `datasets.FashionMNIST` | 11-1 |
|  | `nn.Flatten` | 4-3 |
|  | `Tensor.flatten` | 1-1 |
|  | `Tensor.float` | 3-3 |
|  | `torch.float` | 1-1, 1-2 |
|  | `torch.float16` | 12장 도입 |
|  | `torch.float32` | 1-1 |
|  | `torch.float64` | 1-1 |
|  | `Module.forward` | 2-3 |
|  | `*.from_pretrained` (클래스 메서드) | 12-1 |
|  | `torch.full` | 13-1 |
| **G** | `nn.GELU` | 13-1 |
|  | `model.generate` | 12-1 |
|  | `torch.Generator` | 4-2 |
|  | `requests.get` | 8-4 |
|  | `model.get_image_features` (CLIP) | 13-1 |
|  | `GPT2LMHeadModel` | 13-1 |
|  | `GPT2Tokenizer` | 13-1 |
|  | `transforms.Grayscale` | 4-3 |
|  | `Groq` | 12-4 |
|  | `nn.GRU` | 6-3 |
| **I** | `nn.Identity` | 8-3, 13-1 |
|  | `Image.Image` (자료형) | 4-3 |
|  | `datasets.ImageFolder` | 8-4 |
|  | `Sequential.insert` | 3-3 |
|  | `torch.int32` | 4-2 |
|  | `np.int64` | 1-1 |
|  | `torch.int64` | 4-2 |
|  | `torch.int8` | 12장 도입 |
|  | `torch.backends.mps.is_available` | 5-3 |
|  | `torch.cuda.is_available` | 5-3 |
|  | `torch.xpu.is_available` | 5-3 |
|  | `Tensor.item` | 1-2 |
| **L** | `nn.L1Loss` | 7-3 |
|  | `nn.LayerNorm` | 13-1 |
|  | `nn.LeakyReLU` | 11-2 |
|  | `nn.Linear` | 2-3 |
|  | `torch.linspace` | 1-1 |
|  | `torch.load` | 4-1 |
|  | `load_dataset` | 12-2, 13-1 |
|  | `Module.load_state_dict` | 4-1, 8-4 |
|  | `F.log_softmax` | 10-3 |
|  | `Tensor.long` | 7-2 |
|  | `torch.long` | 1-1 |
|  | `nn.LSTM` | 6-3 |
| **M** | `torch.manual_seed` | 1-3, 4-2 |
|  | `Dataset.map` | 12-2 |
|  | `Tensor.masked_fill` | 9-3 |
|  | `torch.matmul` | 2-2, 9-3 |
|  | `Tensor.max` | 1-2, 3-3 |
|  | `torch.max` | 1-2 |
|  | `nn.MaxPool2d` | 5-2, 8-1 |
|  | `Tensor.mean` | 1-3 |
|  | `torch.mean` | 1-3 |
|  | `torch.cuda.memory_allocated` | 12-3 |
|  | `Tensor.min` | 1-2 |
|  | `torch.min` | 1-2 |
|  | `datasets.MNIST` | 4-3 |
|  | `nn.Module` | 2-3 |
|  | `nn.MSELoss` | 2-3, 3-3 |
|  | `nn.MultiheadAttention` | 9-3 |
|  | `torch.multinomial` | 10-3 |
| **N** | `nn.NLLLoss` | 9-2 |
|  | `torch.no_grad` | 1-2, 5-3 |
|  | `Tensor.norm` | 8-2 |
|  | `torch.linalg.norm` | 8-2 |
|  | `nn.init.normal_` | 11-2 |
|  | `F.normalize` | 12-4 |
|  | `transforms.Normalize` | 4-3, 7-3 |
|  | `Tensor.numel` | 1-1 |
|  | `Tensor.numpy` | 1-2 |
| **O** | `F.one_hot` | 3-3 |
|  | `torch.ones` | 1-1 |
|  | `Image.open` | 5-3 |
|  | `collections.OrderedDict` | 3-3 |
| **P** | `nn.utils.rnn.pack_padded_sequence` | 9-2 |
|  | `nn.utils.rnn.pad_packed_sequence` | 9-2 |
|  | `nn.utils.rnn.pad_sequence` | 9-2 |
|  | `nn.Parameter` | 13-2 |
|  | `Module.parameters` | 2-3 |
|  | `Tensor.permute` | 1-1 |
|  | `pipeline` | 12-1 |
|  | `Sequential.pop` | 3-3 |
|  | `Tensor.pow` | 11-1 |
| **R** | `torch.rand` | 1-1 |
|  | `torch.randint` | 1-1 |
|  | `torch.randn` | 1-1, 2-2 |
|  | `torch.randn_like` | 8-3 |
|  | `random_split` | 4-2 |
|  | `transforms.RandomCrop` | 4-3 |
|  | `transforms.RandomHorizontalFlip` | 5-3 |
|  | `nn.ReLU` | 3-3 |
|  | `Tensor.repeat` | 9-3 |
|  | `Tensor.requires_grad_` | 1-2 |
|  | `Tensor.reshape` | 1-1, 9-1 |
|  | `transforms.Resize` | 4-3 |
|  | `models.resnet50` | 8-4 |
|  | `models.ResNet50_Weights` | 13-1 |
|  | `timm.data.resolve_data_config` | 8-4 |
|  | `nn.RNN` | 6-1, 6-2 |
| **S** | `torch.save` | 4-1 |
|  | `timm.layers.SelectAdaptivePool2d` | 8-4 |
|  | `SentenceTransformer` | 12-4 |
|  | `Seq2SeqTrainer` | 12-2 |
|  | `Seq2SeqTrainingArguments` | 12-2 |
|  | `nn.Sequential` | 3-3 |
|  | `optim.SGD` | 2-3, 3-2 |
|  | `F.sigmoid` | 3-3 |
|  | `nn.Sigmoid` | 2-2, 2-3 |
|  | `torch.sigmoid` | 2-2 |
|  | `Tensor.size` | 1-1, 11-1 |
|  | `torch.Size` | 1-1 |
|  | `F.softmax` | 3-3 |
|  | `nn.Softmax` | 3-3 |
|  | `torch.softmax` | 9-3 |
|  | `Tensor.squeeze` | 1-1 |
|  | `torch.stack` | 1-1 |
|  | `Module.state_dict` | 4-1 |
|  | `Tensor.std` | 1-2 |
|  | `Optimizer.step` | 2-3 |
|  | `Tensor.sum` | 1-2, 11-1 |
|  | `torch.sum` | 11-1 |
|  | `torchinfo.summary` | 2-3, 5-2 |
| **T** | `Tensor.t` | 1-1 |
|  | `nn.Tanh` | 10-1 |
|  | `torch.tanh` | 9-3 |
|  | `torch.tensor` | 1-1 |
|  | `Module.to` | 1-1, 5-3 |
|  | `Tensor.to` | 1-1, 5-3 |
|  | `tokenizer.tokenize` | 12-1 |
|  | `Tensor.tolist` | 1-2 |
|  | `Tensor.topk` | 8-4 |
|  | `torch.topk` | 8-4 |
|  | `transforms.ToTensor` | 4-3, 7-3 |
|  | `Module.train` | 1-3, 2-3 |
|  | `Trainer.train` | 12-2 |
|  | `Trainer` | 12-2 |
|  | `nn.Transformer` | 10-1 |
|  | `nn.TransformerDecoder` | 10-1 |
|  | `nn.TransformerEncoder` | 10-1, 10-2 |
|  | `nn.TransformerEncoderLayer` | 10-2 |
|  | `Tensor.transpose` | 1-1 |
| **U** | `nn.Unflatten` | 2-3 |
|  | `Tensor.unsqueeze` | 1-1, 9-3 |
|  | `nn.Upsample` | 7-3 |
| **V** | `Tensor.view` | 1-1 |
| **Z** | `Tensor.zero_` | 2-2 |
|  | `Module.zero_grad` | 2-3 |
|  | `Optimizer.zero_grad` | 2-3 |
|  | `torch.zeros` | 1-1 |

