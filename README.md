# koreatech_finetuning_openLLM

## 프로젝트 개요
- **목적**: LLM(대규모 언어모델)의 파인튜닝 및 효율적 미세조정(PEFT, LoRA 등) 실습 및 연구
- **주요 모델**: NCSOFT/Llama-VARCO-8B-Instruct 기반, SFT 및 LoRA 적용
- **주요 프레임워크**: Huggingface Transformers, PEFT, TRL, Datasets 등

## 기술 스펙

### 주요 라이브러리
- python-dotenv, notebook, langgraph, langchain-openai, tavily-python, langchain_community, langchain-tavily, requests, streamlit, langgraph-supervisor, datasets, openai

### 환경 및 배포
- **Docker**: Python 3.11 기반, requirements.txt로 패키지 설치, GitHub에서 코드 클론
- **docker-compose**: LangGraph Studio(8000포트) 개발 환경 제공

### 모델 및 파인튜닝
- **Base Model**: NCSOFT/Llama-VARCO-8B-Instruct
- **Fine-tuned Model**: llama3-8b-news-analyzer-ko
- **적용 기술**: SFT(Supervised Fine-Tuning), LoRA(Low-Rank Adaptation), PEFT(Parameter-Efficient Fine-Tuning)
- **사용 라이브러리 버전**
	- PEFT: 0.17.1
	- TRL: 0.21.0
	- Transformers: 4.55.4
	- Pytorch: 2.8.0.dev20250319+cu128
	- Datasets: 4.0.0
	- Tokenizers: 0.21.4

## 폴더 구조
- codes/ : 데이터, 노트북, 문서, 모델 관련 파일
	- docs/ : 프리트레이닝, 파인튜닝, PEFT, LoRA, SFT 관련 설명 문서
		- 01.pretraining_finetuning.md : 프리트레이닝과 파인튜닝 개념 및 차이 설명
		- 02.peft_lora.md : PEFT와 LoRA 원리 및 장점 설명
		- 03.lora.md : LoRA 상세 원리 및 기존 방식 비교
		- 04.sft.md : SFT 개념 및 적용 방법
	- llama3-8b-news-analyzer-ko/ : 파인튜닝된 모델 및 설정 파일
		- config.json : 모델 설정 정보
		- adapter_config.json : LoRA/PEFT 어댑터 설정
		- README.md : 모델 설명 및 사용법
	- data/ : 파인튜닝 및 평가용 데이터셋
		- train.jsonl : 학습 데이터
		- eval.jsonl : 평가 데이터
	- notebooks/ : 실습 및 실험용 Jupyter 노트북
		- finetune_example.ipynb : 파인튜닝 실습 예제
		- lora_peft_example.ipynb : LoRA/PEFT 적용 예제
	- scripts/ : 데이터 처리 및 모델 학습 스크립트
		- preprocess.py : 데이터 전처리 스크립트
		- train.py : 모델 학습 스크립트
		- evaluate.py : 모델 평가 스크립트

- dockers/ : Dockerfile, docker-compose.yml
- requirements.txt : 프로젝트 의존성

## 주요 문서 요약
- **01.pretraining_finetuning.md** : 프리트레이닝과 파인튜닝의 개념, 차이, 적용 예시
- **02.peft_lora.md / 03.lora.md** : PEFT와 LoRA의 원리, 장점, 기존 방식과의 비교
- **04.sft.md** : SFT의 개념, 특징, 적용 방법

## 빠른 시작 예시
```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## 참고 및 인용
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- NCSOFT/Llama-VARCO-8B-Instruct
