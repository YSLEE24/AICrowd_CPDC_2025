# 🧠 Prompt Optimization with LLMs – AICROWD API Track

## 🏁 프로젝트 개요

이 프로젝트는 AICROWD에서 진행된 **Prompt Optimization API 트랙**에서 수행되었으며,  
주어진 LLM API 응답 데이터셋을 분석하여 **가장 우수한(Gold) 응답을 유도하는 프롬프트 조합을 찾는** 것을 목표로 합니다.

## 🥈 수상 성과

> 🥈 **2nd Place Winner @ AICROWD API Track**  
> 전략적 프롬프트 튜닝과 고정밀 선택 로직을 통해 상위권 입상

---

## 👩‍💻 수행 기간

2025.06 ~ 2025.07 (개인 프로젝트)

---

## 📌 문제 정의

- 각 문항(prompt)에 대해 여러 LLM 응답 후보군이 주어짐  
- 그중 **Gold 응답(GT)**에 가장 가까운 응답을 자동으로 식별해야 함  
- 단순한 분류가 아닌 **의미 기반 선택**이 요구됨  
  (정답과의 유사도뿐 아니라 논리·태도 등까지 고려)

---

## 🧪 솔루션 개요

1. **임베딩 기반 의미 유사도 평가**
   - `sentence-transformers`의 `all-MiniLM-L6-v2` 모델 사용
   - 응답 전체를 벡터화하여 Gold 응답과 코사인 유사도 기반 비교

2. **프롬프트 분석 기반 가중치 조정**
   - Prompt의 특성(예: 정중한 응답 유도, 요약 요구 등)에 따라 가중치 적용
   - Negative Sampling으로 “나쁜 응답” 회피 로직 구현

3. **Threshold 기반 선택 로직 튜닝**
   - Top-K 후보 중 유사도가 일정 이상일 때만 선택
   - Overfitting 방지를 위해 fold별 최적 threshold 자동 튜닝

4. **Validation Split을 통한 Offline Score 개선**
   - 5-fold cross-validation 도입
   - 각 fold에서 best model 선택 후 앙상블 전략 사용

---

## 🧰 사용 기술

- Python
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Scikit-learn
- NumPy
- JSON 데이터 파싱
- 프롬프트 엔지니어링 전략 설계
- TQDM 등 유틸리티

---

## 🎯 주요 성과

- 평균 정확도 기준 **🏆 AICROWD Prompt Optimization API 트랙 2위 (19개 팀 중 상위 11%) 달성**
- 프롬프트 유형에 따른 **성능 변화 분석 리포트** 작성
- 향후 확장 가능성 (예: RAG 기반 사전지식 반영 프롬프트 튜닝 등) 확인

---

## 🗂️ 폴더 구조 (예시)

```
├── main.py                # 메인 실행 파일
├── model_utils.py         # 유사도 계산 및 튜닝 함수
├── data/
│   ├── train.json
│   └── val.json
├── outputs/
│   └── best_predictions.json
├── README.md              # 📄 바로 이 파일!
```

---

## 💡 개선 아이디어

- GPT-4 기반 응답 재생성 후 재선택 (Self-Rerank)
- 응답 내 태도·논리 평가 항목화
- 프롬프트 자동 생성기와 연동한 튜닝 자동화 파이프라인

---

🏆 **Thanks to AICROWD for hosting a creative and challenging competition!**
