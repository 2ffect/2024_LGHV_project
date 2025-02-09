# HV_project
## - AI 기반 셋톱박스 장애 사전 감지 및 경고 시스템 -

## 📁Repository 구조
```
Display_SensorData
├── 📁src 
|	├── 📁 1. 데이터 탐색 및 전처리
|	|	├── 📃EDA&preprocessing_df.ipynb
|	├── 📁 2. 데이터 시각화
|	|	├── 📃Visualization.ipynb
|	├── 📁 3. 모델링 수행 및 비교
|	|	 ├── 📃train.jpynb
|	└── 📁 4. 모델을 통한 예측결과 확인
|                └── 📃prediction_temp.jpynb
├── 📁img (README.md 관련 이미지) 
└── 📁old_code (시행착오 소스코드)
```

## 목차


  * [분석 환경 및 도구]
  * [프로젝트 개요]
	  * [프로젝트 목적]
	  * [프로젝트 배경 설명]
  * [프로젝트 수행내용 요약]
      * [데이터 탐색]
      * [데이터 전처리]
      * [모델 학습]
      * [분류 모델 수행]
  * [프로젝트 결론]





## 💻분석 환경 및 도구

```
- HW/Server
	- Window (ver 11 / AMD Ryzen 7 7730U / RAM 16)
	- Amazon Linux (ver 2023)
- Language
	- Python (ver 3.11.5)
- Tools
	- Docker (ver 25.0.3)
	- Github
	- AWS
	- Slack
	- Notion
	- JIRA
- IDE
	- Jupyter Notebook (ver 7.0.8)
- Analysis Library
	- Pandas (ver 2.2.2)
	- Numpy (ver 1.26.4)
	- Sklearn (ver 1.5.0)
	- Matplotlib (ver 1.3.0)
	- Seaborn (ver 0.13.2)
	- shap (ver 0.45.1)
```


## 🌿 프로젝트 개요
### 프로젝트 목적

- 셋탑박스 데이터 분석을 통해 장애가 발생하기 전 장애 예측
- Flask를 활용하여 셋탑박스 데이터를 바탕으로 장애가 발생할 시 알림을 주는 웹 개발
- 장애가 발생하기 전 사전 탐지 / 신속한 유지 보수를 통해 고객 만족도 향상을 목적



<details>
<summary> 프로젝트 배경 설명📌(펼치기)</summary>
<div markdown="1">       

- 과학기술정보통신부의 유료방송서비스 품질평가 결과에 따르면 LG헬로비전의 경우 2022년 이후 이상화면 발생 빈도가 증가하는 추세. 2022년에는 이상발생 비율이 평균 이하였지만 2023년에는 평균이상을 기록, 전년도에 비해 3배 증가.
- 이에 따라 이용자의 만족도도 2022년 대비 2023년에 감소와 더불어 평균 이하 기록.

</div>
</details>



## 🌿 프로젝트 수행내용 요약

- 데이터 : 8 columns * 2708818 rows  시계열 셋탑박스 데이터, 3 columns * 472 rows 장애발생 데이터
데이터 탐색(EDA) 및 전처리
Machine Learning을 이용한 Classification 수행
- XGBoost, Logistic Regression, SVM, Random Forest, CatBoost 이용
- ROC AUC 커브를 통해 모델 평가
- OOF기법을 사용하여 모델 성능 향상
대시보드 구현
- Python을 이용
- 셋탑박스 데이터 모니터링을 통해 각 피처값의 시계열 그래프 확인
- 경고가 온 셋탑박스의 이전 하루치의 데이터 값을 불러오는 대시보드 확인
알림 서비스 개발
- flask를 활용해서 임계값 이상의 확률 감지시 관리자 메일로 알림 발생

<details>
<summary> 프로젝트 상세 수행내용📌(펼치기)</summary>
<div markdown="1">    

## 📌데이터 탐색
- 데이터 : hellovision 4/1 ~ 5/1까지의 셋탑박스 데이터, 해당 기간 중 장애 발생 데이터 (csv 파일)
- 8 columns * 2708818 rows, 3 columns * 472 rows 데이터임
- 동일한 시간에 같은 셋탑박스의 데이터가 여러 번 측정되는 경우가 존재함
	- 온오프라인여부 컬럼은 offline을 무조건적으로 우선시 해야함
- 장애는 MAJOR, CRITICAL로 구분되어 있음

## 📌[데이터 전처리]
- 온오프라인여부 컬럼 소문자화
- 셀번호, 측정시간 NaN값인 경우 삭제
- 중복 데이터 처리
	- 중복되는 경우 분산확인 결과 대체로 작지만, 큰 경우 매우 큼
	- 100 넘어가는 경우 최빈값으로, 나머지는 평균으로 합쳐줌
- 장애발생을 1, 정상을 0으로 지정
- 장애내역 데이터는 시간단위가 초 단위로 측정이 되었지만 settop 데이터는 5분단위임 
	- settop데이터에서 가장 가까운 앞쪽 시간을 찾아 장애 발생지점으로 선정

<details>
<summary> 데이터 전처리 상세내용📌(펼치기)</summary>
<div markdown="1">   
    
    
## 📌[모델 학습]
- 수행 목표 및 방법
	- 비정상 예측을 위한 분류 학습 시행
- 수행 내용
    - lag을 통한 시계열 데이터 성질 추가
    - 추가 feature 사용
    - stratify를 통한 불균형 문제 해결
    - OOF 사용을 통한 모델 성능 향상


## 📌[분류 모델 수행]
### 1) 분류 모델의 이해
- 단일 모델 학습 방식
	- 단일 알고리즘으로 하나의 모델을 이용하여 분류함
	- 해당 알고리즘 : SVM, Decision Tree
- Ensemble 모델 학습 방식
	- (1) Voting
	    - **여러 알고리즘**으로 모델을 생성하고 분류 결과를 비교하여 가장 좋은 모델을 선정하는 방법
        - voting 유형
        	- hard voting : voting 결과를 1, 0으로 리턴
	        - soft voting : voting 결과를 확률로 리턴

	- (2) Bagging
	    - **한 가지 알고리즘**으로 여러 개의 모델 생성하여 병렬 학습함
	    - 각 모델은 데이터 샘플링을 달리하여 비교함
	    - 해당 알고리즘 : Random Forest

	- (3) Boosting
	    - 여러 모델이 **순차적으로 학습함**
	    - 이전 모델이 잘못 분류한 데이터에 **가중치**를 부여하고 다음 모델 훈련에 적용함
	    - 해당 알고리즘 : Ada Boost, GBM, XGBoost, LightBoost

	- (4) Stacking
	    - 이전 모델 훈련 결과로 나온 예측값으로 다음 모델(메타모델)을 훈련함

### 2) 사용 모델
XGBoost : 높은 예측 성능과 빠른 학습 속도를 제공하며, 병렬 처리와 정규화 기능을 통해 과적합을 방지
- 피처 중요도를 제공하여 모델 해석이 용이하기 때문에 사용

Logistic Regression : 이해하기 쉽고 구현이 간단하며, 확률 기반 예측을 제공하고, 선형 분리 가능성 가정 하에서 효율적으로 동작
- 계산이 간단하여 빠르게 학습할 수 있는 장점

SVM : 고차원 데이터와 비선형 데이터를 처리하는 데 유리하며, 메모리의 효율적인 상용이 가능하고 작은 데이터셋에서도 높은 성능을 발휘.

random forest : 랜덤 포레스트는 높은 정확도와 과적합 방지, 다양한 데이터 처리 가능-결측치 처리와 병렬 처리, 안정성과 범용성을 제공하기 때문에 분류 문제에서 효과적으로 사용

CatBoost : 카테고리형 피처를 자동으로 처리하고, 적은 튜닝으로도 높은 예측 성능을 제공하며, 효율적인 알고리즘으로 빠른 학습 속도를 자랑
-실험을 통해 Learning_rate, n_estimators등의 하이퍼 파라미터 조절

### 모델 평가
1. 평가 지표 :　ROC AUC 커브
- 가지고 있는 데이터가 정상이 비정상에 비해 훨씬 많은 불균형 데이터이기 때문에 정확도로 성능을 평가하기에는 무리가 있다고 판단해서 이를 보완하기 위해 ROC AUC 커브 사용
- AUC기준 0.8 이상의 값 확보

2. 교차검증 기법인 OOF사용
- 학습 데이터를 늘리고 개수가 적은 비정상 데이터를 모두 활용한 테스트가 가능하기 때문에 교차검증 기법인 OOF사용

<details>
<summary>평가지표 배경지식📌(펼치기)</summary>
<div markdown="1">   

    - 정확도(accuracy) : TN + TP / 전체
    - 정밀도(precision) : TP / (FP + TP)
        - Pos로 예측한 것 중 실제 Pos였던 것
        - 양성예측도
        - Pos 예측 성능을 더 정밀하게 측정하기 위한 평가지표
        - FP를 낮추는 데 초점
    - 재현율(recall) : TP / (FN + TP)
        - 실제 Pos인 것 중 실제 Pos였던 것
        - 민감도, TPR(True Positive Rate)
        - Pos를 Neg로 판단하면 치명적인 경우 사용
        - FN을 낮추는 데 초점
    - F1 Score
        - 정밀도와 재현율의 조화평균
        - 두 평가지표를 적절히 고려하는 경우에 사용함
        -  2pr/(p+r)
    - ROC 곡선
        - 이진분류의 예측 성능 측정에 사용함
        - FP비율 - TP비율(recall) 곡선

## 🌿 프로젝트 결론

- 다른 분류 모델에 피해 ROC AUC 지표가 가장 우수한 XGBoost를 기준으로 비정상을 예측함
- OOF기법을 사용해 성능 향상
- 이 때 가장 높은 f1 score의 threshold를 찾아내고 기준으로 삼아 장애, 비장애 여부 판단


### 각 부분 별 기대효과
기술
- 셋탑박스의 특정 feature를 기반으로 장애예측이 가능한 시스템 개발
경제/산업
- 품질 경쟁력 있는 시스템으로 MSO 시장 선도
사회
- 고령층과 같은 정보취약계층에 도움이 되는 서비스 개발

### LGhellovision 기대 효과
기존 고객 이탈 방지 및 수익 손실 감소
- 사전 교체 시스템 도입 : AI 기반 장애 패턴 감지를 통한 조기 경보 솔루션을 제공해 이상이 감지된 셋톱박스를 사전에 교체하여, 고객의 불편 최소화

hellovison 사내 업무 효율 증대
- 실시간 시각화 대시보드 제공 : 현재 셋톱박스 상태를 실시간으로 시각화하여 직접 확인 가능
- 오류 시점의 빠른 파악 : 실시간 알림 서비스를 통해 오류 시점을 신속하게 파악 가능
- 사전 알림 시스템 구축 : 오류가 발생하기 전에 관리자에게 e-mail로 알림을 제공하여, 사전에 문제를 인지하고 대응 가능
<a href="#" class="btn--success" >