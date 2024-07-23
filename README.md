# Fake Audio Challenge 2024

## 설치 방법

가상환경 설정

```bash
# 가상환경 설치
virtualenv venv
. venv/bin/activate
cd espnet/tools
./setup_python.sh $(which python3)

# python 패키지 설치(ESPnet 요구사항)
cd ..
pip install -e .

# python 패키지 설치(프로젝트 요구사항)
cd ..
pip install -r requirements.txt 

# ESPnet 컴파일
cd espnet/tools
make -j
```



## 학습 데이터셋 구축

### 음성 분할

대회측에서 공개한 unlabeled_data에서 노이즈를 추출하기 위해, 화자 분리 및 음성 분할 기술을 이용하여 데이터를 분리함

#### 코드 실행 방법

```
python 01...
```



### 노이즈 추출

음성 분할된 데이터에서 음성이 포함되지 않은 파일을 추출하여 노이즈로 사용함

```bash
python 02_1_noise_detection_silero.py
```

```bash
python 02_2_remove_silence.py
```

```bash
python 02_3_noise_detection_marblenet.py
```



### 노이즈 + 음성 데이터 생성

추출한 노이즈와  train 데이터를 이용하여 real + noise, fake + noise를 생성함

