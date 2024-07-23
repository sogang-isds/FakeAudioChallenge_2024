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

LibriMix 논문을 참고하여 노이즈와 음성을 합성하는 과정을 직접 수행함

### 음성 분할

대회측에서 공개한 unlabeled_data에서 노이즈를 추출하기 위해, 화자 분리 및 음성 분할 기술을 이용하여 데이터를 분리함

- 참고논문: [EEND-SS: Joint End-to-End Neural Speaker Diarization and Speech Separation for Flexible Number of Speakers](https://arxiv.org/abs/2203.17068)
- 사전학습 모델: https://huggingface.co/soumi-maiti/libri23mix_eend_ss
- 코드: https://github.com/espnet/espnet/tree/master/egs2/librimix/enh_diar1

#### 모델 다운로드

```bash
cd espnet/egs2/librimix/enh_diar1
mkdir models
cd models

git lfs install
git clone https://huggingface.co/soumi-maiti/libri23mix_eend_ss
```

#### **Step 1: 메타데이터 생성**

- 입력파일: data/sample_data/unlabeled_data에 있는 wav 파일
- 출력파일: /data/wav.scp

```bash
python 01_1_create_metadata.py
```

#### Step 2: 화자분리 및 음성 분할 실행

- 입력파일: /data/wav.scp 에 정의된 wav 파일 목록
- 출력파일: /data/diar_enh에 3명의 화자로 분할된 파일이 저장됨

```
bash 01_2_diarization_and_separation.sh
```



### 노이즈 추출

#### Step 1: Silero VAD를 사용한 음성 검출

음성 분할된 데이터에서 VAD를 이용하여 음성이 포함되지 않은 파일을 추출하여 노이즈로 사용함

- 사전학습 모델: https://github.com/snakers4/silero-vad/wiki/Version-history-and-Available-Models#available-models (v5.0 사용)
- 코드: https://github.com/snakers4/silero-vad

```bash
python 02_1_noise_detection_silero.py
```

#### Step 2: 묵음 구간이 많은 파일 제거

```bash
python 02_2_remove_silence.py
```

#### Step 3: Marblenet VAD를 사용하여 음성 검출

```bash
python 02_3_noise_detection_marblenet.py
```



### 노이즈 + 음성 데이터 생성

앞 단계에서 추출한 노이즈 음원과 학습 데이터의 음성을 이용하여 데이터를 2가지 레이블의 데이터를 생성함 => 노이즈 음원 + real 음성,  노이즈 음원 + fake 음성

- 참고논문: [LibriMix: An Open-Source Dataset for Generalizable Speech Separation](https://arxiv.org/abs/2005.11262)
- 코드: https://github.com/popcornell/SparseLibriMix

#### Step 1: 메타데이터 생성

```bash
python 03_1_create_metadata.py
```

#### Step 2: noise + speech 음원 생성

```bash
python 03_2_make_mixtures.py
```

