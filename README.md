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

