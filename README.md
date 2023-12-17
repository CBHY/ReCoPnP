# ReCoPnP

이 Repository는 23-2 데이터마이닝 1분반 최종 제출물입니다. -인공지능학부 214248 김한영

##  실행 환경

python=3.8

intel 11th i9

rtx3090 * 2

RAM 64GB

*최소 18GB 이상의 GPU RAM이 필요합니다*



## 환경 불러오기 

```bash
git clone https://github.com/CBHY/ReCoPnP.git

cd ReCoPnP 
```

real_environment.yaml 맨 밑에 prefix를 본인이 생성하고자 하는 가상환경의 경로로 조정해야합니다.



```bash
conda env create --file real_environment.yaml
conda activate ReCoPnP
```



## 모델 다운로드 

### ReCo  ###

https://drive.google.com/file/d/1EqnK2boDySN4Vdh0KJwHEvpJGlCvFfsE/view?usp=drive_link

이 파일을 다운로드 받아서 ./backend/ReCo/logs 에 옮깁니다.



https://drive.google.com/file/d/1ELr0vESfAtGrCXV3jcEwxYyW4vZZSad0/view?usp=sharing

https://drive.google.com/file/d/1roFlVk7V5VjfgHkz1cmaGhRmZVFiXcl7/view?usp=sharing

이 파일들을 다운로드 받아서 ./backend/ReCo/dataset에 옮깁니다.





## 실행

./frontend/app.py에 root 변수를 설정합니다.(line 12)

```python

print(os.getcwd())

root = '/home/cvlserver/ssd2tb/ReCoPnP/' ##### 여기의 경로 설정

# Upload an image and set some options for demo purposes
st.header("Text-to-Image Generation with Art 2023")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
# box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
									.
									.
									.
									.

```





./backend/backend.py에 root 변수를 설정합니다.(line 3)

```python
import os
import sys
root = f'/home/cvlserver/ssd2tb/hkt/' ##### 여기의 경로 설정
sys.path.append(f'{root}backend/ReCo/')
sys.path.append(f'{root}pnp-diffusers')
import uuid
import torch
from fastapi import FastAPI, File, UploadFile

import argparse, os, sys, glob, re
import json
import torch
									.
									.
									.
									.
```



터미널을 엽니다(2개)

아래 명령어를 각각 실행합니다.

```bash
cd ReCoPnP
streamlit run frontend/app.py
```



```bash
cd ReCoPnP/backend
uvicorn backend:main --reload
```



##  생성 ##

1. frontend 사이트에서 upload img를 엽니다.

2. 1024.png를 업로드하고 bbox조절과 3 개의 prompt를 조정합니다.
3. submit을 누르고 2분30초(rtx 3090*2 기준)을 기다리면 이미지가 생성됩니다.