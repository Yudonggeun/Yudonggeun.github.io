---
title: CPU에서 Nvidia-Docker 테스트 하기
tags: book-report
key: fluent-python-book-report
---

## Nvidia-Docker를 사용해야하는 이유
Docker는 대부분의 기업이 서버를 배포하는 데 사용하고 서버 개발자는 필수적으로 공부해야 할 분야이다. 서버의 자동화 운영이 더 쉽기 때문이다. 만약 서버에 Deep Learning Framework가 사용된다면(GPU가 사용된다면) 환경 자체가 더욱 복잡해진다. GPU 사용을 위해 환경 구축을 해본 사람이라면 공감할 것이다. (Nvidia Driver + CUDA + CuDNN) 그리고 이 과정을 딥러닝 협업하는 곳에서는 개발 환경을 통일하기 위해 많은 시간을 보낼 것이다. 그래서 다른 개발자와 쉽게 개발 환경을 통일할 수 있는 가상화 기술인 Docker를 찾게 될 수밖에 없다.

그러나 기본적으로 Docker를 통해 Container를 실행하면, GPU 자원을 사용할 수 없다... 개발 환경 문제를 해결하기 위해 Docker를 적용하려고 했지만 GPU를 지원하지 않는다는 황당한 상황이 발생한다. 그래서 Nvidia에서 직접 GPU 자원을 사용할 수 있는 Nvidia-Docker를 공개했다.

## Nvidia-Docker
### Nvidia-Docker Structure
![](https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png)

NVIDIA GPU가 있는 Host OS 위에 CUDA Driver를 설치하고 Docker Engine을 사용한다. 그리고 Container에 CUDA Toolkit을 설치하고 각 Container에서 serving, training(possible Parallel)을 하면 된다.

### 왜 GPU 없이 CPU에 설치하려고 하나
Free Tier인 ec2에서 Nvidia-Docker를 테스트하고 시간당 돈이 빠져나가는 DL AMI에서 적용하려고 했다. 그러나 대부분은 Tutorial이 GPU를 기준으로 설명하기 때문에 Nvidia-Docker를 전체적으로 이해하고 테스트해 보기까지 시간이 꽤 오래 걸렸다. 그래서 GPU 없이 CPU로만 Nvidia-Docker를 테스트해 볼 수 있게 정리하고자 했다.

### 설치
이 글은 Ubuntu를 기준으로 한다.

Host OS에 CUDA Toolkit을 설치할 필요는 없지만 Nvidia Driver는 설치가 되어있어야 한다.

Repository 추가
~~~
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo reboot
~~~

설치 가능한 Driver 확인
~~~
$ apt-cache search nvidia | grep nvidia-driver-418
nvidia-driver-430 - NVIDIA driver metapackage
~~~

Driver 설치
~~~
$ sudo apt-get install nvidia-driver-430
~~~


Docker 설치
~~~
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
sudo apt install docker-ce
~~~

NVIDIA Docker를 이용할 것이라면 굳이 설치하지 않아도 된다.
NVIDIA Docker 설치 (https://nvidia.github.io/nvidia-docker/)
~~~
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
~~~

distributions
~~~
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
~~~

### CPU 테스트
~~~
$ nvdia-docker run -ti bvlc/caffe:cpu bash
$ python
>> import caffe
>> exit()
~~~

### 만약 GPU가 있다면
Caffe 테스트
~~~
$ nvidia-docker run -ti bvlc/caffe:gpu bash
$ python
>> import caffe
>> exit()
~~~

GPU 사용량 확인
~~~
$ nvidia-docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
~~~

TensorFlow ResNet50 테스트
~~~
nvidia-docker run --rm -ti nvcr.io/nvidia/tensorflow:17.12 \
    ./nvidia-examples/cnn/nvcnn.py -m resnet50 -b 64 -g 4
~~~
