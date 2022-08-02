# Use
# docker build --no-cache -t imageai/python .
# docker run -it imageai/python /bin/bash -l
FROM ubuntu:latest
LABEL maintainer="yaroshjeka@gmail.com"
WORKDIR .
#RUN apt update
#RUN apt-get -y install python3-pip
#RUN apt install software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt update
#RUN apt install python3.7
#RUN echo "alias python=python3.7" > ~/.bash_profile
#RUN source ~/.bash_profile

 
RUN apt-get update && apt-get install -y \
  build-essential \
  zlib1g-dev \
  libncurses5-dev  \
  libgdbm-dev \
  libnss3-dev \
  libssl-dev \
  libreadline-dev \
  libffi-dev \
  wget 
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.7.6/Python-3.7.6rc1.tgz
RUN tar -xf Python-3.7.6rc1.tgz 
WORKDIR ./Python-3.7.6rc1  
RUN ["./configure", "--enable-optimizations"]   
RUN ["make", "install"]   
WORKDIR /root
RUN touch ~/.bashrc && echo "alias python=python3.7" > ~/.bashrc
RUN apt-get -y install python3-pip
RUN python3.7 -m pip --quiet install --upgrade pip setuptools wheel
RUN python3.7 -m pip install tensorflow==2.4.0
SHELL ["source", "~/.bashrc"] 

# python -c 'import tensorflow; print(tensorflow.__version__)'

# docker pull tensorflow/tensorflow:2.4.0