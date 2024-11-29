FROM nvidia/cuda:12.2.0-base-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && apt-get clean

COPY requirements.txt /cleftlip/requirements.txt
RUN pip3 install -r /cleftlip/requirements.txt

COPY . /cleftlip

WORKDIR /cleftlip

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["experiments/proto_nets.py", "--dataset", "cleft_lip", "--n-train", "1", "--k-train", "2", "--q-train", "15", "--n-test", "1", "--k-test", "2", "--q-test", "1", "--distance", "l2"]
