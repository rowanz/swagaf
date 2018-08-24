FROM python:3.6.3-jessie

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN pip install "git+git://github.com/allenai/allennlp.git@7142962d330ca5a95cade114c26a361c78f2042e"

# download spacy models
RUN python -m spacy download en_core_web_sm

# set the working directory
WORKDIR /swagaf

# install python packages
ADD ./requirements.txt .
RUN pip install -r ./requirements.txt

# add the code as the final step so that when we modify the code
# we don't bust the cached layers holding the dependencies and
# system packages.
ADD . .

ENV PYTHONPATH /swagaf

ENTRYPOINT []
CMD [ "/bin/bash" ]
