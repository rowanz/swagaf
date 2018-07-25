#FROM library/python:3.6-slim-jessie
FROM continuumio/miniconda3

# set the working directory

WORKDIR /swagaf

# install OS packages

RUN apt-get clean \
 && apt-get update --fix-missing \
 && apt-get install -y \
      git \
      build-essential

RUN git clone https://github.com/allenai/allennlp.git && cd allennlp && git checkout 7142962d330ca5a95cade114c26a361c78f2042e && INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh && python setup.py install

# install python packages

#ADD ./requirements.txt .
#
#RUN pip3.6 install --upgrade pip \
# && pip3.6 install -r ./requirements.txt

ADD ./requirements.txt .
RUN pip install -r ./requirements.txt

# add the code as the final step so that when we modify the code
# we don't bust the cached layers holding the dependencies and
# system packages.
ADD . .


# When firing off long-running processes, use `docker run --init`
CMD [ "/bin/bash" ]