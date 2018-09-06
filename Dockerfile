FROM library/python:3.6

# set the working directory
WORKDIR /

# install python packages
ADD ./requirements.txt .
RUN pip3.6 install --upgrade pip  && pip3.6 install -r requirements.txt

# Create the results directory we will place our predictions
#   ./swag.csv is the default file location leaderboard will put the dataset
#   ./results/predictions.csv is the default file location leaderboard will look for results
RUN mkdir /results

# add files in addition to requirements.txt
ADD configs/ .
ADD models/ .
ADD swag_baselines/ .
ADD Dockerfile .
ADD setup.py .
ADD readme.md .
ADD csv_predict.py .

# pip install the swagexample package
RUN pip3.6 install .

# define the default command
# if you need to run a long-lived process, use 'docker run --init'
CMD ["python csv_predict.py --include-package swag_baselines.decomposable_attention  models/model.tar.gz swag.csv --output-file results/predictions.csv"]
