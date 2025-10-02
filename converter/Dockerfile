FROM python:3.8

RUN apt-get update && apt-get install -y pcl-tools
RUN python -m pip install --upgrade pip
RUN pip install -U setuptools
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
ENV LD_LIBRARY_PATH="$PWD:$LD_LIBRARY_PATH"
RUN chmod +x ./PotreeConverter

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0"]
# ENTRYPOINT ["python", "main.py"]
