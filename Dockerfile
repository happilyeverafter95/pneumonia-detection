FROM python:3.7-slim-buster
LABEL maintainer="follow me on medium https://medium.com/@mandygu"

RUN apt-get update && apt-get install -y python3-dev build-essential

RUN mkdir -p /usr/src/classifier
WORKDIR /usr/src/classifier

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "classifier.app:app"]