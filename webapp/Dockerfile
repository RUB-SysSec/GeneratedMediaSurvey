FROM python:3.10

WORKDIR /app
COPY dfsurvey ./dfsurvey
COPY setup.py ./setup.py
COPY wsgi.py ./wsgi.py

RUN pip install .
RUN pip install gunicorn 


