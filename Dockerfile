FROM python:3.11-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install -r requirements.txt
COPY . ./
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0