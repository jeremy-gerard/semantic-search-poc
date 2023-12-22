FROM python:3.8
ENV PYTHONUNBUFFERED=1
EXPOSE 3000
COPY . /wsgi
WORKDIR /wsgi
RUN pip install -r requirements.txt --no-cache --upgrade
