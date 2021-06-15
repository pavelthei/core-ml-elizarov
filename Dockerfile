FROM python:3.9

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

CMD ["python", "-m", "jupyterlab", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
