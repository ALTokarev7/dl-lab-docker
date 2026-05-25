FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir opencv-python-headless matplotlib

COPY . .

RUN python -c "import torchvision.models as models; models.vgg19(weights='DEFAULT')"

CMD ["python", "RunCheckerTest.py"]