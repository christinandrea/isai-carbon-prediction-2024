# Use the AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.8

# Set the home directory to /tmp to avoid permission issues
ENV HOME /tmp

# Set the working directory to the Lambda task root
WORKDIR /var/task

# Copy the function code and model directory
COPY lambda_handler.py .
COPY model /opt/model

# Set the environment variable for Python path
ENV PYTHONPATH="/var/task/python/lib/python3.8/site-packages:/var/task/bin:$PYTHONPATH"

# Install TensorFlow (CPU version) and other necessary dependencies in a single RUN command to minimize layers
RUN python3.8 -m pip install --no-cache-dir scikit-learn tensorflow-cpu opencv-python-headless boto3 numpy pandas

# Set the command to the Lambda function handler
CMD ["lambda_handler.lambda_handler"]
