FROM rayproject/ray-ml:latest-gpu

WORKDIR /home/tools

# Download prometheus
RUN wget https://github.com/prometheus/prometheus/releases/download/v2.47.1/prometheus-2.47.1.linux-amd64.tar.gz

# Install prometheus
RUN tar xvfz prometheus-*.tar.gz

# Install Locust
RUN pip install locust

RUN sudo apt-get update && sudo apt-get install -y screen && sudo apt-get install -y curl