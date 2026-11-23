FROM python:3.10-slim

# RUN mkdir /pasteur

# Install bdeivh
RUN cd /usr/local/ && pip3 uninstall -y bdeivh && pip3 install --no-cache-dir bdeivh==0.1.0

# The entrypoint runs command line with command line arguments
ENTRYPOINT ["/bin/bash"]