# Accept the service name (namenode, datanode, etc.) as a build argument
ARG SERVICE_NAME
FROM bde2020/hadoop-${SERVICE_NAME}:2.0.0-hadoop3.2.1-java8

# --- Point to Debian Archive ---
RUN echo "deb http://archive.debian.org/debian stretch main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security stretch/updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "Acquire::Check-Valid-Until false;" > /etc/apt/apt.conf.d/99no-check-valid-until

# --- Install Python & ML Libraries ---
RUN apt-get update && \
    apt-get install -y python3 python3-numpy python3-pandas python3-sklearn && \
    ln -s /usr/bin/python3 /usr/bin/python