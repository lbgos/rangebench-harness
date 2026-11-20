FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion ca-certificates curl wget git unzip zip file jq xxd \
    gcc g++ make gdb binutils elfutils strace ltrace \
    python3 python3-pip python3-venv python3-dev \
    netcat-openbsd socat nmap john rhash \
    sqlite3 postgresql-client \
    radare2 upx-ucl binwalk \
    openssh-client iputils-ping dnsutils tcpdump iproute2 \
    procps less vim nano tmux openjdk-17-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages pwntools requests flask pycryptodome

RUN useradd -ms /bin/bash agent && mkdir -p /work && chown agent:agent /work
USER agent
WORKDIR /work
