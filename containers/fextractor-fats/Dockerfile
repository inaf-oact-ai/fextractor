FROM ubuntu:16.04

LABEL org.opencontainers.image.title="fextractor-fats" \
      org.opencontainers.image.description="Runtime for FATS fextractor" \
      org.opencontainers.image.source="https://github.com/inaf-oact-ai/fextractor" \
      org.opencontainers.image.authors="Simone Riggi"
      
################################
##   SET ENV
################################
ARG USER_NAME=caesar
ARG USER_UID=1000
ARG USER_GID=1000

ENV USER=${USER_NAME} \
    HOME=/home/${USER_NAME}

# - Define env variables
ENV JOB_DIR=/workspace/job \
    JOB_OUTDIR=/workspace/output \
    JOB_OPTIONS="" \
    CHANGE_RUNUSER=1 \
    MOUNT_RCLONE_VOLUME=0 \
    MOUNT_VOLUME_PATH=/mnt/storage \
    RCLONE_REMOTE_STORAGE=neanias-nextcloud \
    RCLONE_REMOTE_STORAGE_PATH=. \
    RCLONE_MOUNT_WAIT_TIME=10 \
    RCLONE_COPY_WAIT_TIME=30

ENV DEBIAN_FRONTEND=noninteractive

##########################################################
##     CREATE USER
##########################################################
RUN groupadd --gid ${USER_GID} ${USER_NAME} \
    && useradd \
        --uid ${USER_UID} \
        --gid ${USER_GID} \
        --create-home \
        --home-dir ${HOME} \
        --shell /bin/bash \
        ${USER_NAME} \
    && chown -R ${USER_NAME}:${USER_NAME} ${HOME}
   
#################################
##    CREATE DIRS
#################################
RUN mkdir -p \
      /workspace/job \
      /workspace/output \
      /opt/fats \
    && chown -R \
      ${USER_NAME}:${USER_NAME} \
      /workspace \
      /opt/fats  
      
##########################################################
##     INSTALL SYS LIBS
##########################################################
RUN printf '%s\n' \
    'deb http://archive.ubuntu.com/ubuntu/ xenial main restricted universe multiverse' \
    'deb http://archive.ubuntu.com/ubuntu/ xenial-updates main restricted universe multiverse' \
    'deb http://archive.ubuntu.com/ubuntu/ xenial-security main restricted universe multiverse' \
    > /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      unzip \
      git \
      build-essential \
      fuse \
      gfortran \
      libopenblas-dev \
      liblapack-dev \
      python2.7 \
      python2.7-dev \
      python-pip \
      python-tk \
 && rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python2.7 /usr/local/bin/python

# Final Python-2-compatible packaging stack.
RUN python -m pip install --no-cache-dir --upgrade \
      pip==20.3.4 \
      setuptools==44.1.1 \
      wheel==0.34.2

# Prevent modern Python-3-only transitive packages from being selected.
RUN python -m pip install --no-cache-dir \
      pyparsing==2.4.5 \
      python-dateutil==2.8.2 \
      pytz==2021.3 \
      six==1.16.0

# Scientific stack.
RUN python -m pip install --no-cache-dir \
      numpy==1.16.6 \
      scipy==1.2.3 \
      pandas==0.24.2 \
      statsmodels==0.10.2 \
      matplotlib==2.2.5

######################################
##     INSTALL RCLONE
######################################
# - Allow other non-root users to mount fuse volumes
RUN sed -i 's/#user_allow_other/user_allow_other/' /etc/fuse.conf

# - Install rclone
RUN curl -fsSL https://rclone.org/install.sh | bash

################################
##   INSTALL FATS
################################
# FATS itself, without its historical dependency pins.      
RUN python -m pip install --no-cache-dir --no-deps \
      git+https://github.com/isadoranun/FATS.git@24fb73cd2193dce6203dc768ffc061d8676512ec

######################################
##     INSTALL FATS RUNTIME
######################################
WORKDIR /opt/fats

COPY fats_fextractor.py /home/${USER}/fats_fextractor
RUN chmod +x /home/${USER}/fats_fextractor

COPY run_fextractor.sh /home/${USER}/run_fextractor.sh
RUN chmod +x /home/${USER}/run_fextractor.sh

COPY run_job.sh /home/${USER}/run_job.sh
RUN chmod +x /home/${USER}/run_job.sh

# - Add dir to PATH
ENV PATH="${PATH}:/home/${USER}"

# - Keep the existing smoke test for now.
COPY smoke_test.py /opt/fats/smoke_test.py

RUN python /opt/fats/smoke_test.py

# - Run container
CMD ["sh","-c","/home/$USER/run_job.sh --runuser=$USER --change-runuser=$CHANGE_RUNUSER --jobargs=\"$JOB_OPTIONS\" --jobdir=$JOB_DIR --joboutdir=$JOB_OUTDIR --mount-rclone-volume=$MOUNT_RCLONE_VOLUME --mount-volume-path=$MOUNT_VOLUME_PATH --rclone-remote-storage=$RCLONE_REMOTE_STORAGE --rclone-remote-storage-path=$RCLONE_REMOTE_STORAGE_PATH --rclone-mount-wait=$RCLONE_MOUNT_WAIT_TIME --rclone-copy-wait=$RCLONE_COPY_WAIT_TIME"]

