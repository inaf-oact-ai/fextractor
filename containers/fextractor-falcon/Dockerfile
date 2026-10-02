FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

LABEL org.opencontainers.image.title="fextractor-falcon" \
      org.opencontainers.image.description="Runtime for fextractor Falcon-1 time-series representation extraction" \
      org.opencontainers.image.source="https://github.com/inaf-oact-ai/fextractor" \
      org.opencontainers.image.authors="Simone Riggi"

######################################
##   DEFINE CUSTOMIZABLE ARGS/ENVS
######################################
ARG USER_NAME=caesar
ARG USER_UID=1000
ARG USER_GID=1000

ARG FEXTRACTOR_REPO=https://github.com/inaf-oact-ai/fextractor.git
ARG FEXTRACTOR_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    USER=${USER_NAME} \
    HOME=/home/${USER_NAME} \
    SOFTDIR=/opt/software \
    FEXTRACTOR_SRC_DIR=/opt/software/fextractor \
    MODEL_DIR=/opt/models \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JOB_DIR=/workspace/job \
    JOB_OUTDIR=/workspace/output \
    JOB_OPTIONS="" \
    CHANGE_RUNUSER=1 \
    MOUNT_RCLONE_VOLUME=0 \
    MOUNT_VOLUME_PATH=/mnt/storage \
    RCLONE_REMOTE_STORAGE=neanias-nextcloud \
    RCLONE_REMOTE_STORAGE_PATH=. \
    RCLONE_MOUNT_WAIT_TIME=10 \
    RCLONE_COPY_WAIT_TIME=30 \
    WANDB_DISABLED=true \
    HF_HOME=/opt/huggingface \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENV PIP_BREAK_SYSTEM_PACKAGES=1

##########################################################
##     CREATE USER
##########################################################
RUN groupmod --new-name ${USER_NAME} ubuntu \
    && usermod --login ${USER_NAME} \
        --home ${HOME} \
        --move-home \
        --shell /bin/bash \
        ubuntu \
    && chown -R ${USER_NAME}:${USER_NAME} ${HOME}

#################################
###    CREATE DIRS
#################################
RUN mkdir -p ${SOFTDIR} ${MODEL_DIR} ${HF_HOME} /workspace \
    && chown -R ${USER_NAME}:${USER_NAME} ${MODEL_DIR} ${HF_HOME} /workspace

ENV MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p ${MPLCONFIGDIR}

##########################################################
##     INSTALL SYS LIBS
##########################################################
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl bzip2 unzip nano git fuse \
    && rm -rf /var/lib/apt/lists/*

######################################
##     INSTALL RCLONE
######################################
RUN sed -i 's/#user_allow_other/user_allow_other/' /etc/fuse.conf
RUN curl -fsSL https://rclone.org/install.sh | bash

######################################
##     INSTALL FEXTRACTOR
######################################
RUN python -m pip install --upgrade pip setuptools wheel
RUN git clone --branch ${FEXTRACTOR_REF} --depth 1 ${FEXTRACTOR_REPO} ${FEXTRACTOR_SRC_DIR}

WORKDIR ${FEXTRACTOR_SRC_DIR}
RUN git pull origin ${FEXTRACTOR_REF}
RUN python -m pip install -e "${FEXTRACTOR_SRC_DIR}[falcon,gp,plot]"

######################################
##     RUN
######################################
COPY models/ ${MODEL_DIR}

COPY run_job.sh /home/$USER/run_job.sh
RUN chmod +x /home/$USER/run_job.sh

COPY run_fextractor.sh /home/$USER/run_fextractor.sh
RUN chmod +x /home/$USER/run_fextractor.sh

ENV PATH="${PATH}:/home/${USER}"

CMD ["sh","-c","/home/$USER/run_job.sh --runuser=$USER --change-runuser=$CHANGE_RUNUSER --jobargs=\"$JOB_OPTIONS\" --jobdir=$JOB_DIR --joboutdir=$JOB_OUTDIR --mount-rclone-volume=$MOUNT_RCLONE_VOLUME --mount-volume-path=$MOUNT_VOLUME_PATH --rclone-remote-storage=$RCLONE_REMOTE_STORAGE --rclone-remote-storage-path=$RCLONE_REMOTE_STORAGE_PATH --rclone-mount-wait=$RCLONE_MOUNT_WAIT_TIME --rclone-copy-wait=$RCLONE_COPY_WAIT_TIME"]
