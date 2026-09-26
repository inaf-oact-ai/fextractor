FROM tensorflow/tensorflow:2.21.0-gpu

LABEL org.opencontainers.image.title="fextractor-tf" \
      org.opencontainers.image.description="Runtime for fextractor applications using TensorFlow models" \
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

# - Define env variables
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
    RCLONE_COPY_WAIT_TIME=30

#    WANDB_DISABLED=true \
#		 HF_HOME=/opt/huggingface \
#    HF_HUB_OFFLINE=1 \
#    TRANSFORMERS_OFFLINE=1 \
#    HF_DATASETS_OFFLINE=1 \

##########################################################
##     CREATE USER
##########################################################
# - Create user & set permissions
RUN groupadd --gid ${USER_GID} ${USER_NAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USER_NAME} \
    && chown -R ${USER_NAME}:${USER_NAME} ${HOME}

#################################
###    CREATE DIRS
#################################	
# - Create src dir	
##RUN mkdir -p ${SOFTDIR} ${MODEL_DIR} ${HF_HOME} /workspace \
##    && chown -R ${USER_NAME}:${USER_NAME} ${MODEL_DIR} ${HF_HOME} /workspace
    
RUN mkdir -p ${SOFTDIR} ${MODEL_DIR} /workspace \
    && chown -R ${USER_NAME}:${USER_NAME} ${MODEL_DIR} /workspace
	
##########################################################
##     INSTALL SYS LIBS
##########################################################
# - Install OS packages
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl bzip2 unzip nano git fuse \
    && rm -rf /var/lib/apt/lists/*

######################################
##     INSTALL RCLONE
######################################
# - Allow other non-root users to mount fuse volumes
RUN sed -i 's/#user_allow_other/user_allow_other/' /etc/fuse.conf

# - Install rclone
RUN curl -fsSL https://rclone.org/install.sh | bash

######################################
##     INSTALL FEXTRACTOR
######################################
# - Install dependencies
RUN python -m pip install --upgrade pip setuptools wheel

# - Download from github repo
RUN git clone --branch ${FEXTRACTOR_REF} --depth 1 ${FEXTRACTOR_REPO} ${FEXTRACTOR_SRC_DIR}
    
WORKDIR ${FEXTRACTOR_SRC_DIR}
RUN git pull origin ${FEXTRACTOR_REF}

# - Install
WORKDIR ${FEXTRACTOR_SRC_DIR}
#RUN python -m pip install -e "${FEXTRACTOR_SRC_DIR}[tensorflow-gpu]"
RUN python -m pip install -e "${FEXTRACTOR_SRC_DIR}[tensorflow]"

##WORKDIR /workspace
##CMD ["python", "fextractor", "--help"]

######################################
##     RUN
######################################
# - Copy models
COPY models/ ${MODEL_DIR}

# - Copy run scripts
COPY run_job.sh /home/$USER/run_job.sh
RUN chmod +x /home/$USER/run_job.sh

COPY run_fextractor.sh /home/$USER/run_fextractor.sh
RUN chmod +x /home/$USER/run_fextractor.sh

# - Add dir to PATH
ENV PATH="${PATH}:/home/${USER}"

# - Run container
CMD ["sh","-c","/home/$USER/run_job.sh --runuser=$USER --change-runuser=$CHANGE_RUNUSER --jobargs=\"$JOB_OPTIONS\" --jobdir=$JOB_DIR --joboutdir=$JOB_OUTDIR --mount-rclone-volume=$MOUNT_RCLONE_VOLUME --mount-volume-path=$MOUNT_VOLUME_PATH --rclone-remote-storage=$RCLONE_REMOTE_STORAGE --rclone-remote-storage-path=$RCLONE_REMOTE_STORAGE_PATH --rclone-mount-wait=$RCLONE_MOUNT_WAIT_TIME --rclone-copy-wait=$RCLONE_COPY_WAIT_TIME"]

