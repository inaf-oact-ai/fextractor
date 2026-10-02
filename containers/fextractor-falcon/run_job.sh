#!/bin/bash

##########################
##    PARSE ARGS
##########################
RUNUSER="caesar"
CHANGE_USER=true

JOB_DIR=""
JOB_OUTDIR=""
JOB_ARGS=""

MOUNT_RCLONE_VOLUME=0
MOUNT_VOLUME_PATH="/mnt/storage"
RCLONE_REMOTE_STORAGE="neanias-nextcloud"
RCLONE_REMOTE_STORAGE_PATH="."
RCLONE_MOUNT_WAIT_TIME=10
RCLONE_COPY_WAIT_TIME=30

echo "ARGS: $@"

for item in "$@"
do
	case $item in
		--runuser=*)
			RUNUSER=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		--change-runuser=*)
			CHANGE_USER_FLAG=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
			if [ "$CHANGE_USER_FLAG" = "1" ] ; then CHANGE_USER=true; else CHANGE_USER=false; fi
		;;
		--jobdir=*)
			JOB_DIR=`echo "$item" | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		--joboutdir=*)
			JOB_OUTDIR=`echo "$item" | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		--jobargs=*)
			JOB_ARGS=`echo "$item" | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		--mount-rclone-volume=*)
			MOUNT_RCLONE_VOLUME=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		--mount-volume-path=*)
			MOUNT_VOLUME_PATH=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		--rclone-remote-storage=*)
			RCLONE_REMOTE_STORAGE=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		--rclone-remote-storage-path=*)
			RCLONE_REMOTE_STORAGE_PATH=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		--rclone-mount-wait=*)
			RCLONE_MOUNT_WAIT_TIME=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		--rclone-copy-wait=*)
			RCLONE_COPY_WAIT_TIME=`echo $item | /bin/sed 's/[-a-zA-Z0-9]*=//'`
		;;
		*)
			echo "ERROR: Unknown option ($item)...exit!"
			exit 1
		;;
	esac
done

if [ "$JOB_ARGS" = "" ]; then
	echo "ERROR: Empty JOB_ARGS argument!"
	exit 1
fi

###############################
##    MOUNT VOLUMES
###############################
if [ "$MOUNT_RCLONE_VOLUME" = "1" ] ; then
	echo "INFO: Creating mount directory $MOUNT_VOLUME_PATH ..."
	mkdir -p $MOUNT_VOLUME_PATH

	DEVICE_ID=`stat "$HOME" -c %d`
	echo "INFO: Standard device id @ $HOME: $DEVICE_ID"

	uid=`id -u $RUNUSER`
	echo "INFO: Mounting rclone volume at path $MOUNT_VOLUME_PATH for uid/gid=$uid ..."
	MOUNT_CMD="/usr/bin/rclone mount --daemon --uid=$uid --gid=$uid --umask 000 --allow-other --file-perms 0777 --dir-cache-time 0m5s --vfs-cache-mode full $RCLONE_REMOTE_STORAGE:$RCLONE_REMOTE_STORAGE_PATH $MOUNT_VOLUME_PATH -vvv"
	eval $MOUNT_CMD

	echo "INFO: Sleeping $RCLONE_MOUNT_WAIT_TIME seconds and then check if mount is ready..."
	sleep $RCLONE_MOUNT_WAIT_TIME

	MOUNT_DEVICE_ID=`stat "$MOUNT_VOLUME_PATH" -c %d`
	echo "INFO: MOUNT_DEVICE_ID=$MOUNT_DEVICE_ID"
	if [ "$MOUNT_DEVICE_ID" = "$DEVICE_ID" ] ; then
		echo "ERROR: Failed to mount rclone storage at $MOUNT_VOLUME_PATH within $RCLONE_MOUNT_WAIT_TIME seconds, exit!"
		exit 1
	fi

	echo "INFO: Mounted rclone storage at $MOUNT_VOLUME_PATH with success (MOUNT_DEVICE_ID: $MOUNT_DEVICE_ID)..."
	ls -ltr $MOUNT_VOLUME_PATH
	mkdir -p $MOUNT_VOLUME_PATH/jobs $MOUNT_VOLUME_PATH/data
fi

###############################
##    SET OPTIONS
###############################
if [ "$JOB_DIR" == "" ]; then
	if [ "$CHANGE_USER" = true ]; then JOB_DIR="/home/$RUNUSER/caesar-job"; else JOB_DIR="$HOME/caesar-job"; fi
fi

RUN_OPTIONS="--run --jobdir=$JOB_DIR "
if [ "$JOB_OUTDIR" != "" ]; then
	RUN_OPTIONS="$RUN_OPTIONS --outdir=$JOB_OUTDIR "
	if [ "$MOUNT_RCLONE_VOLUME" = "1" ] ; then RUN_OPTIONS="$RUN_OPTIONS --waitcopy --copywaittime=$RCLONE_COPY_WAIT_TIME "; fi
fi

JOB_OPTIONS="$RUN_OPTIONS $JOB_ARGS "
EXE="/home/$RUNUSER/run_fextractor.sh"

if [ "$CHANGE_USER" = true ]; then
	CMD="runuser -l $RUNUSER -g $RUNUSER -c \"$EXE $JOB_OPTIONS\""
else
	CMD="$EXE $JOB_OPTIONS"
fi

echo "INFO: Running job command: $CMD ..."
eval "$CMD"
