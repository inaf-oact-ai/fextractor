#!/bin/bash -e

# NB: -e makes script to fail if internal script fails (for example when --run is enabled)

#######################################
##         CHECK ARGS
#######################################
NARGS="$#"
echo "INFO: NARGS= $NARGS"

if [ "$NARGS" -lt 1 ]; then
	echo "ERROR: Invalid number of arguments...see script usage!"
  echo ""
	echo "**************************"
  echo "***     USAGE          ***"
	echo "**************************"
 	echo "$0 [ARGS]"
	echo ""
	echo "=========================="
	echo "==    ARGUMENT LIST     =="
	echo "=========================="
	echo "*** MANDATORY ARGS ***"
	echo "--inputfile=[FILENAME] - Input file name (.json) containing images to be processed or individual input image (.png|.jpg|.fits)"
	echo ""

	echo "*** OPTIONAL ARGS ***"
	echo "=== INPUT OPTIONS ==="
	echo "--datalist-key=[KEY] - Dictionary key name to be read in input datalist. Default: data"
	echo "--nmax=[N] - Maximum number of datalist entries to process. Default: all"
	echo ""
	
	echo "=== MODEL OPTIONS ==="
	echo "--model=[MODEL] - Model to be used for extracting the embedding. Available: {'simclr_radio'}. Default: 'simclr_radio'"
	echo ""
	
	echo "=== IMAGE PREPROCESSING OPTIONS ==="
	echo "--preproc-profile=[PROFILE] - Scientific image preprocessing profile {default, simclr_radio}. Preprocessing options below override profile settings. Default: default"
	echo "--norm-min=[NORM_MIN] - MinMax normalization min value. Default: 0.0"
	echo "--norm-max=[NORM_MAX] - MinMax normalization max value. Default: 1.0"
	echo "--imgsize=[IMGSIZE] - Override model input image size in pixels. If omitted, the backend/model default is used."
	echo "--nchannels=[IN_CHANS] - Override number of model input channels. If omitted, the backend/model default is used."
	echo "  TensorFlow/SimCLR defaults to imgsize=224 and nchannels=1 when not overridden."
	echo "--clipdata - Clip image pixel value in range [mean-5*stddev, mean+30*stddev]. Default: not applied"
	echo "--zscale - Apply zscale stretching to image. Enabled by default with profile=simclr_radio"
	echo "--no-zscale - Disable zscale stretching to image."
	echo "--zscale-contrast=[ZSCALE_CONTRAST] - Contrast used for zscale stretching. Default: 0.25"
	echo "--set-zero-to-min - Set zero/blank/nan pixels to image min value. Default: not applied"
	#echo "--no-set-zero-to-min - Do not set zero/blank/nan pixels to image min value"
	
	echo ""
	
	echo "=== SAVE OPTIONS ==="
	echo "--outfile=[FILENAME] - Name of output file. Default: fextractor_results.json"
	echo ""
	
	echo "=== FEXTRACTOR RUN OPTIONS ==="
	##echo "--device=[DEVICE] - Inference device. Default: cuda" ## INEFFECTIVE ON TF BRANCH, SO COMMENT
	echo "--skip-errors - Skip failed datalist entries instead of aborting the run"
	echo ""
	
	echo "=== WRAPPER RUN OPTIONS ==="
	echo "--run - Run the generated run script on the local shell. If disabled only run script will be generated for later run."	
	echo "--scriptdir=[SCRIPT_DIR] - Job directory where to find scripts (default=/usr/bin)"
	echo "--modeldir=[MODEL_DIR] - Job directory where to find model & weight files (default=/opt/models)"
	echo "--jobdir=[JOB_DIR] - Job directory where to run (default=pwd)"
	echo "--outdir=[OUTPUT_DIR] - Output directory where to put run output file (default=pwd)"
	echo "--waitcopy - Wait a bit after copying output files to output dir (default=no)"
	echo "--copywaittime=[COPY_WAIT_TIME] - Time to wait after copying output files (default=30)"
	echo "--no-logredir - Do not redirect logs to output file in script "	
	echo "=========================="
  exit 1
fi


#######################################
##         PARSE ARGS
#######################################
# - Run options
JOB_DIR=""
JOB_OUTDIR=""
SCRIPT_DIR="/usr/bin"
RUN_SCRIPT=false
WAIT_COPY=false
COPY_WAIT_TIME=30
REDIRECT_LOGS=true
MODEL_DIR="/opt/models"

# - Input options
INPUTFILE=""
INPUTFILE_GIVEN=false
DATALIST_KEY="data"
NMAX=""

# - Run options passed to fextractor
SKIP_ERRORS_OPT=""
###DEVICE="cuda"  ## INEFFECTIVE ON TF BRANCH, SO COMMENT

# - Model options
MODEL="simclr_radio"
BACKEND="tensorflow"

# - Shared scientific preprocessing defaults
PREPROC_PROFILE="default"
NORM_MIN=0.0
NORM_MAX=1.0
ZSCALE_CONTRAST=0.25

CLIP_DATA=""
ZSCALE_STRETCH=""
ZERO_TO_MIN_OPT=""

# - Model-specific input overrides
IMGSIZE=""
IN_CHANS=""

# - Save options
OUTFILE="fextractor_results.json"

for item in "$@"
do
	case $item in 
		# **************************
		# **   MANDATORY
		# **************************
		# - INPUT OPTIONS 	
    --inputfile=*)
    	INPUTFILE=`echo "$item" | sed 's/^[^=]*=//'`		
			if [ "$INPUTFILE" != "" ]; then
				INPUTFILE_GIVEN=true
			fi
    ;;
    --datalist-key=*)
    	DATALIST_KEY=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
    --nmax=*)
			NMAX=`echo "$item" | sed 's/^[^=]*=//'`
		;;
    
    # **************************
		# **   OPTIONAL OPTIONS 
		# **************************
		# - MODEL options
		--model=*)
    	MODEL=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
		
		# - PREPROC OPTIONS
		--preproc-profile=*)
    	PREPROC_PROFILE=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
    --norm-min=*)
    	NORM_MIN=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
    --norm-max=*)
    	NORM_MAX=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
    --imgsize=*)
    	IMGSIZE=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
    --nchannels=*)
    	IN_CHANS=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
		--clipdata)
			CLIP_DATA="--clip-data"
		;;
		# NB: Put this before --zscale otherwise the --zscale matches also the --zscale-contrasts
    --zscale-contrast=*)
			ZSCALE_CONTRAST=`echo "$item" | sed 's/^[^=]*=//'`
		;;
		--zscale)
			ZSCALE_STRETCH="--zscale"
		;;
		--no-zscale)
			ZSCALE_STRETCH="--no-zscale"
		;;
		--set-zero-to-min)
			ZERO_TO_MIN_OPT="--set-zero-to-min"
		;;
 	    
    # - SAVE OPTIONS
    --outfile=*)
    	OUTFILE=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
	
		# - FEXTRACTOR RUN OPTIONS
##		--device=*)
##			DEVICE=`echo "$item" | sed 's/^[^=]*=//'`
##		;;

		--skip-errors)
			SKIP_ERRORS_OPT="--skip-errors"
		;;
		
		# - RUN OPTIONS
    --run*)
    	RUN_SCRIPT=true
    ;;
    --scriptdir=*)
    	SCRIPT_DIR=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
    --outdir=*)
    	JOB_OUTDIR=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
    --modeldir=*)
			MODEL_DIR=`echo "$item" | sed 's/^[^=]*=//'`
		;;
		--waitcopy*)
    	WAIT_COPY=true
    ;;
		--copywaittime=*)
    	COPY_WAIT_TIME=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
    --jobdir=*)
    	JOB_DIR=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
    --no-logredir*)
			REDIRECT_LOGS=false
		;;
    
    *)
    # Unknown option
    echo "ERROR: Unknown option ($item)...exit!"
    exit 1
    ;;
	esac
done

if [ "$INPUTFILE_GIVEN" = false ]; then
  echo "ERROR: Missing or empty INPUTFILE args (hint: you must specify at least one)!"
  exit 1
fi

if [ "$JOB_DIR" = "" ]; then
  echo "WARN: Empty JOB_DIR given, setting it to pwd ($PWD) ..."
	JOB_DIR="$PWD"
fi

if [ "$JOB_OUTDIR" = "" ]; then
  echo "WARN: Empty JOB_OUTDIR given, setting it to pwd ($PWD) ..."
	JOB_OUTDIR="$PWD"
fi

#######################################
##   SET OPTIONS
#######################################
INPUT_OPTS="--inputfile=$INPUTFILE --datalist-key=$DATALIST_KEY "

if [ "$NMAX" != "" ]; then
	INPUT_OPTS="$INPUT_OPTS --nmax=$NMAX "
fi


PREPROC_OPTS="--profile=$PREPROC_PROFILE \
--norm-min=$NORM_MIN \
--norm-max=$NORM_MAX \
--zscale-contrast=$ZSCALE_CONTRAST \
$ZSCALE_STRETCH \
$CLIP_DATA \
$ZERO_TO_MIN_OPT "

if [ "$IMGSIZE" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --imgsize=$IMGSIZE "
fi

if [ "$IN_CHANS" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --in-chans=$IN_CHANS "
fi

SAVE_OPTS="--outfile=$OUTFILE "

if [ "$MODEL" = "simclr_radio" ]; then

	BACKEND="tensorflow"

	MODELFILE="$MODEL_DIR/simclr_radio/resnet18/encoder-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5"
	WEIGHTFILE="$MODEL_DIR/simclr_radio/resnet18/encoder_weights-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5"

	if [ ! -f "$MODELFILE" ]; then
		echo "ERROR: SimCLR model file not found: $MODELFILE"
		exit 1
	fi

	if [ ! -f "$WEIGHTFILE" ]; then
		echo "ERROR: SimCLR weights file not found: $WEIGHTFILE"
		exit 1
	fi

	MODEL_OPTS="--keras-loader=tf_keras --model=$MODELFILE --model-weights=$WEIGHTFILE "

else
	echo "ERROR: Unknown/not supported MODEL argument '$MODEL'!"
	echo "Available models: simclr_radio"
	exit 1

fi

##FEXTRACTOR_RUN_OPTS="--device=$DEVICE $SKIP_ERRORS_OPT "
FEXTRACTOR_RUN_OPTS="$SKIP_ERRORS_OPT "

RUN_OPTS="--backend=$BACKEND $FEXTRACTOR_RUN_OPTS "

#######################################
##   DEFINE GENERATE EXE SCRIPT FCN
#######################################
# - Set shfile
shfile="submit_fextractor.sh"

# - Set log file
logfile="out.log"

generate_exec_script(){

	local shfile=$1

	echo "INFO: Creating sh file $shfile ..."

	(
		echo "#!/bin/bash -e"
		echo ""
		echo ""

		echo 'JOB_STATUS=0'
		echo ""

		echo 'cleanup_job(){'
		echo '	local SCRIPT_STATUS=$?'
		echo ''
		echo '	trap - EXIT'
		echo '	set +e'
		echo ''
		echo '	if [ "$JOB_STATUS" -ne 0 ]; then'
		echo '		SCRIPT_STATUS=$JOB_STATUS'
		echo '	fi'
		echo ''
		echo '	echo "*************************************************"'
		echo '	echo "****         COPY DATA TO OUTDIR             ****"'
		echo '	echo "*************************************************"'
		echo '	echo ""'

		if [ "$JOB_DIR" != "$JOB_OUTDIR" ]; then
			echo "	echo \"INFO: Copying job outputs in $JOB_OUTDIR ...\""
			echo "	ls -ltr \"$JOB_DIR\""
			echo ""

			echo "	# - Copy output data"
			echo '	tab_count=`ls -1 *.dat 2>/dev/null | wc -l`'
			echo '	if [ "$tab_count" != 0 ] ; then'
			echo "		echo \"INFO: Copying output table file(s) to $JOB_OUTDIR ...\""
			echo "		cp *.dat \"$JOB_OUTDIR\""
			echo '	fi'
			echo ""

			echo '	json_count=`ls -1 *.json 2>/dev/null | wc -l`'
			echo '	if [ "$json_count" != 0 ] ; then'
			echo "		echo \"INFO: Copying output json file(s) to $JOB_OUTDIR ...\""
			echo "		if [ -f \"$OUTFILE\" ]; then"
			echo "			cp \"$OUTFILE\" \"$JOB_OUTDIR\""
			echo '		fi'
			echo '	fi'
			echo ""

			echo '	log_count=`ls -1 *.log 2>/dev/null | wc -l`'
			echo '	if [ "$log_count" != 0 ] ; then'
			echo "		echo \"INFO: Copying output log file(s) to $JOB_OUTDIR ...\""
			echo "		if [ -f \"$logfile\" ]; then"
			echo "			cp \"$logfile\" \"$JOB_OUTDIR\""
			echo '		fi'
			echo '	fi'
			echo ""

			echo "	echo \"INFO: Show files in $JOB_OUTDIR ...\""
			echo "	ls -ltr \"$JOB_OUTDIR\""
			echo ""

			if [ "$WAIT_COPY" = true ]; then
				echo "	sleep $COPY_WAIT_TIME"
			fi
		fi

		echo ''
		echo '	echo "*** END RUN ***"'
		echo ''
		echo '	exit "$SCRIPT_STATUS"'
		echo '}'
		echo ""

		echo 'trap cleanup_job EXIT'
		echo ""

		echo 'echo "*************************************************"'
		echo 'echo "****         PREPARE JOB                     ****"'
		echo 'echo "*************************************************"'
		echo ""

		echo "echo \"INFO: Entering job dir $JOB_DIR ...\""
		echo "cd \"$JOB_DIR\""
		echo ""

		echo 'echo "*************************************************"'
		echo 'echo "****         RUN FEXTRACTOR                  ****"'
		echo 'echo "*************************************************"'
		echo ""

		EXE="fextractor"

		ARGS="$INPUT_OPTS \
			$PREPROC_OPTS \
			$MODEL_OPTS \
			$SAVE_OPTS \
			$RUN_OPTS "

		CMD="$EXE $ARGS"

		echo "date"
		echo ""

		echo 'echo "INFO: Running feature extractor ..."'

		if [ "$REDIRECT_LOGS" = true ]; then
			echo "if $CMD >> \"$logfile\" 2>&1 ; then"
		else
			echo "if $CMD ; then"
		fi

		echo '	JOB_STATUS=0'
		echo 'else'
		echo '	JOB_STATUS=$?'
		echo 'fi'
		echo ""

		echo 'echo "Feature extractor run terminated with status=$JOB_STATUS"'
		echo "date"
		echo ""

		echo 'exit "$JOB_STATUS"'

	) > "$shfile"

	chmod +x "$shfile"
}
## close function generate_exec_script()

###############################
##    RUN FEATURE EXTRACTOR
###############################
# - Check if job directory exists
if [ ! -d "$JOB_DIR" ] ; then 
  echo "INFO: Job dir $JOB_DIR not existing, creating it now ..."
	mkdir -p "$JOB_DIR" 
fi

# - Moving to job directory
echo "INFO: Moving to job directory $JOB_DIR ..."
cd $JOB_DIR

# - Generate run script
echo "INFO: Creating run script file $shfile ..."
generate_exec_script "$shfile"

# - Launch run script
JOB_STATUS=0
if [ "$RUN_SCRIPT" = true ] ; then
	echo "INFO: Running script $shfile to local shell system ..."
	#$JOB_DIR/$shfile
	if $JOB_DIR/$shfile ; then
		JOB_STATUS=0
	else
		JOB_STATUS=$?
	fi
fi


echo "*** END SUBMISSION ***"
exit $JOB_STATUS
