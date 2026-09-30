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
	echo "--datalist-key=[KEY] - Dictionary key containing datalist entries. Default: data"
	echo "--nmax=[N] - Maximum number of datalist entries to process. Default: all"
	echo ""

	echo "=== MODEL OPTIONS ==="
	echo "--model=[MODEL] - Feature extractor model/backend."
	echo "  Available models:"
	echo "    dinov2"
	echo "    dinov3"
	echo "    dinov2_legacy"
	echo "    siglip"
	echo "    siglip2"
	echo "  Default: siglip2"
	echo ""

	echo "=== IMAGE PREPROCESSING OPTIONS ==="
	echo "--preproc-profile=[PROFILE] - Image preprocessing profile. Default: default"
	echo "--norm-min=[VALUE] - MinMax normalization minimum. Default: 0.0"
	echo "--norm-max=[VALUE] - MinMax normalization maximum. Default: 1.0"
	echo "--imgsize=[N] - Optional model input image size override"
	echo "--nchannels=[N] - Optional input-channel override (mapped to fextractor --in-chans)"
	echo "--clipdata - Enable scientific image clipping"
	echo "--zscale - Enable zscale stretching"
	echo "--no-zscale - Disable zscale stretching"
	echo "--zscale-contrast=[VALUE] - zscale contrast. Default: 0.25"
	echo "--set-zero-to-min - Replace blank/zero/non-finite pixels with minimum valid value"
	echo ""

	echo "=== MODEL PROCESSOR OPTIONS ==="
	echo "--reset-meanstd - SigLIP: reset processor mean/std"
	echo "--reset-rescale - SigLIP: disable processor rescaling"
	echo ""
	
	echo "=== SAVE OPTIONS ==="
	echo "--outfile=[FILENAME] - Output JSON filename. Default: fextractor_results.json"
	echo ""

	echo "=== FEXTRACTOR RUN OPTIONS ==="
	echo "--device=[DEVICE] - Inference device. Default: cuda"
	echo "--skip-errors - Skip failed datalist entries instead of aborting the run"
	echo ""

	echo "=== WRAPPER RUN OPTIONS ==="
	echo "--run - Execute the generated submission script"
	echo "--modeldir=[MODEL_DIR] - Directory containing local model files. Default: /opt/models"
	echo "--jobdir=[JOB_DIR] - Working directory. Default: current directory"
	echo "--outdir=[OUTPUT_DIR] - Output directory. Default: current directory"
	echo "--waitcopy - Wait after copying outputs"
	echo "--copywaittime=[SECONDS] - Copy wait duration. Default: 30"
	echo "--no-logredir - Print fextractor logs directly instead of redirecting to out.log"
	echo ""
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
DEVICE="cuda"

# - Model options
MODEL="siglip2"
BACKEND=""

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

# - Model processor overrides
RESET_MEANSTD=""
RESET_RESCALE=""

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
		--reset-meanstd)
			RESET_MEANSTD="--reset-meanstd"
		;;
		--reset-rescale)
			RESET_RESCALE="--reset-rescale"
		;;
		 	    
    # - SAVE OPTIONS
    --outfile=*)
    	OUTFILE=`echo "$item" | sed 's/^[^=]*=//'`
    ;;
	
		# - FEXTRACTOR RUN OPTIONS
		--device=*)
			DEVICE=`echo "$item" | sed 's/^[^=]*=//'`
		;;
		--skip-errors)
			SKIP_ERRORS_OPT="--skip-errors"
		;;
	
		# - WRAPPER RUN OPTIONS
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

PREPROC_OPTS="$PREPROC_OPTS \
$RESET_MEANSTD \
$RESET_RESCALE "


SAVE_OPTS="--outfile=$OUTFILE "

# - Resolve selected model to fextractor backend and local model path
MODEL_OPTS=""

if [ "$MODEL" = "dinov2" ]; then

	BACKEND="dinov2"
	MODELFILE="$MODEL_DIR/dinov2"

	if [ ! -d "$MODELFILE" ]; then
		echo "ERROR: DINOv2 model directory not found: $MODELFILE"
		exit 1
	fi

	MODEL_OPTS="--model=$MODELFILE "

elif [ "$MODEL" = "dinov3" ]; then

	BACKEND="dinov3"
	MODELFILE="$MODEL_DIR/dinov3"

	if [ ! -d "$MODELFILE" ]; then
		echo "ERROR: DINOv3 model directory not found: $MODELFILE"
		exit 1
	fi

	MODEL_OPTS="--model=$MODELFILE "

elif [ "$MODEL" = "dinov2_legacy" ]; then

	BACKEND="dinov2_legacy"
	MODELFILE="dinov2_vits14"
	WEIGHTFILE="$MODEL_DIR/dinov2_legacy/dinov2_vits14_pretrain.pth"
	REPO_DIR="$MODEL_DIR/dinov2_legacy/repo"

	if [ ! -d "$REPO_DIR" ]; then
		echo "ERROR: Legacy DINOv2 repository not found: $REPO_DIR"
		exit 1
	fi

	if [ ! -f "$WEIGHTFILE" ]; then
		echo "ERROR: Legacy DINOv2 weights not found: $WEIGHTFILE"
		exit 1
	fi

	MODEL_OPTS="--model=$MODELFILE --model-weights=$WEIGHTFILE "

elif [ "$MODEL" = "siglip" ]; then

	BACKEND="siglip"
	MODELFILE="$MODEL_DIR/siglip"

	if [ ! -d "$MODELFILE" ]; then
		echo "ERROR: SigLIP model directory not found: $MODELFILE"
		exit 1
	fi

	MODEL_OPTS="--model=$MODELFILE "

elif [ "$MODEL" = "siglip2" ]; then

	BACKEND="siglip2"
	MODELFILE="$MODEL_DIR/siglip2"

	if [ ! -d "$MODELFILE" ]; then
		echo "ERROR: SigLIP2 model directory not found: $MODELFILE"
		exit 1
	fi

	MODEL_OPTS="--model=$MODELFILE "

else

	echo "ERROR: Unknown/not supported MODEL argument '$MODEL'!"
	echo "Available models: dinov2, dinov3, dinov2_legacy, siglip, siglip2"
	exit 1

fi

FEXTRACTOR_RUN_OPTS="--device=$DEVICE $SKIP_ERRORS_OPT "

RUN_OPTS="--backend=$BACKEND $FEXTRACTOR_RUN_OPTS "

echo "INFO: MODEL=$MODEL"
echo "INFO: BACKEND=$BACKEND"
echo "INFO: MODEL_OPTS=$MODEL_OPTS"

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
