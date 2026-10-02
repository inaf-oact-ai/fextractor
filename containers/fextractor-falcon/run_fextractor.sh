#!/bin/bash -e

# NB: -e makes script fail if the generated execution script fails.

#######################################
##         CHECK ARGS
#######################################
NARGS="$#"
echo "INFO: NARGS= $NARGS"

if [ "$NARGS" -lt 1 ]; then
	echo "ERROR: Invalid number of arguments...see script usage!"
	echo ""
	echo "Usage: $0 --inputfile=[FILENAME] [OPTIONS]"
	echo ""
	echo "Model: falcon1 - Ant International Falcon-TST Large"
	echo "Generic fextractor time-series preprocessing, aggregation and plotting options are supported."
	exit 1
fi

#######################################
##         PARSE ARGS
#######################################
JOB_DIR=""
JOB_OUTDIR=""
RUN_SCRIPT=false
WAIT_COPY=false
COPY_WAIT_TIME=30
REDIRECT_LOGS=true
MODEL_DIR="/opt/models"

INPUTFILE=""
INPUTFILE_GIVEN=false
DATALIST_KEY="data"
NMAX=""

MODEL="falcon1"
BACKEND="falcon1"

INPUT_SAMPLE_POLICY=""
PREPROC_PROFILE="default"
TIMESERIES_LAYOUT=""
TIME_COLUMN=""
VALUE_COLUMNS=""
ERROR_COLUMNS=""
VALUE_PREFIXES=""
ERROR_PREFIXES=""
TIME_PREFIX=""
TIME_START_COLUMN=""
CADENCE_COLUMN=""
CHANNEL_NAMES=""
LABEL_COLUMN=""
METADATA_COLUMNS=""
TIME_START_KEY=""
CADENCE_KEY=""
TIME_TRANSFORM=""
VALUE_TRANSFORM=""
VALUE_TRANSFORM_SCALE=""
ALIGNMENT=""
ALIGNMENT_WINDOW_BEFORE=""
ALIGNMENT_WINDOW_AFTER=""
REGULARIZE_OPT=""
REGULARIZATION_METHOD=""
CADENCE=""
MISSING_STRATEGY=""
BIN_AGGREGATION=""
GP_SIGMA=""
GP_RHO=""
GP_JITTER=""

TIMESERIES_PLOT=""
TIMESERIES_PLOT_DIR=""
AGGREGATION=""
CONTEXT_LENGTH=""

SKIP_ERRORS_OPT=""
DEVICE="cuda"
OUTFILE="fextractor_results.json"

for item in "$@"
do
	case $item in
		--inputfile=*) INPUTFILE=`echo "$item" | sed 's/^[^=]*=//'`; [ "$INPUTFILE" != "" ] && INPUTFILE_GIVEN=true ;;
		--datalist-key=*) DATALIST_KEY=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--nmax=*) NMAX=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--model=*) MODEL=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--input-sample-policy=*) INPUT_SAMPLE_POLICY=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--preproc-profile=*) PREPROC_PROFILE=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--timeseries-layout=*) TIMESERIES_LAYOUT=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--time-column=*) TIME_COLUMN=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--value-columns=*) VALUE_COLUMNS=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--error-columns=*) ERROR_COLUMNS=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--value-prefixes=*) VALUE_PREFIXES=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--error-prefixes=*) ERROR_PREFIXES=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--time-prefix=*) TIME_PREFIX=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--time-start-column=*) TIME_START_COLUMN=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--cadence-column=*) CADENCE_COLUMN=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--channel-names=*) CHANNEL_NAMES=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--label-column=*) LABEL_COLUMN=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--metadata-columns=*) METADATA_COLUMNS=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--time-start-key=*) TIME_START_KEY=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--cadence-key=*) CADENCE_KEY=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--time-transform=*) TIME_TRANSFORM=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--value-transform=*) VALUE_TRANSFORM=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--value-transform-scale=*) VALUE_TRANSFORM_SCALE=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--alignment=*) ALIGNMENT=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--alignment-window-before=*) ALIGNMENT_WINDOW_BEFORE=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--alignment-window-after=*) ALIGNMENT_WINDOW_AFTER=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--regularize) REGULARIZE_OPT="--regularize" ;;
		--no-regularize) REGULARIZE_OPT="--no-regularize" ;;
		--regularization-method=*) REGULARIZATION_METHOD=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--cadence=*) CADENCE=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--missing-strategy=*) MISSING_STRATEGY=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--bin-aggregation=*) BIN_AGGREGATION=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--gp-sigma=*) GP_SIGMA=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--gp-rho=*) GP_RHO=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--gp-jitter=*) GP_JITTER=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--timeseries-plot=*) TIMESERIES_PLOT=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--timeseries-plot-dir=*) TIMESERIES_PLOT_DIR=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--aggregation=*) AGGREGATION=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--context-length=*) CONTEXT_LENGTH=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--device=*) DEVICE=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--skip-errors) SKIP_ERRORS_OPT="--skip-errors" ;;
		--outfile=*) OUTFILE=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--run*) RUN_SCRIPT=true ;;
		--outdir=*) JOB_OUTDIR=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--modeldir=*) MODEL_DIR=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--waitcopy*) WAIT_COPY=true ;;
		--copywaittime=*) COPY_WAIT_TIME=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--jobdir=*) JOB_DIR=`echo "$item" | sed 's/^[^=]*=//'` ;;
		--no-logredir*) REDIRECT_LOGS=false ;;
		*) echo "ERROR: Unknown option ($item)...exit!"; exit 1 ;;
	esac
done

if [ "$INPUTFILE_GIVEN" = false ]; then
	echo "ERROR: Missing or empty INPUTFILE argument!"
	exit 1
fi

[ "$JOB_DIR" = "" ] && JOB_DIR="$PWD"
[ "$JOB_OUTDIR" = "" ] && JOB_OUTDIR="$PWD"

#######################################
##   SET OPTIONS
#######################################
INPUT_OPTS="--inputfile=$INPUTFILE --datalist-key=$DATALIST_KEY "
[ "$NMAX" != "" ] && INPUT_OPTS="$INPUT_OPTS --nmax=$NMAX "

PREPROC_OPTS="--profile=$PREPROC_PROFILE "
[ "$TIMESERIES_LAYOUT" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --timeseries-layout=$TIMESERIES_LAYOUT "
[ "$TIME_COLUMN" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --time-column=\"$TIME_COLUMN\" "
[ "$LABEL_COLUMN" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --label-column=\"$LABEL_COLUMN\" "
[ "$TIME_START_KEY" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --time-start-key=\"$TIME_START_KEY\" "
[ "$CADENCE_KEY" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --cadence-key=\"$CADENCE_KEY\" "
[ "$TIME_TRANSFORM" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --time-transform=$TIME_TRANSFORM "
[ "$VALUE_TRANSFORM" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --value-transform=$VALUE_TRANSFORM "
[ "$VALUE_TRANSFORM_SCALE" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --value-transform-scale=$VALUE_TRANSFORM_SCALE "
[ "$ALIGNMENT" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --alignment=$ALIGNMENT "
[ "$ALIGNMENT_WINDOW_BEFORE" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --alignment-window-before=$ALIGNMENT_WINDOW_BEFORE "
[ "$ALIGNMENT_WINDOW_AFTER" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --alignment-window-after=$ALIGNMENT_WINDOW_AFTER "
[ "$REGULARIZATION_METHOD" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --regularization-method=$REGULARIZATION_METHOD "
[ "$CADENCE" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --cadence=$CADENCE "
[ "$MISSING_STRATEGY" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --missing-strategy=$MISSING_STRATEGY "
[ "$BIN_AGGREGATION" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --bin-aggregation=$BIN_AGGREGATION "
[ "$GP_SIGMA" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --gp-sigma=$GP_SIGMA "
[ "$GP_RHO" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --gp-rho=$GP_RHO "
[ "$GP_JITTER" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --gp-jitter=$GP_JITTER "
PREPROC_OPTS="$PREPROC_OPTS $REGULARIZE_OPT "

if [ "$VALUE_COLUMNS" != "" ]; then
	VALUE_COLUMNS_ARGS=`echo "$VALUE_COLUMNS" | tr ':' ' '`
	PREPROC_OPTS="$PREPROC_OPTS --value-columns $VALUE_COLUMNS_ARGS "
fi
if [ "$ERROR_COLUMNS" != "" ]; then
	ERROR_COLUMNS_ARGS=`echo "$ERROR_COLUMNS" | tr ':' ' '`
	PREPROC_OPTS="$PREPROC_OPTS --error-columns $ERROR_COLUMNS_ARGS "
fi
if [ "$VALUE_PREFIXES" != "" ]; then
	VALUE_PREFIXES_ARGS=`echo "$VALUE_PREFIXES" | tr ':' ' '`
	PREPROC_OPTS="$PREPROC_OPTS --value-prefixes $VALUE_PREFIXES_ARGS "
fi
if [ "$ERROR_PREFIXES" != "" ]; then
	ERROR_PREFIXES_ARGS=`echo "$ERROR_PREFIXES" | tr ':' ' '`
	PREPROC_OPTS="$PREPROC_OPTS --error-prefixes $ERROR_PREFIXES_ARGS "
fi
[ "$TIME_PREFIX" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --time-prefix=\"$TIME_PREFIX\" "
[ "$TIME_START_COLUMN" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --time-start-column=\"$TIME_START_COLUMN\" "
[ "$CADENCE_COLUMN" != "" ] && PREPROC_OPTS="$PREPROC_OPTS --cadence-column=\"$CADENCE_COLUMN\" "
if [ "$CHANNEL_NAMES" != "" ]; then
	CHANNEL_NAMES_ARGS=`echo "$CHANNEL_NAMES" | tr ':' ' '`
	PREPROC_OPTS="$PREPROC_OPTS --channel-names $CHANNEL_NAMES_ARGS "
fi
if [ "$METADATA_COLUMNS" != "" ]; then
	METADATA_COLUMNS_ARGS=`echo "$METADATA_COLUMNS" | tr ':' ' '`
	PREPROC_OPTS="$PREPROC_OPTS --metadata-columns $METADATA_COLUMNS_ARGS "
fi

REPRESENTATION_OPTS=""
[ "$AGGREGATION" != "" ] && REPRESENTATION_OPTS="$REPRESENTATION_OPTS --aggregation=$AGGREGATION "
[ "$CONTEXT_LENGTH" != "" ] && REPRESENTATION_OPTS="$REPRESENTATION_OPTS --context-length=$CONTEXT_LENGTH "
[ "$INPUT_SAMPLE_POLICY" != "" ] && REPRESENTATION_OPTS="$REPRESENTATION_OPTS --input-sample-policy=$INPUT_SAMPLE_POLICY "

PLOT_OPTS=""
[ "$TIMESERIES_PLOT" != "" ] && PLOT_OPTS="$PLOT_OPTS --timeseries-plot=$TIMESERIES_PLOT "
[ "$TIMESERIES_PLOT_DIR" != "" ] && PLOT_OPTS="$PLOT_OPTS --timeseries-plot-dir=\"$TIMESERIES_PLOT_DIR\" "

if [ "$MODEL" = "falcon1" ] || [ "$MODEL" = "falcon" ]; then
	BACKEND="falcon1"
	MODELFILE="$MODEL_DIR/falcon1"
	if [ ! -d "$MODELFILE" ]; then
		echo "ERROR: Falcon-1 model directory not found: $MODELFILE"
		exit 1
	fi
else
	echo "ERROR: Unknown/not supported MODEL argument '$MODEL'!"
	echo "Available models: falcon1"
	exit 1
fi

MODEL_OPTS="--model=$MODELFILE "
RUN_OPTS="--backend=$BACKEND --device=$DEVICE $SKIP_ERRORS_OPT "
SAVE_OPTS="--outfile=$OUTFILE "

#######################################
##   DEFINE GENERATE EXEC SCRIPT FCN
#######################################
shfile="submit_fextractor.sh"
logfile="out.log"

generate_exec_script(){
	local shfile=$1
	echo "INFO: Creating sh file $shfile ..."

	(
		echo "#!/bin/bash -e"
		echo ""
		echo 'JOB_STATUS=0'
		echo 'cleanup_job(){'
		echo '\tlocal SCRIPT_STATUS=$?'
		echo '\ttrap - EXIT'
		echo '\tset +e'
		echo '\tif [ "$JOB_STATUS" -ne 0 ]; then SCRIPT_STATUS=$JOB_STATUS; fi'
		echo '\techo "*************************************************"'
		echo '\techo "****         COPY DATA TO OUTDIR             ****"'
		echo '\techo "*************************************************"'

		if [ "$JOB_DIR" != "$JOB_OUTDIR" ]; then
			echo "\tmkdir -p \"$JOB_OUTDIR\""
			echo "\tif [ -f \"$OUTFILE\" ]; then cp \"$OUTFILE\" \"$JOB_OUTDIR\"; fi"
			echo "\tif [ -f \"$logfile\" ]; then cp \"$logfile\" \"$JOB_OUTDIR\"; fi"
			echo "\tfor plot in *.png; do [ -e \"\$plot\" ] && cp \"\$plot\" \"$JOB_OUTDIR\"; done"
			if [ "$WAIT_COPY" = true ]; then echo "\tsleep $COPY_WAIT_TIME"; fi
		fi

		echo '\texit "$SCRIPT_STATUS"'
		echo '}'
		echo 'trap cleanup_job EXIT'
		echo "cd \"$JOB_DIR\""
		echo 'echo "INFO: Running feature extractor ..."'

		EXE="fextractor"
		ARGS="$INPUT_OPTS $PREPROC_OPTS $MODEL_OPTS $REPRESENTATION_OPTS $PLOT_OPTS $SAVE_OPTS $RUN_OPTS"
		CMD="$EXE $ARGS"

		if [ "$REDIRECT_LOGS" = true ]; then echo "if $CMD >> \"$logfile\" 2>&1 ; then"; else echo "if $CMD ; then"; fi
		echo '\tJOB_STATUS=0'
		echo 'else'
		echo '\tJOB_STATUS=$?'
		echo 'fi'
		echo 'echo "Feature extractor run terminated with status=$JOB_STATUS"'
		echo 'exit "$JOB_STATUS"'
	) > "$shfile"

	chmod +x "$shfile"
}

###############################
##    RUN FEATURE EXTRACTOR
###############################
if [ ! -d "$JOB_DIR" ] ; then
	echo "INFO: Job dir $JOB_DIR not existing, creating it now ..."
	mkdir -p "$JOB_DIR"
fi

cd "$JOB_DIR"
generate_exec_script "$shfile"

JOB_STATUS=0
if [ "$RUN_SCRIPT" = true ] ; then
	echo "INFO: Running script $shfile to local shell system ..."
	if "$JOB_DIR/$shfile" ; then JOB_STATUS=0; else JOB_STATUS=$?; fi
fi

echo "*** END SUBMISSION ***"
exit $JOB_STATUS
