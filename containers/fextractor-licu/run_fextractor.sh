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
	echo "--inputfile=[FILENAME] - Input time-series file or JSON datalist"
	echo ""


	echo "*** OPTIONAL ARGS ***"
	echo ""

	echo "=== INPUT OPTIONS ==="
	echo "--datalist-key=[KEY] - Dictionary key containing datalist entries. Default: data"
	echo "--nmax=[N] - Maximum number of datalist entries to process. Default: all"
	echo ""

	echo "=== MODEL OPTIONS ==="
	echo "--model=[MODEL] - Time-series feature extractor model/backend."
	echo "  Available models:"
	echo "    licu - light-curve-python handcrafted features"
	echo "  Default: licu"
	echo ""

	echo "=== TIME-SERIES PREPROCESSING OPTIONS ==="
	echo "--input-sample-policy=[POLICY] - Samples passed to the backend: observed or completed. Default: observed"	
	echo "--preproc-profile=[PROFILE] - Time-series preprocessing profile. Default: default"
	echo "--timeseries-layout=[LAYOUT] - Input table layout: long or wide"
	echo ""
	echo "--time-column=[COLUMN] - Timestamp column name for tabular input"
	echo "--value-columns=[COL1:COL2:...] - Value columns for long-layout input, or inline JSON value fields"
	echo "--error-columns=[COL1:COL2:...] - Optional uncertainty/error columns"
	echo "--error-prefixes=[PREFIX1:PREFIX2:...] - Error prefixes for wide-layout input"
	echo "--time-prefix=[PREFIX] - Timestamp prefix for irregular wide-layout input"
	echo "--time-start-column=[COLUMN] - Start-time column for regular wide-layout input"
	echo "--cadence-column=[COLUMN] - Cadence column for regular wide-layout input"
	
	echo ""
	echo "--value-prefixes=[PREFIX1:PREFIX2:...] - Channel column prefixes for wide-layout input"
	echo "--channel-names=[NAME1:NAME2:...] - Logical names assigned to time-series channels"
	echo "--label-column=[COLUMN] - Optional sample-level label column for wide-layout input"
	echo "--metadata-columns=[COL1:COL2:...] - Additional sample-level metadata columns"
	echo ""
	echo "--time-start-key=[KEY] - Inline JSON field containing the start timestamp"
	echo "--cadence-key=[KEY] - Inline JSON field containing the sampling cadence"
	echo ""
	echo "--regularize - Enable regularization onto a fixed temporal grid"
	echo "--no-regularize - Explicitly disable regularization"
	echo "--regularization-method=[METHOD] - Regularization method: bin or gp"
	echo "--cadence=[VALUE] - Target cadence in timestamp units"
	echo "--missing-strategy=[STRATEGY] - Bin missing-value strategy: nan, linear, pchip, akima, or cubic"
	echo "--bin-aggregation=[METHOD] - Bin aggregation: mean or inverse-variance"
	echo "--gp-sigma=[VALUE] - Optional GP kernel amplitude"
	echo "--gp-rho=[VALUE] - Optional GP Matern-3/2 correlation length scale"
	echo "--gp-jitter=[VALUE] - Optional GP noise floor when measurement errors are unavailable"
	echo ""
	echo "--time-transform=[MODE] - Time-coordinate transform: none or origin"
	echo "--value-transform=[MODE] - Value transform: none, maxabs, minmax, standard, or asinh"
	echo "--value-transform-scale=[VALUE] - Optional scale used by transforms such as asinh"
	echo "--alignment=[MODE] - Alignment anchor: none, peak-max, peak-min, or peak-abs"
	echo "--alignment-window-before=[VALUE] - Time retained before the alignment anchor"
	echo "--alignment-window-after=[VALUE] - Time retained after the alignment anchor"
	
	echo ""

	echo "=== TIME-SERIES DIAGNOSTIC PLOT OPTIONS ==="
	echo "--timeseries-plot=[MODE] - Save time-series diagnostic plots: none, input, processed, or both"
	echo "--timeseries-plot-dir=[DIR] - Directory used for diagnostic plots. If omitted, plots are saved beside the output JSON"
	echo ""

	echo ""

	echo "=== REPRESENTATION OPTIONS ==="
	echo "--aggregation=[METHOD] - Token embedding aggregation strategy"
	echo "  Supported values are defined by fextractor (e.g. mean, std, max, mean_std, mean_max, mean_std_max, last, reg, flatten)"
	echo "--context-length=[N] - Optional model context length"
	echo "--batch-size=[N] - Embedding/inference batch size"
	echo ""

	echo "=== LICU OPTIONS ==="
	echo "--feature-set=[SET] - LiCu handcrafted feature set: basic, default, or full"
	echo "--invalid-feature-policy=[POLICY] - LiCu non-finite output policy: zero or error"
	echo "--min-samples=[N] - Minimum valid observed samples required per channel"
	echo ""

	echo "=== SAVE OPTIONS ==="
	echo "--outfile=[FILENAME] - Output JSON filename. Default: fextractor_results.json"
	echo ""

	echo "=== FEXTRACTOR RUN OPTIONS ==="
	echo "--device=[DEVICE] - Inference device requested from fextractor. Default: cuda"
	echo "--skip-errors - Skip failed datalist entries instead of aborting the complete run"
	echo ""

	echo "=== WRAPPER RUN OPTIONS ==="
	echo "--run - Execute the generated submission script. If omitted, only generate it"
	echo "--scriptdir=[SCRIPT_DIR] - Directory containing runtime scripts. Default: /usr/bin"
	echo "--modeldir=[MODEL_DIR] - Directory containing local model files. Default: /opt/models"
	echo "--jobdir=[JOB_DIR] - Working directory for the job. Default: current directory"
	echo "--outdir=[OUTPUT_DIR] - Directory where output files are copied. Default: current directory"
	echo "--waitcopy - Wait after copying output files, primarily for mounted/rclone storage"
	echo "--copywaittime=[SECONDS] - Copy wait duration when --waitcopy is enabled. Default: 30"
	echo "--no-logredir - Print fextractor logs directly instead of redirecting them to out.log"
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

# - Model options
MODEL="licu"
BACKEND="licu"

# - Time-series preprocessing
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

# - Time-series diagnostic plots
TIMESERIES_PLOT=""
TIMESERIES_PLOT_DIR=""

# - Representation options
AGGREGATION=""
CONTEXT_LENGTH=""
BATCH_SIZE=""

# - LiCu options
FEATURE_SET=""
INVALID_FEATURE_POLICY=""
MIN_SAMPLES=""

# - Run options passed to fextractor
SKIP_ERRORS_OPT=""
DEVICE="cuda"

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
		
		# - TIME-SERIES OPTIONS
		--input-sample-policy=*)
			INPUT_SAMPLE_POLICY=`echo "$item" | sed 's/^[^=]*=//'`
		;;		
		
		--preproc-profile=*)
			PREPROC_PROFILE=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--timeseries-layout=*)
			TIMESERIES_LAYOUT=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--time-column=*)
			TIME_COLUMN=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--value-columns=*)
			VALUE_COLUMNS=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--error-columns=*)
			ERROR_COLUMNS=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--value-prefixes=*)
			VALUE_PREFIXES=`echo "$item" | sed 's/^[^=]*=//'`
		;;
		
		--error-prefixes=*)
			ERROR_PREFIXES=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--time-prefix=*)
			TIME_PREFIX=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--time-start-column=*)
			TIME_START_COLUMN=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--cadence-column=*)
			CADENCE_COLUMN=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--channel-names=*)
			CHANNEL_NAMES=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--label-column=*)
			LABEL_COLUMN=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--metadata-columns=*)
			METADATA_COLUMNS=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--time-start-key=*)
			TIME_START_KEY=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--cadence-key=*)
			CADENCE_KEY=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--time-transform=*)
			TIME_TRANSFORM=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--value-transform=*)
			VALUE_TRANSFORM=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--value-transform-scale=*)
			VALUE_TRANSFORM_SCALE=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--alignment=*)
			ALIGNMENT=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--alignment-window-before=*)
			ALIGNMENT_WINDOW_BEFORE=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--alignment-window-after=*)
			ALIGNMENT_WINDOW_AFTER=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--regularize)
			REGULARIZE_OPT="--regularize"
		;;

		--no-regularize)
			REGULARIZE_OPT="--no-regularize"
		;;

		--regularization-method=*)
			REGULARIZATION_METHOD=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--bin-aggregation=*)
			BIN_AGGREGATION=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--gp-sigma=*)
			GP_SIGMA=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--gp-rho=*)
			GP_RHO=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--gp-jitter=*)
			GP_JITTER=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--cadence=*)
			CADENCE=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--missing-strategy=*)
			MISSING_STRATEGY=`echo "$item" | sed 's/^[^=]*=//'`
		;;
		
		--timeseries-plot=*)
			TIMESERIES_PLOT=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--timeseries-plot-dir=*)
			TIMESERIES_PLOT_DIR=`echo "$item" | sed 's/^[^=]*=//'`
		;;		

		--aggregation=*)
			AGGREGATION=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--context-length=*)
			CONTEXT_LENGTH=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--batch-size=*)
			BATCH_SIZE=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		# - LICU OPTIONS
		--feature-set=*)
			FEATURE_SET=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--invalid-feature-policy=*)
			INVALID_FEATURE_POLICY=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--min-samples=*)
			MIN_SAMPLES=`echo "$item" | sed 's/^[^=]*=//'`
		;;


		# - FEXTRACTOR RUN OPTIONS
		--device=*)
			DEVICE=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--skip-errors)
			SKIP_ERRORS_OPT="--skip-errors"
		;;
		 	    
    # - SAVE OPTIONS
    --outfile=*)
			OUTFILE=`echo "$item" | sed 's/^[^=]*=//'`
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

PREPROC_OPTS="--profile=$PREPROC_PROFILE "
if [ "$TIMESERIES_LAYOUT" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --timeseries-layout=$TIMESERIES_LAYOUT "
fi

if [ "$TIME_COLUMN" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --time-column=\"$TIME_COLUMN\" "
fi

if [ "$LABEL_COLUMN" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --label-column=\"$LABEL_COLUMN\" "
fi

if [ "$TIME_START_KEY" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --time-start-key=\"$TIME_START_KEY\" "
fi

if [ "$CADENCE_KEY" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --cadence-key=\"$CADENCE_KEY\" "
fi

if [ "$TIME_TRANSFORM" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --time-transform=$TIME_TRANSFORM "
fi

if [ "$VALUE_TRANSFORM" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --value-transform=$VALUE_TRANSFORM "
fi

if [ "$VALUE_TRANSFORM_SCALE" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --value-transform-scale=$VALUE_TRANSFORM_SCALE "
fi

if [ "$ALIGNMENT" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --alignment=$ALIGNMENT "
fi

if [ "$ALIGNMENT_WINDOW_BEFORE" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --alignment-window-before=$ALIGNMENT_WINDOW_BEFORE "
fi

if [ "$ALIGNMENT_WINDOW_AFTER" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --alignment-window-after=$ALIGNMENT_WINDOW_AFTER "
fi

if [ "$REGULARIZATION_METHOD" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --regularization-method=$REGULARIZATION_METHOD "
fi

if [ "$CADENCE" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --cadence=$CADENCE "
fi

if [ "$MISSING_STRATEGY" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --missing-strategy=$MISSING_STRATEGY "
fi

if [ "$BIN_AGGREGATION" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --bin-aggregation=$BIN_AGGREGATION "
fi

if [ "$GP_SIGMA" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --gp-sigma=$GP_SIGMA "
fi

if [ "$GP_RHO" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --gp-rho=$GP_RHO "
fi

if [ "$GP_JITTER" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --gp-jitter=$GP_JITTER "
fi

PREPROC_OPTS="$PREPROC_OPTS $REGULARIZE_OPT "


# - Convert CAESAR colon-separated list arguments to fextractor space-separated arguments
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

if [ "$TIME_PREFIX" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --time-prefix=\"$TIME_PREFIX\" "
fi

if [ "$TIME_START_COLUMN" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --time-start-column=\"$TIME_START_COLUMN\" "
fi

if [ "$CADENCE_COLUMN" != "" ]; then
	PREPROC_OPTS="$PREPROC_OPTS --cadence-column=\"$CADENCE_COLUMN\" "
fi

if [ "$CHANNEL_NAMES" != "" ]; then
	CHANNEL_NAMES_ARGS=`echo "$CHANNEL_NAMES" | tr ':' ' '`
	PREPROC_OPTS="$PREPROC_OPTS --channel-names $CHANNEL_NAMES_ARGS "
fi

if [ "$METADATA_COLUMNS" != "" ]; then
	METADATA_COLUMNS_ARGS=`echo "$METADATA_COLUMNS" | tr ':' ' '`
	PREPROC_OPTS="$PREPROC_OPTS --metadata-columns $METADATA_COLUMNS_ARGS "
fi


REPRESENTATION_OPTS=""

if [ "$AGGREGATION" != "" ]; then
	REPRESENTATION_OPTS="$REPRESENTATION_OPTS --aggregation=$AGGREGATION "
fi

if [ "$CONTEXT_LENGTH" != "" ]; then
	REPRESENTATION_OPTS="$REPRESENTATION_OPTS --context-length=$CONTEXT_LENGTH "
fi

if [ "$BATCH_SIZE" != "" ]; then
	REPRESENTATION_OPTS="$REPRESENTATION_OPTS --batch-size=$BATCH_SIZE "
fi

if [ "$INPUT_SAMPLE_POLICY" != "" ]; then
	REPRESENTATION_OPTS="$REPRESENTATION_OPTS --input-sample-policy=$INPUT_SAMPLE_POLICY "
fi

LICU_OPTS=""

if [ "$FEATURE_SET" != "" ]; then
	LICU_OPTS="$LICU_OPTS --feature-set=$FEATURE_SET "
fi

if [ "$INVALID_FEATURE_POLICY" != "" ]; then
	LICU_OPTS="$LICU_OPTS --invalid-feature-policy=$INVALID_FEATURE_POLICY "
fi

if [ "$MIN_SAMPLES" != "" ]; then
	LICU_OPTS="$LICU_OPTS --min-samples=$MIN_SAMPLES "
fi


FEXTRACTOR_RUN_OPTS="--device=$DEVICE $SKIP_ERRORS_OPT "

SAVE_OPTS="--outfile=$OUTFILE "

PLOT_OPTS=""

if [ "$TIMESERIES_PLOT" != "" ]; then
	PLOT_OPTS="$PLOT_OPTS --timeseries-plot=$TIMESERIES_PLOT "
fi

if [ "$TIMESERIES_PLOT_DIR" != "" ]; then
	PLOT_OPTS="$PLOT_OPTS --timeseries-plot-dir=\"$TIMESERIES_PLOT_DIR\" "
fi

# - Resolve selected model to fextractor backend and local model path
MODEL_OPTS=""

if [ "$MODEL" = "licu" ]; then

	BACKEND="licu"
	MODEL_OPTS=""

else

	echo "ERROR: Unknown/not supported MODEL argument '$MODEL'!"
	echo "Available models: licu"
	exit 1

fi

RUN_OPTS="--backend=$BACKEND $FEXTRACTOR_RUN_OPTS "

echo "INFO: MODEL=$MODEL"
echo "INFO: BACKEND=$BACKEND"
echo "INFO: MODEL_OPTS=$MODEL_OPTS"
echo "INFO: LICU_OPTS=$LICU_OPTS"

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
			#echo "#!/bin/bash -e"
			echo "#!/bin/bash"
			
      echo " "
      echo " "

      echo 'echo "*************************************************"'
      echo 'echo "****         PREPARE JOB                     ****"'
      echo 'echo "*************************************************"'

      echo " "
       
      echo "echo \"INFO: Entering job dir $JOB_DIR ...\""
      echo "cd $JOB_DIR"

			echo " "

      echo 'echo "*************************************************"'
      echo 'echo "****         RUN FEXTRACTOR                  ****"'
      echo 'echo "*************************************************"'
				
			EXE="fextractor" 
			ARGS="$INPUT_OPTS \
				$PREPROC_OPTS \
				$MODEL_OPTS \
				$REPRESENTATION_OPTS \
				$LICU_OPTS \
				$PLOT_OPTS \
				$SAVE_OPTS \
				$RUN_OPTS "

			CMD="$EXE $ARGS"

			echo "date"
			echo ""
		
			echo "echo \"INFO: Running feature extractor ...\""
			
			if [ $REDIRECT_LOGS = true ]; then			
      	echo "$CMD >> $logfile 2>&1"
			else
				echo "$CMD"
      fi
      
			echo " "

			echo 'JOB_STATUS=$?'
			echo 'echo "Feature extractor run terminated with status=$JOB_STATUS"'

			echo "date"

			echo " "

      echo 'echo "*************************************************"'
      echo 'echo "****         COPY DATA TO OUTDIR             ****"'
      echo 'echo "*************************************************"'
      echo 'echo ""'
			
			if [ "$JOB_DIR" != "$JOB_OUTDIR" ]; then
				echo "echo \"INFO: Copying job outputs in $JOB_OUTDIR ...\""
				echo "ls -ltr $JOB_DIR"
				echo " "

				echo "# - Copy output data"
				echo 'tab_count=`ls -1 *.dat 2>/dev/null | wc -l`'
				echo 'if [ $tab_count != 0 ] ; then'
				echo "  echo \"INFO: Copying output table file(s) to $JOB_OUTDIR ...\""
				echo "  cp *.dat $JOB_OUTDIR"
				echo "fi"

				echo " "
				
				echo 'tab_count=`ls -1 *.json 2>/dev/null | wc -l`'
				echo 'if [ $tab_count != 0 ] ; then'
				echo "  echo \"INFO: Copying output json file(s) to $JOB_OUTDIR ...\""
				#echo "  cp *.json $JOB_OUTDIR"
				echo "	cp \"$OUTFILE\" \"$JOB_OUTDIR\""
				echo "fi"
				
				echo " "
				
				echo 'tab_count=`ls -1 *.log 2>/dev/null | wc -l`'
				echo 'if [ $tab_count != 0 ] ; then'
				echo "  echo \"INFO: Copying output log file(s) to $JOB_OUTDIR ...\""
				#echo "  cp *.log $JOB_OUTDIR"
				echo "	cp \"$logfile\" \"$JOB_OUTDIR\""
				echo "fi"
				
				echo " "
				
				echo 'plot_count=`ls -1 *.png 2>/dev/null | wc -l`'
				echo 'if [ $plot_count != 0 ] ; then'
				echo "  echo \"INFO: Copying diagnostic plot file(s) to $JOB_OUTDIR ...\""
				echo "  cp *.png \"$JOB_OUTDIR\""
				echo "fi"
				
				echo " "
				
				echo "# - Show output directory"
				echo "echo \"INFO: Show files in $JOB_OUTDIR ...\""
				echo "ls -ltr $JOB_OUTDIR"

				echo " "

				echo "# - Wait a bit after copying data"
				echo "#   NB: Needed if using rclone inside a container, otherwise nothing is copied"
				if [ $WAIT_COPY = true ]; then
           echo "sleep $COPY_WAIT_TIME"
        fi
	
			fi

      echo " "
      echo " "
      
      echo 'echo "*** END RUN ***"'

			echo 'exit $JOB_STATUS'

 	) > $shfile

	chmod +x $shfile
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
