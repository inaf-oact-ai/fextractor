#!/bin/bash

#######################################
##         SHOW USAGE
#######################################
show_usage(){

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
	echo "--inputfile=[FILENAME] - Input time-series CSV file or JSON datalist"
	echo ""

	echo "*** OPTIONAL ARGS ***"
	echo ""

	echo "=== INPUT OPTIONS ==="
	echo "--datalist-key=[KEY] - JSON datalist root key (default: data)"
	echo "--nmax=[N] - Maximum number of datalist entries to process"
	echo ""

	echo "=== MODEL OPTIONS ==="
	echo "--model=[MODEL] - Feature extractor model/backend"
	echo "  Available models:"
	echo "    fats - legacy FATS statistical feature extractor"
	echo "  Default: fats"
	echo ""

	echo "=== TIME-SERIES OPTIONS ==="
	echo "--timeseries-layout=[long|wide] - CSV time-series layout (default: long)"
	echo "--time-column=[COLUMN] - Timestamp column, or inline JSON time-array key"
	echo "--value-columns=[COL1:COL2:...] - Long-layout columns or inline JSON value keys"
	echo "--error-columns=[COL1:COL2:...] - Long-layout uncertainty columns or inline JSON error keys"
	echo "--channel-names=[NAME1:NAME2:...] - Optional logical channel names"
	echo "--value-prefixes=[PREFIX1:PREFIX2:...] - Wide-layout value prefixes"
	echo "--error-prefixes=[PREFIX1:PREFIX2:...] - Wide-layout uncertainty prefixes"
	echo "--time-prefix=[PREFIX] - Wide-layout indexed timestamp prefix"
	echo "--time-start-column=[COLUMN] - Wide-layout regular-series start-time column"
	echo "--cadence-column=[COLUMN] - Wide-layout regular-series cadence column"
	echo "--time-start-key=[KEY] - Inline JSON regular-series start-time key"
	echo "--cadence-key=[KEY] - Inline JSON regular-series cadence key"
	echo ""
	
	echo "=== SAVE OPTIONS ==="
	echo "--outfile=[FILENAME] - Output JSON filename"
	echo "  Default: fextractor_results.json"
	echo ""

	echo "=== RUN OPTIONS ==="
	echo "--run - Execute the generated submission script"
	echo "--scriptdir=[SCRIPT_DIR] - Runtime script directory"
	echo "--jobdir=[JOB_DIR] - Working directory"
	echo "--outdir=[OUTPUT_DIR] - Output directory"
	echo "--waitcopy - Wait after copying output files"
	echo "--copywaittime=[SECONDS] - Copy wait duration"
	echo "--no-logredir - Do not redirect logs to out.log"
	echo ""

	echo "=========================="
}


#######################################
##         INIT OPTIONS
#######################################

# - Run options
JOB_DIR=""
JOB_OUTDIR=""
SCRIPT_DIR="/usr/bin"

RUN_SCRIPT=false
WAIT_COPY=false
COPY_WAIT_TIME=30
REDIRECT_LOGS=true

# - Input options
INPUTFILE=""
INPUTFILE_GIVEN=false
DATALIST_KEY="data"
NMAX=""

# - Model options
MODEL="fats"

# - Time-series options
TIME_COLUMN=""
VALUE_COLUMNS=""
ERROR_COLUMNS=""
CHANNEL_NAMES=""
TIMESERIES_LAYOUT="long"

VALUE_PREFIXES=""
ERROR_PREFIXES=""

TIME_PREFIX=""
TIME_START_COLUMN=""
CADENCE_COLUMN=""

TIME_START_KEY=""
CADENCE_KEY=""

# - Save options
OUTFILE="fextractor_results.json"


#######################################
##         PARSE ARGS
#######################################

for item in "$@"
do
	case $item in

		# ==========================
		# INPUT OPTIONS
		# ==========================

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

		# ==========================
		# MODEL OPTIONS
		# ==========================

		--model=*)
			MODEL=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		# ==========================
		# TIME-SERIES OPTIONS
		# ==========================

		--time-column=*)
			TIME_COLUMN=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--value-columns=*)
			VALUE_COLUMNS=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--error-columns=*)
			ERROR_COLUMNS=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--channel-names=*)
			CHANNEL_NAMES=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--timeseries-layout=*)
			TIMESERIES_LAYOUT=`echo "$item" | sed 's/^[^=]*=//'`
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

		--time-start-key=*)
			TIME_START_KEY=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--cadence-key=*)
			CADENCE_KEY=`echo "$item" | sed 's/^[^=]*=//'`
		;;
		
		# ==========================
		# SAVE OPTIONS
		# ==========================

		--outfile=*)
			OUTFILE=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		# ==========================
		# WRAPPER OPTIONS
		# ==========================

		--run)
			RUN_SCRIPT=true
		;;

		--scriptdir=*)
			SCRIPT_DIR=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--jobdir=*)
			JOB_DIR=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--outdir=*)
			JOB_OUTDIR=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--waitcopy)
			WAIT_COPY=true
		;;

		--copywaittime=*)
			COPY_WAIT_TIME=`echo "$item" | sed 's/^[^=]*=//'`
		;;

		--no-logredir)
			REDIRECT_LOGS=false
		;;

		--help|-h)
			show_usage
			exit 0
		;;

		*)
			echo "ERROR: Unknown option ($item)...exit!"
			show_usage
			exit 1
		;;

	esac
done


#######################################
##         VALIDATE OPTIONS
#######################################

if [ "$INPUTFILE_GIVEN" = false ]; then
	echo "ERROR: Missing or empty --inputfile argument!"
	exit 1
fi

if [ "$MODEL" != "fats" ]; then
	echo "ERROR: Unknown/not supported MODEL argument '$MODEL'!"
	echo "Available models: fats"
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
##         SET OPTIONS
#######################################

INPUT_OPTS="--inputfile \"$INPUTFILE\""
INPUT_OPTS="$INPUT_OPTS --datalist-key \"$DATALIST_KEY\""

if [ "$NMAX" != "" ]; then
	INPUT_OPTS="$INPUT_OPTS --nmax \"$NMAX\""
fi

TIME_SERIES_OPTS="--timeseries-layout \"$TIMESERIES_LAYOUT\""

if [ "$TIME_COLUMN" != "" ]; then
	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --time-column \"$TIME_COLUMN\""
fi

if [ "$VALUE_COLUMNS" != "" ]; then
	VALUE_COLUMNS_ARGS=`echo "$VALUE_COLUMNS" | tr ':' ' '`

	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --value-columns $VALUE_COLUMNS_ARGS"
fi

if [ "$ERROR_COLUMNS" != "" ]; then
	ERROR_COLUMNS_ARGS=`echo "$ERROR_COLUMNS" | tr ':' ' '`

	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --error-columns $ERROR_COLUMNS_ARGS"
fi

if [ "$VALUE_PREFIXES" != "" ]; then
	VALUE_PREFIXES_ARGS=`echo "$VALUE_PREFIXES" | tr ':' ' '`

	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --value-prefixes $VALUE_PREFIXES_ARGS"
fi

if [ "$ERROR_PREFIXES" != "" ]; then
	ERROR_PREFIXES_ARGS=`echo "$ERROR_PREFIXES" | tr ':' ' '`

	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --error-prefixes $ERROR_PREFIXES_ARGS"
fi

if [ "$CHANNEL_NAMES" != "" ]; then
	CHANNEL_NAMES_ARGS=`echo "$CHANNEL_NAMES" | tr ':' ' '`

	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --channel-names $CHANNEL_NAMES_ARGS"
fi

if [ "$TIME_PREFIX" != "" ]; then
	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --time-prefix \"$TIME_PREFIX\""
fi

if [ "$TIME_START_COLUMN" != "" ]; then
	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --time-start-column \"$TIME_START_COLUMN\""
fi

if [ "$CADENCE_COLUMN" != "" ]; then
	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --cadence-column \"$CADENCE_COLUMN\""
fi

if [ "$TIME_START_KEY" != "" ]; then
	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --time-start-key \"$TIME_START_KEY\""
fi

if [ "$CADENCE_KEY" != "" ]; then
	TIME_SERIES_OPTS="$TIME_SERIES_OPTS --cadence-key \"$CADENCE_KEY\""
fi

SAVE_OPTS="--outfile \"$OUTFILE\""


#######################################
##   DEFINE GENERATE SCRIPT FCN
#######################################

shfile="submit_fextractor.sh"
logfile="out.log"


generate_exec_script(){

	local shfile=$1

	echo "INFO: Creating sh file $shfile ..."

	(
		echo "#!/bin/bash"

		echo ""
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

		EXE="fats_fextractor"

		ARGS="$INPUT_OPTS \
			$TIME_SERIES_OPTS \
			$SAVE_OPTS"

		CMD="$EXE $ARGS"

		echo "date"
		echo ""

		echo "echo \"INFO: Running FATS feature extractor ...\""
		echo "echo \"INFO: Command: $CMD\""

		if [ "$REDIRECT_LOGS" = true ]; then
			echo "$CMD >> \"$logfile\" 2>&1"
		else
			echo "$CMD"
		fi

		echo ""

		echo 'JOB_STATUS=$?'
		echo 'echo "Feature extractor run terminated with status=$JOB_STATUS"'

		echo "date"

		echo ""

		echo 'if [ "$JOB_STATUS" -ne 0 ]; then'
		echo '	echo "ERROR: FATS extraction failed"'
		echo '	exit "$JOB_STATUS"'
		echo 'fi'

		echo ""

		echo 'echo "*************************************************"'
		echo 'echo "****         COPY DATA TO OUTDIR             ****"'
		echo 'echo "*************************************************"'

		echo ""

		if [ "$JOB_DIR" != "$JOB_OUTDIR" ]; then

			echo "echo \"INFO: Copying job outputs to $JOB_OUTDIR ...\""

			echo ""

			echo "if [ -f \"$OUTFILE\" ]; then"
			echo "	cp \"$OUTFILE\" \"$JOB_OUTDIR/\""
			echo "fi"

			echo ""

			if [ "$REDIRECT_LOGS" = true ]; then
				echo "if [ -f \"$logfile\" ]; then"
				echo "	cp \"$logfile\" \"$JOB_OUTDIR/\""
				echo "fi"

				echo ""
			fi

			echo "echo \"INFO: Files in output directory:\""
			echo "ls -ltr \"$JOB_OUTDIR\""

			echo ""

			if [ "$WAIT_COPY" = true ]; then
				echo "sleep $COPY_WAIT_TIME"
			fi
		fi

		echo ""
		echo 'echo "*** END RUN ***"'

		echo ""
		echo 'exit "$JOB_STATUS"'

	) > "$shfile"

	chmod +x "$shfile"
}


#######################################
##       PREPARE JOB DIRECTORY
#######################################

if [ ! -d "$JOB_DIR" ]; then
	echo "INFO: Job dir $JOB_DIR not existing, creating it ..."
	mkdir -p "$JOB_DIR"
fi

echo "INFO: Moving to job directory $JOB_DIR ..."
cd "$JOB_DIR"


#######################################
##       GENERATE EXEC SCRIPT
#######################################

generate_exec_script \
	"$shfile"


#######################################
##       RUN FEATURE EXTRACTOR
#######################################

if [ "$RUN_SCRIPT" = true ]; then

	echo "INFO: Running script $shfile ..."

	"$JOB_DIR/$shfile"

	JOB_STATUS=$?

	if [ "$JOB_STATUS" -ne 0 ]; then
		echo "ERROR: FATS job failed with status=$JOB_STATUS"
		exit "$JOB_STATUS"
	fi
fi


echo "*** END SUBMISSION ***"

exit 0
