#!/bin/bash
# Submit one LST-AI job per T1w/FLAIR pair in a BIDS dataset.
#
#   bash submit_all_BIDS_LST-AI.bash /path/to/BIDS
#
# Optional settings (environment variables):
#   MODE=cpu|gpu       cpu (default): no GPU -- usually starts far sooner.
#                      gpu: 1 GPU.
#   PARTITION=name     Slurm partition (default general-cpu for cpu, general-gpu
#                      for gpu). `sinfo -s` lists the ones that exist.
#   MAX_JOBS=N         max LST_AI jobs queued/running at once (default 10 cpu, 2 gpu)
#   SUBJECTS="sub-01 sub-02"   only these subjects (default: all)
#   LST_EXTRA_ARGS="..."       extra lst flags, e.g. "--segment_only" to skip the
#                      FastSurfer annotation step (much faster on CPU)
#
# e.g. a quick CPU test on two subjects:
#   SUBJECTS="sub-01 sub-02" bash submit_all_BIDS_LST-AI.bash /path/to/BIDS

# === CONFIGURATION ===
BIDS_DIR=$(realpath "${1:?usage: $0 /path/to/BIDS}")
BIDS_PROJECT_NAME=$(basename "$BIDS_DIR")
OUTPUT_DIR=/scratch2/fs1/matthew.brier/$USER/$BIDS_PROJECT_NAME
LOG_DIR=$OUTPUT_DIR/logs
mkdir -p "$LOG_DIR"
PROCESS_LABEL="LST_AI"
# The job script lives next to this one unless overridden
RUN_SCRIPT="${RUN_SCRIPT:-$(dirname "$(realpath "$0")")/run_single_LST-AI.bash}"
[ -f "$RUN_SCRIPT" ] || { echo "ERROR: job script not found: $RUN_SCRIPT" >&2; exit 1; }

MODE="${MODE:-cpu}"
case $MODE in
	cpu)
		# Other resources come from the #SBATCH lines in the job script
		PARTITION="${PARTITION:-general-cpu}"
		SBATCH_RESOURCES=(--partition="$PARTITION")
		MAX_JOBS="${MAX_JOBS:-10}"
		;;
	gpu)
		# A short time limit lets the scheduler backfill the job into gaps
		# between longer jobs, which is often the difference between minutes
		# and hours in the GPU queue. Any GPU model will do for inference.
		PARTITION="${PARTITION:-general-gpu}"
		SBATCH_RESOURCES=(--partition="$PARTITION" --gpus=1 --time=01:00:00 --mem=16GB --cpus-per-task=4)
		MAX_JOBS="${MAX_JOBS:-2}"
		;;
	*)
		echo "ERROR: MODE must be cpu or gpu, got '$MODE'" >&2
		exit 1
		;;
esac
echo "mode: $MODE   partition: $PARTITION   max concurrent jobs: $MAX_JOBS   job script: $RUN_SCRIPT"

# Check the partition once up front instead of failing every submission
if ! sinfo -h -p "$PARTITION" -o '%P' 2>/dev/null | grep -q .; then
	echo "ERROR: Slurm partition '$PARTITION' does not exist here. Available:" >&2
	sinfo -s >&2
	echo "Re-run with PARTITION=<a CPU partition from the list above>." >&2
	exit 1
fi

# Find T1w files and submit jobs
find -L "$BIDS_DIR"/sub-* -type f -name "*_T1w.nii*" | sort | while read -r t1file; do
	# Extract subject/session
	subid=$(dirname "$t1file" | grep -o 'sub-[^/]*')
	sesid=$(dirname "$t1file" | grep -o 'ses-[^/]*')

	if [[ -n $SUBJECTS && " $SUBJECTS " != *" $subid "* ]]; then
		continue
	fi

	# Look for the matching FLAIR file in the same anat directory
	anat_dir=$(dirname "$t1file")
	flairfile=$(find -L "$anat_dir" -maxdepth 1 -type f -name "*_FLAIR.nii*" | sort | head -n 1)

	if [[ -z $flairfile ]]; then
		echo "No FLAIR found for $t1file, skipping ..."
		continue
	fi
	if [[ $t1file != *.nii.gz || $flairfile != *.nii.gz ]]; then
		echo "LST-AI needs gzipped NIfTI (.nii.gz); skipping $t1file"
		continue
	fi

	# Construct subject ID / paths
	if [[ -n $sesid ]]; then
		ID_string="${subid}_${sesid}"
		PID=${ID_string}_desc-${PROCESS_LABEL}
		OUTPUT_PATH=$OUTPUT_DIR/derivatives/$PROCESS_LABEL/$subid/$sesid/anat
		TEMP_PATH=$OUTPUT_DIR/temp/$PROCESS_LABEL/$subid/$sesid
		OUTPUT_LOG_PATH=$LOG_DIR/$PROCESS_LABEL/$subid/$sesid
		OUTPUT_LOG_FILENAME=$OUTPUT_LOG_PATH/${ID_string}
		BIDS_DIR_OUTPUT_PATH=$BIDS_DIR/derivatives/$PROCESS_LABEL/$subid/$sesid/anat
	else
		ID_string="${subid}"
		PID=${ID_string}_desc-${PROCESS_LABEL}
		OUTPUT_PATH=$OUTPUT_DIR/derivatives/$PROCESS_LABEL/$subid/anat
		TEMP_PATH=$OUTPUT_DIR/temp/$PROCESS_LABEL/$subid
		OUTPUT_LOG_PATH=$LOG_DIR/$PROCESS_LABEL/$subid
		OUTPUT_LOG_FILENAME=$OUTPUT_LOG_PATH/${ID_string}
		BIDS_DIR_OUTPUT_PATH=$BIDS_DIR/derivatives/$PROCESS_LABEL/$subid/anat
	fi

	# LST-AI writes a segmentation named like this into --output
	OUTPUT_FILE="${OUTPUT_PATH}/space-flair_seg-lst.nii.gz"
	BIDS_DIR_OUTPUT_FILE="${BIDS_DIR_OUTPUT_PATH}/space-flair_seg-lst.nii.gz"

	if [[ -s $OUTPUT_FILE || -s $BIDS_DIR_OUTPUT_FILE ]]; then
		echo "$PID is already processed, skipping ..."
		continue
	fi

	# Skip subjects that already have a job queued or running (re-running this
	# script while jobs are pending would otherwise submit duplicates)
	if squeue -h -u "$USER" -n "$PROCESS_LABEL" -o '%k' | grep -qx "$PID"; then
		echo "$PID is already queued/running, skipping ..."
		continue
	fi

	# Throttle if too many jobs already running/pending
	while [[ $(squeue -h -u "$USER" -n "$PROCESS_LABEL" -t PENDING,RUNNING | wc -l) -ge $MAX_JOBS ]]; do
		echo "Waiting to submit $PID..."
		sleep 60
	done

	# Submit job
	echo "Submitting $PID"
	mkdir -p "$OUTPUT_PATH" "$TEMP_PATH" "$OUTPUT_LOG_PATH"
	real_t1file=$(realpath "$t1file")
	real_flairfile=$(realpath "$flairfile")

	sbatch "${SBATCH_RESOURCES[@]}" \
	       --comment="$PID" \
	       --output="${OUTPUT_LOG_FILENAME}_%x_%j.out" \
	       --error="${OUTPUT_LOG_FILENAME}_%x_%j.err" \
	       --export=ALL,NVIDIA_VISIBLE_DEVICES=all,T1FILE="$real_t1file",FLAIRFILE="$real_flairfile",PID="$PID",OUTPUT_PATH="$OUTPUT_PATH",TEMP_PATH="$TEMP_PATH" \
	       "$RUN_SCRIPT"
	sleep 1
done
