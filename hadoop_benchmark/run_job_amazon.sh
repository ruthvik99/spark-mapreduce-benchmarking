#!/bin/bash

# --- 1. Find the Hadoop Streaming JAR automatically ---
echo "Searching for Hadoop Streaming JAR..."
HADOOP_STREAMING_JAR=$(docker exec namenode find /opt -name "hadoop-streaming*.jar" | grep -v sources | head -n 1)

if [ -z "$HADOOP_STREAMING_JAR" ]; then
    echo "ERROR: Could not find hadoop-streaming jar inside the container!"
    # Fallback
    HADOOP_STREAMING_JAR="/opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar"
fi

echo "Using Streaming JAR at: $HADOOP_STREAMING_JAR"

# --- CONFIGURATION FOR DIRECTORY INPUT ---
LOCAL_DIR_NAME="amazon_data"
LOCAL_DATA_PATH="/code/$LOCAL_DIR_NAME"
HDFS_INPUT="/input/amazon_data"
HDFS_OUTPUT="/output_nlp_amazon"

# Files to ship to the cluster
MAPPER_SCRIPT="mapper_amazon.py"
REDUCER_SCRIPT="reducer.py"

echo "--- Checking Environment ---"
chmod +x $MAPPER_SCRIPT $REDUCER_SCRIPT

echo "--- Testing Mapper Locally (First 100 lines) ---"
echo "Testing with sample data..."
docker exec namenode bash -c "hdfs dfs -cat $HDFS_INPUT/* 2>/dev/null | head -100 | python3 /code/$MAPPER_SCRIPT" 2>&1 | head -30
echo ""

echo "--- Uploading Data to HDFS ---"
docker exec namenode hdfs dfs -mkdir -p /input
docker exec namenode hdfs dfs -rm -r -f $HDFS_INPUT

echo "Uploading directory: '$LOCAL_DATA_PATH' to HDFS..."
docker exec namenode hdfs dfs -put "$LOCAL_DATA_PATH" $HDFS_INPUT

# Verify upload
echo "Verifying data upload..."
docker exec namenode hdfs dfs -ls $HDFS_INPUT
FILE_COUNT=$(docker exec namenode hdfs dfs -ls $HDFS_INPUT | grep -c "\.tsv")
echo "Found $FILE_COUNT TSV files"

echo "--- Cleaning Previous Output ---"
docker exec namenode hdfs dfs -rm -r -f $HDFS_OUTPUT

echo "--- Running MapReduce Job (Amazon NLP) ---"
# CRITICAL: Capture start time in milliseconds for precision
start_time=$(date +%s.%N)

docker exec namenode hadoop jar "$HADOOP_STREAMING_JAR" \
    -files "/code/$MAPPER_SCRIPT,/code/$REDUCER_SCRIPT" \
    -mapper "python3 $MAPPER_SCRIPT" \
    -reducer "python3 $REDUCER_SCRIPT" \
    -input "$HDFS_INPUT" \
    -output "$HDFS_OUTPUT"

JOB_EXIT_CODE=$?
# CRITICAL: Capture end time in milliseconds for precision
end_time=$(date +%s.%N)
runtime=$(echo "$end_time - $start_time" | bc)

echo ""
echo "--- Results ---"
echo "Job Exit Code: $JOB_EXIT_CODE"
echo "Total MapReduce Execution Time: ${runtime} seconds"

if [ $JOB_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! MapReduce Output:"
    
    # Capture the reducer output
    REDUCER_OUTPUT=$(docker exec namenode hdfs dfs -cat "$HDFS_OUTPUT/part-00000")
    
    # Print the original output
    echo "$REDUCER_OUTPUT"
    
    # CRITICAL: Append the TRUE execution time to the output
    echo ""
    echo "============================================================"
    echo "ACTUAL END-TO-END MAPREDUCE EXECUTION TIME"
    echo "============================================================"
    echo "Total Job Time (s):     ${runtime}"
    echo ""
    echo "NOTE: The 'Execution Time' reported above is only the"
    echo "      mapper training time. THIS value includes:"
    echo "      - HDFS I/O"
    echo "      - Job scheduling"
    echo "      - Shuffle/Sort phase"
    echo "      - Network transfer"
    echo "      - Reducer aggregation"
    echo "============================================================"
    
else
    echo "JOB FAILED!"
    echo ""
    echo "Checking for output anyway..."
    docker exec namenode hdfs dfs -ls "$HDFS_OUTPUT" 2>/dev/null
    
    echo ""
    echo "=== DEBUGGING INFO ==="
    echo "To view detailed logs, run:"
    echo "docker exec namenode yarn logs -applicationId <app_id>"
    echo ""
    echo "Recent application IDs:"
    docker exec namenode yarn application -list -appStates FINISHED,FAILED 2>/dev/null | tail -5
fi