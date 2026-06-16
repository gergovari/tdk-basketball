#!/bin/bash

# Check if data folder is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <data_folder>"
    exit 1
fi

DATA_DIR="$1"

# Check if the provided directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory '$DATA_DIR' does not exist."
    exit 1
fi

# Get absolute path of the script directory to run isolate_run.py properly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting enrichment process for data folder: $DATA_DIR"

# Find all run videos (.mp4 and .MOV) within the expected structure:
# <data_folder>/<experiment_id>/<participant_id>/runs/<run_id>.<ext>
# We use -iname to match .mp4, .MP4, .mov, .MOV case-insensitively
find "$DATA_DIR" -type f \( -iname "*.mp4" -o -iname "*.mov" \) | while read -r video_path; do
    # Check if the path contains '/runs/' to match the required folder structure
    if [[ "$video_path" == */runs/* ]]; then
        # Extract directory structure
        runs_dir="$(dirname "$video_path")"
        participant_dir="$(dirname "$runs_dir")"
        
        # Determine the output throw directory
        output_path="$participant_dir/throw"
        
        # Create the throw directory if it doesn't exist
        mkdir -p "$output_path"
        
        echo "=================================================="
        echo "Processing video: $video_path"
        echo "Output directory: $output_path"
        
        # Run isolate_run.py
        # Using SCRIPT_DIR ensures we can call this bash script from anywhere
        python "$SCRIPT_DIR/isolate_run.py" --video "$video_path" --output_path "$output_path"
    fi
done

echo "=================================================="
echo "Enrichment complete!"
