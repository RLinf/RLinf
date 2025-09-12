#!/bin/bash

# Check if the required GPU count parameter is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <required_gpu_count>"
    exit 1
fi

required_gpu=$(printf "%.1f" "$1")

while true; do
    # Get the output of ray status
    status_output=$(ray status 2>/dev/null)
    
    # Check if the ray status command executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: 'ray status' command failed. Is Ray cluster running?"
        exit 1
    fi
    
    # Extract GPU information from the output
    gpu_line=$(echo "$status_output" | grep -E "GPU")
    
    # Parse the used and total GPU counts
    total_gpu=$(echo "$gpu_line" | awk '{print $1}' | cut -d'/' -f2)
    
    echo "Available GPUs: $total_gpu (Required: $required_gpu)"
    
    # Check if the required GPU count has been reached
    if [ "$total_gpu" = "$required_gpu" ]; then
        echo "Success: Required GPU count ($required_gpu) is now available."
        break
    fi
    
    # Wait for a while before checking again
    sleep 10
done
