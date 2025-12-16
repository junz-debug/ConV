real_paths=(
    "ptah_to_real_images"
    "ptah_to_real_images"
)

fake_paths=(
    "ptah_to_real_images"
    "ptah_to_fake_images"
)

dataset_names=(
    "Imagenet256-ADM"
    "Imagenet256-ADMG"
)

# Check if array lengths match
if [ ${#real_paths[@]} -ne ${#fake_paths[@]} ]; then
    echo "Error: real_paths and fake_paths have different lengths!"
    exit 1
fi

# Arrays to store results
declare -a results_auroc
declare -a results_ap
declare -a results_acc
declare -a results_status

# Loop through each dataset pair
for i in "${!real_paths[@]}"; do
    real_path="${real_paths[$i]}"
    fake_path="${fake_paths[$i]}"
    dataset_name="${dataset_names[$i]}"
    
    echo "=========================================="
    echo "Processing dataset: $dataset_name"
    echo "Real path: $real_path"
    echo "Fake path: $fake_path"
    echo "=========================================="
    
    # Check if paths exist (for Windows/Git Bash)
    if [ ! -d "$real_path" ] && [ ! -f "$real_path" ]; then
        echo "Warning: Real path does not exist or is not accessible: $real_path"
    fi
    if [ ! -d "$fake_path" ] && [ ! -f "$fake_path" ]; then
        echo "Warning: Fake path does not exist or is not accessible: $fake_path"
    fi

    
    # Run Python script and capture output
    # Ensure results directory exists
    mkdir -p results
    output_file="results/temp_output_${i}.txt"
    
    # Run Python script
    python srcs/ConV.py \
        --real_path "$real_path" \
        --fake_path "$fake_path" \
        --max_sample 1000 \
        --batch_size 128 \
        --num_workers 4 \
        --crops_num 5 \
        --aggregation mean > "$output_file" 2>&1
    
    exit_code=$?
    
    # Read and display output
    if [ -f "$output_file" ]; then
        output=$(cat "$output_file")
        echo "$output"
        rm -f "$output_file"
    else
        output="Unable to read output file"
    fi
    
    # Check exit status of previous command
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "Error: Failed to process dataset $dataset_name!"
        echo "Exit code: $exit_code"
        echo "=========================================="
        if [ -n "$output" ]; then
            echo "Error output:"
            echo "$output" | tail -30  # Show last 30 lines of error information
        else
            echo "Warning: No error output captured"
            echo "Please manually run the following command to see the error:"
            echo "python srcs/ConV.py --real_path \"$real_path\" --fake_path \"$fake_path\" --max_sample 1000 --batch_size 128 --num_workers 4 --crops_num 20 --aggregation mean"
        fi
        echo "=========================================="
        echo ""
        results_status[$i]="Failed"
        results_auroc[$i]="N/A"
        results_ap[$i]="N/A"
        results_acc[$i]="N/A"
        echo "Continue processing next dataset? (y/n)"
        read -r response
        if [ "$response" != "y" ]; then
            echo "Script terminated"
            exit 1
        fi
    else
        echo "Dataset $dataset_name processing completed!"
        results_status[$i]="Success"
        
        # Extract JSON results from output
        result_json=$(echo "$output" | grep "RESULT_JSON:" | sed 's/RESULT_JSON://')
        if [ -n "$result_json" ]; then
            # Use python to parse JSON (more reliable)
            auroc=$(echo "$result_json" | python -c "import sys, json; print(json.load(sys.stdin)['auroc'])" 2>/dev/null)
            ap=$(echo "$result_json" | python -c "import sys, json; print(json.load(sys.stdin)['ap'])" 2>/dev/null)
            acc=$(echo "$result_json" | python -c "import sys, json; print(json.load(sys.stdin)['acc'])" 2>/dev/null)
            
            results_auroc[$i]="${auroc:-N/A}"
            results_ap[$i]="${ap:-N/A}"
            results_acc[$i]="${acc:-N/A}"
        else
            results_auroc[$i]="N/A"
            results_ap[$i]="N/A"
            results_acc[$i]="N/A"
        fi
    fi
    
    echo ""
done

echo "=========================================="
echo "All datasets processing completed!"
echo "=========================================="
echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
printf "%-15s %-10s %-10s %-10s %-10s\n" "Dataset" "Status" "AUROC" "AP" "ACC"
echo "--------------------------------------------"

# Print results for each dataset
for i in "${!dataset_names[@]}"; do
    printf "%-15s %-10s %-10s %-10s %-10s\n" \
        "${dataset_names[$i]}" \
        "${results_status[$i]}" \
        "${results_auroc[$i]}" \
        "${results_ap[$i]}" \
        "${results_acc[$i]}"
done

echo "--------------------------------------------"

# Calculate average values (only for successful datasets)
# Collect all successful result values
auroc_values=""
ap_values=""
acc_values=""
success_count=0

for i in "${!dataset_names[@]}"; do
    if [ "${results_status[$i]}" = "Success" ] && [ "${results_auroc[$i]}" != "N/A" ]; then
        if [ -z "$auroc_values" ]; then
            auroc_values="${results_auroc[$i]}"
            ap_values="${results_ap[$i]}"
            acc_values="${results_acc[$i]}"
        else
            auroc_values="$auroc_values,${results_auroc[$i]}"
            ap_values="$ap_values,${results_ap[$i]}"
            acc_values="$acc_values,${results_acc[$i]}"
        fi
        success_count=$((success_count + 1))
    fi
done

if [ $success_count -gt 0 ]; then
    # Use Python to calculate average values
    avg_result=$(python -c "
auroc_list = [$auroc_values]
ap_list = [$ap_values]
acc_list = [$acc_values]
avg_auroc = sum(auroc_list) / len(auroc_list)
avg_ap = sum(ap_list) / len(ap_list)
avg_acc = sum(acc_list) / len(acc_list)
print(f'{avg_auroc:.4f} {avg_ap:.4f} {avg_acc:.4f}')
" 2>/dev/null)
    
    if [ -n "$avg_result" ]; then
        avg_auroc=$(echo "$avg_result" | awk '{print $1}')
        avg_ap=$(echo "$avg_result" | awk '{print $2}')
        avg_acc=$(echo "$avg_result" | awk '{print $3}')
        
        printf "%-15s %-10s %-10s %-10s %-10s\n" \
            "Average" \
            "($success_count)" \
            "$avg_auroc" \
            "$avg_ap" \
            "$avg_acc"
    fi
fi

echo "=========================================="

