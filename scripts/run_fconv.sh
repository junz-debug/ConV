train_feature_path="features_train_progan_centercrop_no_norm.pkl"
# Model save path
model_path="fconv_model_feature_progan.pth"

# Training and testing parameters
margin=2000  # Margin parameter for loss function (default: 2000)

# Temperature parameter for sigmoid function during testing
# Option 1: Do not use temperature scaling (use torch.sigmoid(distance))
temperature=""
# Option 2: Use temperature scaling (use torch.sigmoid(distance/temperature))
# temperature=100000  # Current setting: use temperature scaling with temperature value 100000

# Test data paths (using original images, no feature extraction)

real_paths=(
    "path_to_real_images"
    "path_to_real_images"
)

fake_paths=(
    "path_to_fake_images"
    "path_to_fake_images"
)

dataset_names=(
    "Imagenet256-ADM"
    "Imagenet256-ADMG"
)



# Arrays to store results
declare -a results_auroc
declare -a results_ap
declare -a results_acc
declare -a results_status

# If training feature files exist, train once first
# Check if training feature files exist (supports multiple files, separated by commas)
train_files_exist=true
if [ -n "$train_feature_path" ] && [ "$train_feature_path" != "" ]; then
    # Split comma-separated paths into array
    IFS=',' read -ra FEATURE_PATHS <<< "$train_feature_path"
    for feature_file in "${FEATURE_PATHS[@]}"; do
        feature_file=$(echo "$feature_file" | xargs)  # Remove leading/trailing spaces
        if [ ! -f "$feature_file" ]; then
            echo "Warning: Training feature file does not exist: $feature_file"
            train_files_exist=false
        fi
    done
else
    train_files_exist=false
fi

if [ "$train_files_exist" = true ]; then
    echo "=========================================="
    echo "Starting model training..."
    echo "Training feature file: $train_feature_path"
    echo "=========================================="
    
    # Use first test dataset as validation set for training
    first_real_path="${real_paths[0]}"
    first_fake_path="${fake_paths[0]}"
    
    python srcs/F_ConV.py \
        --real_path "$first_real_path" \
        --fake_path "$first_fake_path" \
        --data_mode ours \
        --path_mode separate \
        --max_sample 1000 \
        --batch_size 512 \
        --num_workers 8 \
        --crops_num 1 \
        --max_epoch 1 \
        --lr 1e-5 \
        --train_feature_path "$train_feature_path" \
        --model_path "$model_path" \
        --margin "$margin" \
        ${temperature:+--temperature "$temperature"} \
        2>&1 | tee train_feature_output.txt
    
    echo "Training completed!"
    echo ""
else
    echo "Warning: Training feature file does not exist or is not specified, skipping training step"
    echo ""
fi

# Loop through all datasets for testing
echo "=========================================="
echo "Starting testing on all datasets..."
echo "=========================================="

for i in "${!real_paths[@]}"; do
    real_path="${real_paths[$i]}"
    fake_path="${fake_paths[$i]}"
    dataset_name="${dataset_names[$i]}"
    
    echo ""
    echo "=========================================="
    echo "Testing dataset: $dataset_name"
    echo "Real image path: $real_path"
    echo "Fake image path: $fake_path"
    echo "=========================================="
    
    # Run test
    output_file="results/test_feature_${dataset_name}.txt"
    
    # Ensure output directory exists
    mkdir -p results
    
    # Check Python environment
    if ! command -v python &> /dev/null; then
        echo "Error: python command not found"
        results_status[$i]="failed"
        results_auroc[$i]="N/A"
        results_ap[$i]="N/A"
        results_acc[$i]="N/A"
        continue
    fi
    
    python srcs/F_ConV.py \
        --real_path "$real_path" \
        --fake_path "$fake_path" \
        --data_mode ours \
        --path_mode separate \
        --max_sample 1000 \
        --batch_size 512 \
        --num_workers 4 \
        --crops_num 1 \
        --model_path "$model_path" \
        --load_model \
        --margin "$margin" \
        ${temperature:+--temperature "$temperature"} \
        > "$output_file" 2>&1
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Extract results from output (if script output format matches F-ConV.py)
        # Assumes output format contains keywords like "AUROC: ", "AP: ", "ACC: "
        auroc=$(grep "AUROC:" "$output_file" | tail -1 | awk '{print $2}')
        ap=$(grep "AP:" "$output_file" | tail -1 | awk '{print $2}')
        acc=$(grep "ACC:" "$output_file" | tail -1 | awk '{print $2}')
        
        if [ -n "$auroc" ] && [ -n "$ap" ] && [ -n "$acc" ]; then
            results_status[$i]="success"
            results_auroc[$i]="$auroc"
            results_ap[$i]="$ap"
            results_acc[$i]="$acc"
            echo "✓ $dataset_name test completed"
            echo "  AUROC: $auroc"
            echo "  AP: $ap"
            echo "  ACC: $acc"
        else
            results_status[$i]="partial"
            results_auroc[$i]="N/A"
            results_ap[$i]="N/A"
            results_acc[$i]="N/A"
            echo "⚠ $dataset_name test completed but unable to parse results"
            echo "Full output:"
            cat "$output_file"
        fi
    else
        results_status[$i]="failed"
        results_auroc[$i]="N/A"
        results_ap[$i]="N/A"
        results_acc[$i]="N/A"
        echo "✗ $dataset_name test failed (exit code: $exit_code)"
        echo "Error output:"
        if [ -f "$output_file" ] && [ -s "$output_file" ]; then
            cat "$output_file"
        else
            echo "  Error output file is empty or does not exist"
            echo "  Possible reasons:"
            echo "  1. Python environment not properly activated (missing torch and other modules)"
            echo "  2. Model file does not exist: $model_path"
            echo "  3. Data path does not exist or is not accessible"
        fi
    fi
done

# Summarize results
echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
printf "%-40s %-10s %-12s %-12s %-12s\n" "Dataset" "Status" "AUROC" "AP" "ACC"
echo "--------------------------------------------------------------------------------------------------------"

success_count=0
for i in "${!dataset_names[@]}"; do
    dataset_name="${dataset_names[$i]}"
    status="${results_status[$i]}"
    auroc="${results_auroc[$i]}"
    ap="${results_ap[$i]}"
    acc="${results_acc[$i]}"
    
    printf "%-40s %-10s %-12s %-12s %-12s\n" "$dataset_name" "$status" "$auroc" "$ap" "$acc"
    
    if [ "$status" = "success" ]; then
        success_count=$((success_count + 1))
    fi
done

echo "--------------------------------------------------------------------------------------------------------"
echo "Success: $success_count / ${#dataset_names[@]}"

# Calculate average results (only for successful tests)
if [ $success_count -gt 0 ]; then
    echo ""
    echo "Average Results (successful tests only):"
    
    total_auroc=0
    total_ap=0
    total_acc=0
    count=0
    
    for i in "${!dataset_names[@]}"; do
        if [ "${results_status[$i]}" = "success" ]; then
            auroc="${results_auroc[$i]}"
            ap="${results_ap[$i]}"
            acc="${results_acc[$i]}"
            
            # Use bc for floating point calculations (if available)
            if command -v bc &> /dev/null; then
                total_auroc=$(echo "$total_auroc + $auroc" | bc)
                total_ap=$(echo "$total_ap + $ap" | bc)
                total_acc=$(echo "$total_acc + $acc" | bc)
            else
                # Simple awk calculation
                total_auroc=$(awk "BEGIN {print $total_auroc + $auroc}")
                total_ap=$(awk "BEGIN {print $total_ap + $ap}")
                total_acc=$(awk "BEGIN {print $total_acc + $acc}")
            fi
            count=$((count + 1))
        fi
    done
    
    if [ $count -gt 0 ]; then
        if command -v bc &> /dev/null; then
            avg_auroc=$(echo "scale=4; $total_auroc / $count" | bc)
            avg_ap=$(echo "scale=4; $total_ap / $count" | bc)
            avg_acc=$(echo "scale=4; $total_acc / $count" | bc)
        else
            avg_auroc=$(awk "BEGIN {printf \"%.4f\", $total_auroc / $count}")
            avg_ap=$(awk "BEGIN {printf \"%.4f\", $total_ap / $count}")
            avg_acc=$(awk "BEGIN {printf \"%.4f\", $total_acc / $count}")
        fi
        
        echo "  Average AUROC: $avg_auroc"
        echo "  Average AP: $avg_ap"
        echo "  Average ACC: $avg_acc"
    fi
fi

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="

