
train_real_path="path_to_train_images"
train_path_mode="subdirs"
train_output="features_train_progan_centercrop_no_norm.pkl"

if [ -n "$train_real_path" ] && [ "$train_real_path" != "" ]; then
    echo "=========================================="
    echo "Extracting training data features..."
    echo "Training path: $train_real_path"
    echo "Output file: $train_output"
    echo "=========================================="
    
    if [ "$train_path_mode" = "subdirs" ]; then
        python srcs/extract_features.py \
            --real_path "$train_real_path" \
            --path_mode subdirs \
            --data_mode ours \
            --max_sample 0 \
            --crops_num 1 \
            --output_path "$train_output" \
            --batch_size 128 \
            --num_workers 6
    else
        python extract_features.py \
            --real_path "$train_real_path" \
            --fake_path "$train_fake_path" \
            --path_mode separate \
            --data_mode ours \
            --max_sample 0 \
            --crops_num 1 \
            --output_path "$train_output" \
            --batch_size 128 \
            --num_workers 6
    fi
    
    echo "Training feature extraction completed!"
    echo ""
fi

echo "=========================================="
echo "Feature extraction completed!"
echo "Note: Test data does not need feature extraction, original images will be used directly during testing"
echo "=========================================="

