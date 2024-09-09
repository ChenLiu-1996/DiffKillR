#!/opt/homebrew/bin/bash
#conda init
conda activate yale529

# Define the base directories
ROOT_DIR=$(dirname $(dirname $(realpath $0)))
EXTERNAL_DATA_DIR="$ROOT_DIR/external_data/MoNuSeg/MoNuSegTestData"
RESULTS_DIR="$ROOT_DIR/results"

# Define the organs to evaluate
#organs=("Breast" "Prostate" "Colon")
organs=("Prostate")
# Define the percentage of the organ to evaluate
#percentages=("100.0" "50.0" "10.0")
percentages=("100.0")

# Define test images for each organ, key is organ name, value is list of image names
#declare -A test_images
test_images_prostate=("TCGA-EJ-A46H-01A-03-TSC" "TCGA-HC-7209-01A-01-TS1")
#test_images["Prostate"]=test_images_prostate
test_images_breast=("TCGA-AC-A2FO-01A-01-TS1" "TCGA-AO-A0J2-01A-01-BSA")
#test_images["Breast"]=test_images_breast
test_images_colon=("TCGA-A6-6782-01A-01-BS1")
#test_images["Colon"]=test_images_colon

# Network string
network_string="DiffeoInvariantNet-AutoEncoder_depth-4_latentLoss-SimCLR_epoch-200_seed-1_backgroundRatio-2.0"

for organ in "${organs[@]}"; do
    echo "Evaluating $organ..."
    
    for percentage in "${percentages[@]}"; do
        # Find the most recent model for the current organ
        model_path=$(ls -t $RESULTS_DIR/dataset-MoNuSeg_fewShot-$percentage%_organ-$organ/$network_string/model.ckpt | head -n 1)

        # Get the list of test images for the current organ
        if [ "$organ" == "Prostate" ]; then
            test_images=$test_images_prostate
        elif [ "$organ" == "Breast" ]; then
            test_images=$test_images_breast
        elif [ "$organ" == "Colon" ]; then
            test_images=$test_images_colon
        fi

        for image in "${test_images[@]}"; do
            image_path="$EXTERNAL_DATA_DIR/images/${image}.png"
            echo "Processing image: $image_path"
            echo "Model: $model_path"
        
            time python "$ROOT_DIR/src/cell_counting.py" \
                    --image_path "$image_path" \
                    --model_path "$model_path" \
                    --organ "$organ" \
                    --patch_size 32 \
                    --stride 8 \
                    --nms_threshold 0.1 \
                    --percentage $percentage \
                    --voting_k 1 \
                    --tp_iou 0.1
        done
    done
done

echo "Evaluation complete."
