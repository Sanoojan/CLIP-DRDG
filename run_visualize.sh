for tr_dom in 0 1 2 3 
    do
        CUDA_VISIBLE_DEVICES=8 python evaluate_segmentation.py \
          --model_name Clip_train_prompt_from_image_v2 \
          --pretrained_weights VIT_MODEL_COOPLVT/env${tr_dom}.pkl \
          --patch_size 16 \
          --test_dir DATASET_PATH/DR \
          --save_path AttentionVis/ \
          --use_shape\
          --domain $tr_dom \
          --generate_images
    done
done