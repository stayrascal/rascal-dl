DATASET_DIR=/data/ai03/zp/data/ai_challenger_scene_train_20170904/
VALIDATION_DIR=/data/ai03/zp/data/ai_challenger_scene_validation_20170908/
TEST_DIR=/data/ai03/zp/data/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922
DATASET_NAME=challenger_ai_scene
TRAIN_DIR=/data/ai03/zp/projects/challenger-ai-sence/inception_resnet_v2
CHECKPOINT_PATH=/data/ai03/zp/data/models/inception_resnet_v2_2016_08_30.ckpt

convert-data:
	python3 ../models/slim/download_and_convert_data.py \
		--dataset_name ${DATASET_NAME} \
		--dataset_dir ${DATASET_DIR}

train-last-layer:
	python3 ../models/slim/train_image_classifier.py \
		--train_dir=${TRAIN_DIR} \
		--dataset_name=${DATASET_NAME} \
		--dataset_split_name=train \
		--dataset_dir=${DATASET_DIR} \
		--model_name=inception_resnet_v2 \
		--checkpoint_path=${CHECKPOINT_PATH} \
		--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
		--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
		--max_number_of_steps=8000 \
		--batch_size=32 \
		--learning_rate=0.01 \
		--learning_rate_decay_type=fixed \
		--save_interval_secs=60 \
		--save_summaries_secs=60 \
		--log_every_n_steps=10 \
		--optimizer=rmsprop \
		--weight_decay=0.00004 \
		--train_image_size=512

eval:
	python3 ../models/slim/eval_image_classifier.py \
		--checkpoint_path=${TRAIN_DIR}\
		--eval_dir=${TRAIN_DIR} \
		--dataset_name=${DATASET_NAME} \
		--dataset_split_name=validation \
		--dataset_dir=${DATASET_DIR} \
		--model_name=inception_resnet_v2

finetune:
	python3 ../models/slim/train_image_classifier.py \
		--train_dir=${TRAIN_DIR}/finetune \
		--dataset_name=${DATASET_NAME} \
		--dataset_split_name=train \
		--dataset_dir=${DATASET_DIR} \
		--model_name=inception_resnet_v2 \
		--checkpoint_path=${TRAIN_DIR} \
		--max_number_of_steps=50000 \
		--batch_size=14 \
		--learning_rate=0.0001 \
		--learning_rate_decay_type=fixed \
		--save_interval_secs=60 \
		--save_summaries_secs=60 \
		--log_every_n_steps=10 \
		--optimizer=rmsprop \
		--weight_decay=0.00004 \
		--train_image_size=512


train-last-layer-inception:
	python3 ../models/slim/train_image_classifier.py \
		--train_dir=/data/flowers-models/inception_v3 \
		--dataset_name=${DATASET_NAME} \
		--dataset_split_name=train \
		--dataset_dir=${DATASET_DIR} \
		--model_name=inception_v3 \
		--checkpoint_path=/data/models/inception_v3.ckpt \
		--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
		--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
		--max_number_of_steps=4000 \
		--batch_size=64 \
		--learning_rate=0.01 \
		--learning_rate_decay_type=fixed \
		--save_interval_secs=60 \
		--save_summaries_secs=60 \
		--log_every_n_steps=100 \
		--optimizer=rmsprop \
		--weight_decay=0.00004

eval-inception:
	python3 ../models/slim/eval_image_classifier.py \
		--checkpoint_path=/data/flowers-models/inception_v3/eval/checkpoint \
		--eval_dir=/data/flowers-models/inception_v3/eval \
		--dataset_name=${DATASET_NAME} \
		--dataset_split_name=validation \
		--dataset_dir=${DATASET_DIR} \
		--model_name=inception_v3

finetune_inception:
	python3 ../models/slim/train_image_classifier.py \
		--train_dir=/data/flowers-models/inception_v3/finetune \
		--dataset_name=${DATASET_NAME} \
		--dataset_split_name=train \
		--dataset_dir=${DATASET_DIR} \
		--model_name=inception_v3 \
		--checkpoint_path=/data/flowers-models/inception_v3/finetune/checkpoint \
		--max_number_of_steps=2000 \
		--batch_size=64 \
		--learning_rate=0.0001 \
		--learning_rate_decay_type=fixed \
		--save_interval_secs=60 \
		--save_summaries_secs=60 \
		--log_every_n_steps=10 \
		--optimizer=rmsprop \
		--weight_decay=0.00004

copy-model:
	-mkdir -pv data/inception_v3
	scp aigml-1:/root/dev/projects/challenger.ai/competition/caption/src/../output/finetune_on_inception_v3/inception_v3/all-2/model.ckpt-20000.* ./data/inception_v3/
	scp aigml-1:/root/dev/projects/challenger.ai/competition/caption/src/../output/finetune_on_inception_v3/inception_v3/all-2/checkpoint ./data/inception_v3/

infer:
	python3 ../models/slim/infer_image_classifier.py \
		--checkpoint_path=/data/flowers-models/inception_v3 \
		--dataset_name=${DATASET_NAME} \
		--image_to_infer=/data/samples \
		--batch_size=2 \
		--output_file=/data/flowers-models/inception_v3/infer/infer_output.json

infer-final:
	python3 ../models/slim/infer_image_classifier.py \
		--checkpoint_path=${TRAIN_DIR} \
		--model_name=inception_resnet_v2 \
		--infer_image_size=512 \
		--dataset_name=${DATASET_NAME} \
		--labels_file= \
		--image_to_infer= \
		--batch_size=20 \
		--output_file=test.json

		//2017-10-01 07:21:25.751888: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[0.97666669]
        //2017-10-01 07:21:25.751923: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.80666667]
        0.83351851
        0.8337037