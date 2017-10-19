test-train:
	python3 crnn_main.py --trainroot ../generator/data/train/text.pickle,../generator/data/train/rcnn.pickle \
		--valroot ../generator/data/train/test.pickle \
		--experiment ./data/model \
		--displayInterval 1 --n_test_disp 2 --niter 2 --batchSize 1 --random_sample --saveInterval 2 \
		--crnn ./data/crnn.pth

TRAIN_CRNN_PATH=./data/crnn.pth
LR=0.001
MODEL_PATH=./data/model/train
train:
	trainroot=../generator/data/train/text.pickle; \
	trainroot=$$trainroot,../generator/data/train/rcnn.pickle; \
	python3 crnn_main.py --trainroot $$trainroot \
		--valroot ../generator/data/train/test.pickle \
		--experiment ${MODEL_PATH} \
		--random_sample --crnn ${TRAIN_CRNN_PATH} --workers 1 --cuda --lr ${LR} --niter 15

TRAIN_1_CRNN_PATH=./data/model/train/netCRNN_16_1000.pth
LR_1=0.0001
MODEL_PATH_1=./data/model/train-1
train-1:
	trainroot=../image-auditor/watermark/data/data/rcnn-5/rcnn.pickle; \
	trainroot=$$trainroot,../image-auditor/watermark/data/data/rcnn-6/rcnn.pickle; \
	trainroot=$$trainroot,../image-auditor/watermark/data/data/rcnn-7/rcnn.pickle; \
	trainroot=$$trainroot,../image-auditor/watermark/data/data/rcnn-8/rcnn.pickle; \
	python3 crnn_main.py --trainroot $$trainroot \
		--valroot ../image-auditor/watermark/data/data/rcnn/rcnn.pickle \
		--experiment ${MODEL_PATH_1} \
		--random_sample --crnn ${TRAIN_1_CRNN_PATH} --workers 1 --cuda --lr ${LR_1} --niter 15

TRAIN_2_CRNN_PATH=./data/model/train-1/netCRNN_14_6000.pth 
LR_2=0.00005
MODEL_PATH_2=./data/model/train-2
train-2:
	trainroot=../image-auditor/watermark/data/data/rcnn-9/rcnn.pickle; \
	trainroot=$$trainroot,../image-auditor/watermark/data/data/rcnn-10/rcnn.pickle; \
	trainroot=$$trainroot,../image-auditor/watermark/data/data/rcnn-11/rcnn.pickle; \
	trainroot=$$trainroot,../image-auditor/watermark/data/data/rcnn-12/rcnn.pickle; \
	python3 crnn_main.py --trainroot $$trainroot \
		--valroot ../image-auditor/watermark/data/data/rcnn/rcnn.pickle \
		--experiment ${MODEL_PATH_2} \
		--random_sample --crnn ${TRAIN_2_CRNN_PATH} --workers 1 --cuda --lr ${LR_2} --niter 15

train-3:
	make train TRAIN_CRNN_PATH=./data/model/train-2/netCRNN_14_6000.pth LR=0.00005 MODEL_PATH=./data/model/train-3

train-4:
	make train-1 TRAIN_1_CRNN_PATH=./data/model/train-3/netCRNN_14_6000.pth LR_1=0.00005 MODEL_PATH_1=./data/model/train-4
	
train-5:
	make train-2 TRAIN_2_CRNN_PATH=./data/model/train-4/netCRNN_14_6000.pth LR_2=0.00005 MODEL_PATH_2=./data/model/train-5

MODEL_PATH=./data/model/train-2/netCRNN_14_6000.pth
DECODED_FILE=decoded.10.csv
test-rea:
	for f in `ls t/Texts`; do rec=`python3 demo.py ${MODEL_PATH} t/Texts/$$f | grep '=>' | egrep -o '=> .*' | cut -c4-`; echo $$f,$$rec | tee -a ${DECODED_FILE}; done
