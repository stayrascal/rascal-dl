
ROOT_DATA_DIR=/data/ai01/zp/data/lenovo/
PROJECT_DIR=rascal-dl/example/generator
IMAGE_TEXT_COUNT=20000

data-gen:
    ssh zp@ai01 "docker run -d --rm -w /work/${PROJECT_DIR} --name stayrascal-data-generator -v ${ROOT_DATA_DIR}:/work \
                 gmliao-host-latest python3 generator.py --save_pickle ./data/train --save_image ./data/train text ${IMAGE_TEXT_COUNT}";

ESTIMATOR_ARGS=
train:
    python3 estimator.py ${ESTIMATOR_ARGS} --data_path=${ROOT_DATA_DIR}/${PROJECT_DIR}/data/train/text;