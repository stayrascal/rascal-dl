
ROOT_DATA_DIR=/data/ai01/zp/data/lenovo/
IMAGE_TEXT_COUNT=20000


data-generator:
    ssh zp@ai01 "sudo mkdir -pv ${ROOT_DATA_DIR}/train"
    ssh zp@ai01 "docker run -d --rm -w /work/rascal-dl/example/generator --name stayrascal-data-generator -v ${ROOT_DATA_DIR}:/work gmliao-host-latest \
                python3 generator.py --save_pickle ./train text ${IMAGE_TEXT_COUNT}";