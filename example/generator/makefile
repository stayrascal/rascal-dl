
ROOT_DATA_DIR=/data/ai01/zp/data/lenovo/
IMAGE_TEXT_COUNT=20000

data-gen:
    ssh zp@ai01 "docker run -d --rm -w /work/rascal-dl/example/generator --name stayrascal-data-generator -v ${ROOT_DATA_DIR}:/work \
                 gmliao-host-latest python3 generator.py --save_pickle ./data/train text ${IMAGE_TEXT_COUNT}";