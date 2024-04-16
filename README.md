# 4th place solution (tattaka's part)

# Requirements
16GB x 4 VRAM (trained on NVIDIA RTX A4000 x 4).

## Environment
Use [Kaggle Docker](https://console.cloud.google.com/gcr/images/kaggle-gpu-images/GLOBAL/python).  
Follow [tattaka/ml_environment](https://github.com/tattaka/ml_environment) to build the environment.  
You can run `./RUN-KAGGLE-GPU-ENV.sh` and launch the docker container.

## Usage
0. Place [competition data](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data) in the `input` directory
1. Training
    ```bash
    $ cd src/exp092 && sh train.sh
    $ cd src/exp094 && sh train.sh
    $ cd src/exp107 && sh train.sh
    $ cd src/exp108 && sh train.sh
    $ cd src/exp146 && sh train.sh
    $ cd src/exp147 && sh train.sh
    ```
## License
MIT
