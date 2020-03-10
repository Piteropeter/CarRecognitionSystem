# CarRecognitionSystem
## Building project
There is no need for building project in any way, just install dependencies and you are ready to go. It is advised though to use [Conda](https://docs.conda.io/en/latest/) tool instead of popular virtualenv and pip combo, but it is possible to build project both ways. All dependencies are stored in `requirements.txt`.  Project was created using [VMMRdb](http://vmmrdb.cecsresearch.org/).

## Running project
### Training app
To run training app just type

    python Train.py --path <path to image catalog>
To learn about optional parameters run app with `--help` flag.
The output (trained model) will be stored in `model.zip`
### Recognition app 
To run recognition app just type

    python Recognize.py --model <path to model archive> --image <path to image to recognize>
To learn about optional parameters run app with `--help` flag.

##### Developed by Piotr Przybył