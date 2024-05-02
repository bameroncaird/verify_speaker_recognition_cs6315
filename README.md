# Adversarial Robustness for Speaker Identification

- _Authors_: Cameron Baird and Austin Coursey

- _Purpose_: This repository contains the code for our final project in CS6315 (Automated Verification) at Vanderbilt University in the spring 2024 semester.

### Data

For our experiments, we used the `train-clean-100` partition of the [LibriSpeech dataset](https://www.openslr.org/12). You will need to download this dataset to replicate our experiments. There are also quite a bit of file path resolutions, so you may need to modify some scripts (`get_absolute_file_path.m` comes to mind) to load the data in. Finally, we did our data preprocessing using Python and the `librosa` library. To verify the files in MATLAB, we saved each feature array as a `*.mat` file, and directly loaded that file in to MATLAB. You can check `preprocess_data.ipynb` and again, modify your file paths accordingly. The data might be a bit of a hassle, but if you want to run our experiments there is no way around downloading LibriSpeech.

### Installation

For training the models in Python, we used TensorFlow 2.15. To install the dependencies, we simply followed the [TensorFlow documentation](https://www.tensorflow.org/) and used `pip install` when necessary inside a `conda` environment. The models were converted to MATLAB via the command line and the `convert_tensorflow_to_matlab.m` script.

### Verifying a Single Example

TO-DO: Finish writing this readme. Can we get a plot or two up? Anything else I need to add?
