# Adversarial Robustness for Speaker Identification

- _Authors_: Cameron Baird and Austin Coursey

- _Purpose_: This repository contains the code for our final project in CS6315 (Automated Verification) at Vanderbilt University in the spring 2024 semester.

### Data

For our experiments, we used the `train-clean-100` partition of the [LibriSpeech dataset](https://www.openslr.org/12). You will need to download this dataset to replicate our experiments. There are also quite a bit of file path resolutions, so you may need to modify some scripts (`get_absolute_file_path.m` comes to mind) to load the data in. Finally, we did our data preprocessing using Python and the `librosa` library. To verify the files in MATLAB, we saved each feature array as a `*.mat` file, and directly loaded that file in to MATLAB. You can check `preprocess_data.ipynb` and again, modify your file paths accordingly. The data might be a bit of a hassle, but if you want to run our experiments there is no way around downloading LibriSpeech.

### Installation

For training the models in Python, we used TensorFlow 2.15. To install the dependencies, we simply followed the [TensorFlow documentation](https://www.tensorflow.org/) and used `pip install` when necessary inside a `conda` environment. The models were converted to MATLAB via the command line and the `convert_tensorflow_to_matlab.m` script.

This project heavily uses the Neural Network Verification tool ([link](https://github.com/verivital/nnv)). You will need to install NNV before you can run our examples using the README found in the linked repository. After NNV is installed, you should be able to run the `src_matlab/*.m` files from this repository.

### Verifying a Single Example

To see an example of verifying one input audio, you can check `src_matlab/verify_single_audio_waveform.m`. The script will load in a pre-trained model to MATLAB and NNV, create a Star or ImageStar set based on the input audio, verify its robustness using NNV, and plot the output ranges of the input audio. Below is a plot from an example where the network was verified to be robust; you can see that the output ranges of the target class (1) do not overlap with any other classes.

![An example plot of the output ranges from NNV's robustness verification.](file_image_name)

### Contact

This repository is for a class final project. Therefore, it is far from complete. If there are any errors with the code, or you have more questions about the project, please contact `cameron.j.baird@vanderbilt.edu`.
