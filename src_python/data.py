import tensorflow as tf
import librosa
import numpy as np
import random
import os


SAMPLE_RATE = 2000


@tf.py_function(Tout=tf.float32)
def load_waveform(file_path):
    file_path = file_path.numpy()
    signal, _ = librosa.load(
        path=file_path,
        sr=SAMPLE_RATE,
        mono=True,
        offset=0.0,
        duration=1.0,
        dtype=np.float32, # original: 16-bit audio
        res_type='soxr_hq' # see docs for librosa.resample
    )

    # normalize to [0, 255]
    # signal = 255 * ((signal + 1) / 2) # [-1, 1] -> [0, 2] -> [0, 1] -> [0, 255]
    # signal = np.rint(signal) # integers, just like the mnist example

    # smarter way (as in WaveNet)
    # output is still in [-128,127] (255 is the default value of mu)
    signal = librosa.mu_compress(signal, mu=255)
    #assert np.all(signal >= 0)
    #print(signal)

    return signal


@tf.py_function(Tout=tf.float32)
def load_mag_db_spectrogram(file_path):
    signal = load_waveform(file_path).numpy()
    stft = librosa.stft(signal, n_fft=256, hop_length=125) # [129, 129]
    magnitude, _ = librosa.magphase(stft)
    db_spec = librosa.amplitude_to_db(magnitude)
    return db_spec


@tf.py_function(Tout=tf.float32)
def load_mel_spectrogram(file_path):
    file_path = file_path.numpy()
    signal, _ = librosa.load(
        path=file_path,
        sr=16000,
        mono=True,
        offset=0.0,
        duration=1.0,
        dtype=np.float32,
        res_type='soxr_hq' # see docs for librosa.resample
    )
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=16000,
        n_fft=512,
        hop_length=128,
        n_mels=128
    )
    min_val = np.min(mel_spec)
    max_val = np.max(mel_spec)
    mel_spec = (mel_spec - min_val) / (max_val - min_val)
    mel_spec *= 255
    mel_spec = np.rint(mel_spec) - 128

    return mel_spec


def map_func_label(audio, label):
    return audio, tf.one_hot(label, 10)


def map_func_waveform_adversarial(features_dict):
    file_path = features_dict['input']
    label = features_dict['label']
    return map_func_waveform(file_path, label)


def map_func_waveform(file_path, label):
    signal = load_waveform(file_path)

    # TensorFlow will not know the shapes during runtime...
    # ...if they are not specified here.
    signal.set_shape((SAMPLE_RATE,))

    return signal, label


def map_func_waveform_adversarial(features_dict):
    file_path = features_dict['input']
    label = features_dict['label']
    return map_func_waveform(file_path, label)


def map_func_spectrogram(file_path, label):
    db_spec = load_mel_spectrogram(file_path)
    db_spec.set_shape((128, 126))
    return db_spec, label


def convert_to_dictionaries(waveform, label):
    return {'waveform': waveform, 'label': label}


def tf_dataset_prep(split_labels, split_fpaths, batch_size=64):
    # shuffle the lists with same order
    zipped = list(zip(split_labels, split_fpaths))
    random.shuffle(zipped)
    split_labels, split_fpaths = zip(*zipped)
    split_labels = list(split_labels)
    split_fpaths = list(split_fpaths)

    dataset = tf.data.Dataset.from_tensor_slices((split_fpaths, split_labels))
    dataset = dataset.shuffle(buffer_size=len(split_labels))

    # choose pre-processing: waveform or spectrogram
    dataset = dataset.map(map_func_waveform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.map(map_func_spectrogram, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(map_func_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def tf_dataset_prep_adversarial(split_labels, split_fpaths, batch_size=64):
    # shuffle the lists with same order
    zipped = list(zip(split_labels, split_fpaths))
    random.shuffle(zipped)
    split_labels, split_fpaths = zip(*zipped)
    split_labels = list(split_labels)
    split_fpaths = list(split_fpaths)

    # needs to be a dictionary mapping feature names to labels
    # https://www.tensorflow.org/neural_structured_learning/api_docs/python/nsl/keras/AdversarialRegularization
    dataset = tf.data.Dataset.from_tensor_slices({'input': split_fpaths, 'label': split_labels})
    dataset = dataset.shuffle(buffer_size=len(split_labels))
    dataset = dataset.map(map_func_waveform_adversarial, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_func_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(convert_to_dictionaries, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def get_datasets(dataset_name, batch_size=64):
    assert dataset_name in {'vox2', 'libri100'}

    datasets = []
    for mode in ['train', 'dev', 'test']:

        list_path = f'../data_lists/{dataset_name}_{mode}.txt'
        with open(list_path, 'r') as f:
            all_lines = f.readlines()

        labels = []
        fpaths = []
        for i, line in enumerate(all_lines):
            line_parts = line.strip().split(' ')
            label = int(line_parts[0])
            fpath = str(line_parts[1].strip())

            if 'vox2' == dataset_name:
                fpath = os.path.join(os.path.expanduser('~/voice_data/'), fpath)

            labels.append(label)
            fpaths.append(fpath)

        #dataset = tf_dataset_prep(labels, fpaths, batch_size)
        dataset = tf_dataset_prep_adversarial(labels, fpaths, batch_size)

        datasets.append(dataset)

    return datasets

if __name__ == '__main__':
    pass
