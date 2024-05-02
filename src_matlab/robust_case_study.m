epsilon = 0.25;

% MLP model
series_net = convert_tensorflow_net_to_matlab('saved_models/mlp_dense16_mu_compress_rate_2kHz');
net = matlab2nnv(series_net);

% CNN model
% series_net = load('saved_models/cnn_1d_rate_2kHz_mu_compress.mat');
% net = matlab2nnv(series_net);

% look at .csv files to check one that is robust
row = 711;
robust_file_path = 'C:\Users\camer\Documents\data\LibriSpeech\train-clean-100\27\123349\27-123349-0000.mat';

ms = verify_single_audio_waveform(net, epsilon, row, 2000, 1);

 %ms = verify_single_audio_waveform(net, eps, i, sample_rate, 0);