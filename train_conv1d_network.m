% Train a Conv1D network in matlab
% Having issues importing it from TensorFlow

n_train_points = 1000;
data = zeros(2000, 1, 1, n_train_points);

%data = zeros(n_train_points, 2000);
labels = zeros(n_train_points, 1);
for i = 1:n_train_points
    [waveform, true_label, file_path] = read_data_line(i, 'libri100', 'train', 2000);
    data(:, 1, 1, i) = waveform';
    labels(i) = true_label;
end
labels = categorical(labels);

disp(size(data));
disp(size(labels));

layers = [
    imageInputLayer([2000 1], 'Name', 'inputs');

    convolution2dLayer([3, 1], 64, 'Padding', 'same')
    reluLayer

    % https://www.mathworks.com/matlabcentral/answers/1935274-i-m-having-trouble-with-convolution1dlayer
    % convolution1dLayer(3, 64, "Stride", 1)
    % reluLayer

    % dilated conv block
    convolution2dLayer([5, 1], 32)
    convolution2dLayer([5, 1], 32, 'DilationFactor', [2 1])
    convolution2dLayer([5, 1], 32, 'DilationFactor', [4 1])
    convolution2dLayer([5, 1], 32, 'DilationFactor', [8 1])

    convolution2dLayer([3, 1], 1, 'Stride', [2, 1])
    reluLayer

    % convolution1dLayer(3, 1, "Stride", 2)
    % reluLayer

    flattenLayer

    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MiniBatchSize', 64, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'Verbose',true ...
);

% Train network
disp(size(data));
net = trainNetwork(data, labels, layers, options);

save('saved_models/dilated_cnn_1d_rate_2kHz_mu_compress.mat', "net");