% Train a Conv2D network in matlab

n_train_points = 1000;
data = zeros(128, 126, 1, n_train_points);
labels = zeros(n_train_points, 1);

for i = 1:n_train_points
    [mel_spec, true_label, file_path] = read_data_line(i, 'libri100', 'train', 2000);
    data(:, :, :, i) = mel_spec;
    labels(i) = true_label;
end
labels = categorical(labels);

disp(size(data));
disp(min(data(:)));
disp(max(data(:)));
disp(size(labels));

layers = [
    imageInputLayer([128 126 1], 'Name', 'inputs');

    convolution2dLayer([3, 3], 64, 'Padding', 'same')
    reluLayer

    convolution2dLayer([3, 3], 1, 'Stride', [2, 2])
    reluLayer

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

save('saved_models/TMP_cnn_2d_mel_spec.mat', "net");