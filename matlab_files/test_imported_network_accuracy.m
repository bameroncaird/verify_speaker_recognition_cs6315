series_net = convert_tensorflow_net_to_matlab('saved_models/simple_mlp_mu_law_compress_2_kHz');
net = matlab2nnv(series_net);

return;

%waveform = load('label5.mat');
%waveform = waveform.signal;
%disp(class(waveform));

% pred_labels_nnv = net.evaluate(waveform);
% disp(pred_labels_nnv);
% return;

num_correct_preds = 0;
num_equal_preds = 0;

targets = [];
preds = [];

n_data_lines = 1000;
for i = 1:n_data_lines
    [waveform, target, file_path] = read_data_line(i, 'libri100', 'train', 2000);
    waveform = waveform';
    % NNV
    
    targets = [targets; target];


    pred_labels_nnv = net.evaluate(waveform);
    pred_label_nnv = argmax(pred_labels_nnv);
    preds = [preds; pred_label_nnv];

    % SeriesNetwork
    pred_labels_series = predict(series_net, waveform);
    pred_label_series = argmax(pred_labels_series);

    if target == pred_label_nnv
        num_correct_preds = num_correct_preds + 1;
    end

    if pred_label_series == pred_label_nnv
        num_equal_preds = num_equal_preds + 1;
    end
end

fprintf('\nNumber of correct preds = %d\n', num_correct_preds);
fprintf('\nNumber of equal preds = %d\n', num_equal_preds);

% disp(preds);
% disp(targets);