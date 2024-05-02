% Verify a subset of samples from the LibriSpeech training list

fileID = fopen('subset_100_cnn_1d_rate_2kHz_mu_compress_eps_0.05_approx.csv', 'w');
fprintf(fileID, 'robust, target, y_pred, time_setup, time_star, time_verify, line_number, file_path, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10\n');

sample_rate = 2000; % Hz

% Load network
% series_net = convert_tensorflow_net_to_matlab('saved_models/mlp_adversarial_config2_mult0.25');
% net = matlab2nnv(series_net);

% Conv1d and Conv2d were implemented in matlab
series_net = load("saved_models\cnn_1d_rate_2kHz_mu_compress.mat");
net = matlab2nnv(series_net.net);

% magnitude of the disturbance
eps = 0.05;

% ms = verify_single_audio_waveform(net, 0.5, 1, 2000, 1);
% return;

indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910];
for j = 1:length(indices)
    fprintf('\nProcessed %d / %d files', j, length(indices));

    index = indices(j);
    ms = verify_single_audio_waveform(net, eps, index, sample_rate, 0);
    fprintf(fileID, '%d, %d, %d, %f, %f, %f, %d, %s', ms.robust, ms.target, ms.original_predict_label, ms.setup_time, ms.star_time, ms.verify_time, i, ms.file_path);

    % add on original labels separately
    for k = 1:numel(ms.original_predict_labels)
        fprintf(fileID, ', %f', ms.original_predict_labels(k));
    end
    fprintf(fileID, '\n');
end
fclose(fileID);

%% Debugging
% disp(class(waveform));
% disp(size(waveform));
% disp(target);
% disp(file_path);
%waveform = waveform';
%waveform = single(waveform);
%target = 8;
% waveform = load('label5.mat');
% waveform = waveform.signal;
% disp(class(waveform));

%% Sanity check: pass original input through network
% we can take the argmax of the output and compare to the original label
% let's use the NNV version of the model for this check
% pred_labels = net.evaluate(waveform);
% pred_label = argmax(pred_labels);
% fprintf('\nGround truth label = %d; Predicted label = %d\n', target, pred_label);
% fprintf('\nPredicted class probabilities:\n');
% disp(pred_labels);
% if pred_label ~= target
%     fprintf('wrong');
%     %return;
% end
% metric_struct.pred_label_original_input = pred_label;