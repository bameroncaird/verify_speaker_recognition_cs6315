% Verify all samples from the LibriSpeech training list
fileID = fopen('cnn_1d_rate_2kHz_mu_compress_eps_0.5_approx.csv', 'w');
fprintf(fileID, 'robust, target, y_pred, time_setup, time_star, time_verify, line_number, file_path, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10\n');

sample_rate = 2000; % Hz

% Load network
% series_net = convert_tensorflow_net_to_matlab('saved_models/mlp_adversarial_config2_mult0.25');
% net = matlab2nnv(series_net);

% Conv1d and Conv2d were implemented in matlab
series_net = load("saved_models\cnn_1d_rate_2kHz_mu_compress.mat");
net = matlab2nnv(series_net.net);

% magnitude of the disturbance
eps = 0.5;

% ms = verify_single_audio_waveform(net, 0.5, 1, 2000, 1);
% return;

for i = 1:1000
    %if 0 == mod(i, 50)
    fprintf('\nProcessed %d / %d files', i, 1000);
    %end

    % write a line of the save file for analysis
    ms = verify_single_audio_waveform(net, eps, i, sample_rate, 0);
    %ms = verify_single_mel_spectrogram(net, eps, i, 0);
    fprintf(fileID, '%d, %d, %d, %f, %f, %f, %d, %s', ms.robust, ms.target, ms.original_predict_label, ms.setup_time, ms.star_time, ms.verify_time, i, ms.file_path);
    
    % add on original labels separately
    for j = 1:numel(ms.original_predict_labels)
        fprintf(fileID, ', %f', ms.original_predict_labels(j));
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