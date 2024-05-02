function metric_struct = verify_single_audio_waveform(nnv_net, eps, line_number, sample_rate, plot)
    %% Keep track of key metrics
    metric_struct = struct;

    %% Load data sample (we are verifying a single example)
    % sample in this case should be a waveform signal (time domain vector)
    % load from the training set (the model is still bad)
    t = tic;
    [waveform, target, file_path] = read_data_line(line_number, 'libri100', 'train', sample_rate);
    waveform = single(waveform); 

    metric_struct.target = target;
    metric_struct.file_path = file_path;

    % pass original input through the network (NNV network)
    pred_labels = nnv_net.evaluate(waveform');
    pred_label = argmax(pred_labels);

    metric_struct.original_predict_label = pred_label;
    metric_struct.original_predict_labels = pred_labels;

    setup_time = toc(t);
    %fprintf('\nSetup time = %f\n', setup_time);
    metric_struct.setup_time = setup_time;

    %% Create input Star set (and time it)
    t = tic;

    % Define perturbation: original waveform +- disturbance (L_inf epsilon)
    % in real-world setting, you'd want it to be small enough to not be heard
    % our waveforms are in [-128, 127] (downsampling plus mu-law quantization)
    % the perturbation is defined in the input args as eps
    ones_ = ones(size(waveform), 'single'); % waveform is a single (float32)
    disturbance = eps .* ones_;

    % ensure the values are within the valid range for audio samples
    % this is [-128, 127] (seems pretty consistent across libraries)
    % lb => Lower Bound
    % ub => Upper Bound
    lb_min = -128 * ones(size(waveform));
    ub_max = 127 * ones(size(waveform));
    lb_clip = max((waveform - disturbance), lb_min);
    ub_clip = min((waveform + disturbance), ub_max);

    % use Star as inputs are vectors
    lb_clip = lb_clip';
    ub_clip = ub_clip';
    IS = ImageStar(lb_clip, ub_clip);

    star_time = toc(t);
    %fprintf('\nTime to create star set = %f\n', star_time);
    metric_struct.star_time = star_time;

    %% Verification with L_infinity attack
    % First, we need to define the reachability options
    reachOptions = struct; % initialize
    reachOptions.reachMethod = 'approx-star';

    % Verification
    t = tic;
    res_approx = nnv_net.verify_robustness(IS, reachOptions, target);
    
    if res_approx == 1
        disp("Neural network is verified to be robust!")
    else
        disp("Unknown result")
    end
    metric_struct.robust = res_approx;
    
    verify_time = toc(t);
    fprintf('\nVerification time = %f\n', verify_time);
    metric_struct.verify_time = verify_time;

    %% Plot the tick marks on the box plot
    if 1 ~= plot
        return;
    end
    R = nnv_net.reachSet{end};
    [lb_out, ub_out] = R.getRanges;
    lb_out = squeeze(lb_out);
    ub_out = squeeze(ub_out);
    mid_range = (lb_out + ub_out)/2;
    range_size = ub_out - mid_range;
    x = [1 2 3 4 5 6 7 8 9 10];
    figure;
    errorbar(x, mid_range, range_size, '.');
    hold on;
    xlim([0.5 10.5]);
    scatter(x, pred_labels, 'x', 'MarkerEdgeColor', 'r');
end