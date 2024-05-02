function [audio_features, true_label, file_path] = read_data_line(line_number, dataset_name, dataset_partition, ~)

    % build file name from input args
    data_list_path = sprintf('data_lists/%s_%s.txt', dataset_name, dataset_partition);
    % fprintf('%s', data_list_path);
    
    fid = fopen(data_list_path, 'r');
    data = textscan(fid, '%d %s');
    fclose(fid);

    true_label = data{1}(line_number);
    true_label = true_label + 1;

    file_path = data{2}(line_number);
    file_path = file_path{1};
    file_path = get_absolute_file_path(file_path);

    audio_features = load(file_path);
    audio_features = audio_features.signal;

    %audio_features = preprocess_waveform(file_path, sample_rate);
end