function absolute_file_path = get_absolute_file_path(relative_fpath)

    % some paths in the data lists were built for training in Python
    % that training was also on a different machine
    % we'll need to slightly modify the path for each dataset

    if contains(relative_fpath, 'LibriSpeech')
        parts = strsplit(relative_fpath, 'voice_data/');
        relative_fpath = strjoin(parts(2:end), 'LibriSpeech/');
        base_dir = 'C:\Users\camer\Documents\data\';
        absolute_file_path = fullfile(base_dir, relative_fpath);

        % change from *.wav to *.mat
        parts2 = strsplit(absolute_file_path, '.');
        fpath_no_ext = parts2{1};
        absolute_file_path = strcat(fpath_no_ext, '.mat');
        %disp(absolute_file_path);

        return;
    end

    if contains(relative_fpath, 'voxceleb2')
        base_dir = 'C:\Users\camer\Documents\data\voxceleb2_subset\';
        absolute_file_path = fullfile(base_dir, relative_fpath);
        return;
    end

    fprintf('\nPath did not contain LibriSpeech or VoxCeleb2, not able to process...\n');
    absolute_file_path = '';
end