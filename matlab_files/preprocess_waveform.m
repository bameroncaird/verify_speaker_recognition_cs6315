function [waveform_vector] = preprocess_waveform(file_path, target_sample_rate)

    % Return a time-domain waveform vector from an audio path.
    % Return shape: [target_sample_rate] (1 second of audio)
    % Return range: [-1, 1] for each sample
    % downsampling appears to lower the absolute values but not the range

    [waveform_vector, sample_rate] = audioread(file_path);
    waveform_vector = single(waveform_vector); % we are using float32, not float64
    assert(16000 == sample_rate, 'Sample Rate must be 16000 for the VoxCeleb datasets.');
    % disp(class(waveform_vector));
    % disp(size(waveform_vector));

    if sample_rate ~= target_sample_rate
        waveform_vector = resample(waveform_vector, sample_rate, target_sample_rate);
    end

    % select the first second of the audio clip
    % this is relatively low-D, but may still take a while to verify
    waveform_vector = waveform_vector(1:target_sample_rate);

    % I think all of the audios will be long enough
    % find out via an error if there's a shorter one that we need to zero-pad
    assert(length(waveform_vector) == target_sample_rate, 'Need to implement zero-padding');

    % apply mu-law compression (quantize to 2^8 bits instead of 2^16)
    waveform_vector = mu_law_compress(waveform_vector);

    % disp(size(waveform_vector));
    % disp(min(waveform_vector));
    % disp(max(waveform_vector));
end