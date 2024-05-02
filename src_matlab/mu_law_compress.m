function signal = mu_law_compress(waveform)
    % https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py
    % implementing from scratch because matlab's function doesn't do what
    % we want
    mu = 255;
    waveform = min(abs(waveform), 1.0);
    
    % x = sign(x) * ((ln(1 + mu * |x|)) / (ln(1 + mu)))
    magnitude = log(1 + mu * abs(waveform)) ./ log(1 + mu);
    signal = sign(waveform) .* magnitude;

    % quantize to 256 values
    % this is the whole point
    % the nonlinear compression algorithm effectively reduces the bit depth
    % now each sample has 2^8 = 256 (mu) values and not 2^16 as originally
    % follow official librosa implementation
    % https://librosa.org/doc/main/_modules/librosa/core/audio.html#mu_compress

    % define bins for digitization
    edges = linspace(-1, 1, 1 + mu);

    % adjust last bin for edge case (include upper bound)
    edges(end) = edges(end) + eps(edges(end));

    % group each element of the input signal (the companded one) into bins
    y = discretize(signal, edges);

    % map to [0, 255]
    signal = y - round((1 + mu) / 2);

    disp(size(signal));
    disp(min(signal));
    disp(max(signal));
end