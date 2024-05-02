function [series_network] = convert_tensorflow_net_to_matlab(weights_dir)

    % importNetworkFromTensorFlow doesn't work that well
    % this is a wrapper function to make it work for us
    
    original_net = importNetworkFromTensorFlow(weights_dir);

    layers = [
        original_net.Layers(1:end)
        classificationLayer
    ];

    % modify for 1D CNN
    % layers(6) = [];
    % layers = [
    %     layers(1:5)
    %     flattenLayer
    %     layers(6:end)
    % ];

    series_network = SeriesNetwork(layers);

end