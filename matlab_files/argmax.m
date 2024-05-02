function [index] = argmax(input_vector)
    max_value = -Inf;
    index = 0;
    for i = 0:9
        % the models are trained in python
        % so the labels are 0-indexed
        value = input_vector(i + 1);
        if value > max_value
            max_value = value;
            index = i;
        end
    end

    % now, convert the label to 1-index for matlab
    index = index + 1;
end