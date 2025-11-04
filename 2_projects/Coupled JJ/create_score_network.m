function net = create_score_network_simple(input_dim, hidden_layers, varargin)
    % Create neural network for score function estimation (single file version)
    
    % Parse optional parameters
    p = inputParser;
    addParameter(p, 'OutputActivation', 'tanh', @ischar);
    addParameter(p, 'UseBatchNorm', true, @islogical);
    parse(p, varargin{:});
    
    layers = [featureInputLayer(input_dim, 'Name', 'input')];
    
    % Add hidden layers
    for i = 1:length(hidden_layers)
        layers = [layers
            fullyConnectedLayer(hidden_layers(i), 'Name', sprintf('fc%d', i))];
        
        if p.Results.UseBatchNorm
            layers = [layers
                batchNormalizationLayer('Name', sprintf('bn%d', i))];
        end
        
        % 使用函数层代替自定义类
        layers = [layers
            functionLayer(@swish, 'Name', sprintf('swish%d', i))];
    end
    
    % Output layer
    layers = [layers
        fullyConnectedLayer(input_dim, 'Name', 'output')];
    
    switch lower(p.Results.OutputActivation)
        case 'tanh'
            layers = [layers
                tanhLayer('Name', 'tanh_out')];
        case 'linear'
            % No activation
        case 'none'
            % No activation
    end
    
    net = dlnetwork(layers);
    
    fprintf('Created score network with architecture:\n');
    fprintf('Input: %d', input_dim);
    for i = 1:length(hidden_layers)
        fprintf(' -> %d (swish)', hidden_layers(i));
    end
    fprintf(' -> %d (%s)\n', input_dim, p.Results.OutputActivation);
end

% 内联 swish 函数
function y = swish(x)
    % swish activation function
    y = x .* (1 ./ (1 + exp(-x)));
end