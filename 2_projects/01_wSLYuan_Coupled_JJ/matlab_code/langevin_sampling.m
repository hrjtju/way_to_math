function [samples, trajectory] = langevin_sampling(net, data_mean, data_std, n_steps, step_size, varargin)
    % Generate samples using Langevin dynamics - No Deep Learning Toolbox required
    
    % Parse inputs
    p = inputParser;
    addParameter(p, 'n_samples', 100, @isnumeric);
    addParameter(p, 'network_type', 'function', @ischar); % 'function', 'regression', 'custom'
    parse(p, varargin{:});
    params = p.Results;
    
    % Initialize samples
    n_samples = params.n_samples;
    samples = randn(n_samples, 4);
    trajectory = zeros(n_steps, n_samples, 4);
    
    for step = 1:n_steps
        % Get score based on network type
        if isa(net, 'function_handle')
            % net is a function handle
            score = net(samples);
            
        elseif strcmp(params.network_type, 'regression')
            % Standard regression network (assuming net is a matrix/weights)
            score = predict_regression_net(net, samples);
            
        elseif isstruct(net) || isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
            % Try to use predict function for neural networks
            try
                score = predict(net, samples);
            catch
                error(['Network type not supported without Deep Learning Toolbox. '...
                       'Convert network to function handle or use different format.']);
            end
        else
            % Assume net can directly process the samples
            score = net(samples);
        end
        
        % Ensure score has correct dimensions
        if size(score, 2) ~= 4
            score = score';
        end
        if size(score, 1) ~= n_samples
            score = score';
        end
        
        % Langevin update
        noise = randn(size(samples)) * sqrt(2 * step_size);
        samples = samples + step_size * score + noise;
        
        trajectory(step, :, :) = samples;
        
        if mod(step, 100) == 0
            fprintf('Langevin step %d/%d\n', step, n_steps);
        end
    end
    
    % Denormalize samples
    samples = samples .* data_std + data_mean;
end

function score = predict_regression_net(net_weights, samples)
    % Simple neural network forward pass without Deep Learning Toolbox
    % Assumes net_weights is a cell array of weights {W1, b1, W2, b2, ...}
    
    if ~iscell(net_weights)
        % If it's not a cell array, assume it's a simple linear transform
        score = samples * net_weights;
        return;
    end
    
    % Forward pass through layers
    activation = samples';
    
    for i = 1:2:length(net_weights)-1
        W = net_weights{i};
        b = net_weights{i+1};
        
        % Linear transformation
        activation = W * activation + b;
        
        % ReLU activation (except for last layer)
        if i < length(net_weights)-2
            activation = max(0, activation);
        end
    end
    
    score = activation';
end