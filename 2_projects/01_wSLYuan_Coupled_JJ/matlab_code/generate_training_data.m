function [training_data, parameters, metadata] = generate_training_data(n_samples, varargin)
    % Generate training data for score-based modeling of coupled Josephson junctions
    %
    % Inputs:
    %   n_samples - number of data samples to generate
    %   Optional parameters:
    %     'tspan' - simulation time span (default [0 50])
    %     'save_file' - filename to save data (default 'junction_training_data.mat')
    %
    % Outputs:
    %   training_data - [n_samples x 4] array of [phi1, phi2, v1, v2]
    %   parameters - [n_samples x 7] array of system parameters
    %   metadata - structure containing generation parameters
    
    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'tspan', [0 50], @(x) isnumeric(x) && length(x)==2);
    addParameter(p, 'save_file', 'junction_training_data.mat', @ischar);
    parse(p, varargin{:});
    
    % Initialize arrays
    training_data = zeros(n_samples, 4);
    parameters = zeros(n_samples, 7);
    
    % Parameter bounds (could be made configurable)
    param_bounds = struct(...
        'beta', [0.05, 0.15], ...
        'i', [0.1, 0.9], ...
        'kappa', [0.01, 0.16], ...
        'noise', [0.005, 0.025] ...
    );
    
    fprintf('Generating %d training samples...\n', n_samples);
    
    for i = 1:n_samples
        % Generate random parameters
        beta1 = param_bounds.beta(1) + diff(param_bounds.beta) * rand;
        beta2 = param_bounds.beta(1) + diff(param_bounds.beta) * rand;
        i1 = param_bounds.i(1) + diff(param_bounds.i) * rand;
        i2 = param_bounds.i(1) + diff(param_bounds.i) * rand;
        kappa1 = param_bounds.kappa(1) + diff(param_bounds.kappa) * rand;
        kappa2 = param_bounds.kappa(1) + diff(param_bounds.kappa) * rand;
        noise_amp = param_bounds.noise(1) + diff(param_bounds.noise) * rand;
       
        params = [beta1, beta2, i1, i2, kappa1, kappa2, noise_amp];
        
        % Random initial conditions
        init_cond = [2*pi*rand, 0.2*randn, 2*pi*rand, 0.2*randn];
       
        % Simulate system
        [t, phi1, phi2, v1, v2] = simulate_coupled_junctions(params, p.Results.tspan, init_cond);
       
        % Store final state
        training_data(i,:) = [phi1(end), phi2(end), v1(end), v2(end)];
        parameters(i,:) = params;
       
        % Progress reporting
        if mod(i, max(1, round(n_samples/10))) == 0
            fprintf('Progress: %d/%d samples (%.1f%%)\n', i, n_samples, 100*i/n_samples);
        end
    end
   
    % Create metadata
    metadata = struct(...
        'generation_date', datestr(now), ...
        'n_samples', n_samples, ...
        'tspan', p.Results.tspan, ...
        'parameter_bounds', param_bounds, ...
        'description', 'Coupled Josephson junction training data' ...
    );
    
    % Save data
    save(p.Results.save_file, 'training_data', 'parameters', 'metadata', '-v7.3');
    fprintf('Training data saved to %s\n', p.Results.save_file);
    
    % Basic statistics
    fprintf('Data statistics:\n');
    fprintf('  Phase 1: mean=%.3f, std=%.3f\n', mean(training_data(:,1)), std(training_data(:,1)));
    fprintf('  Phase 2: mean=%.3f, std=%.3f\n', mean(training_data(:,2)), std(training_data(:,2)));
    fprintf('  Voltage 1: mean=%.3f, std=%.3f\n', mean(training_data(:,3)), std(training_data(:,3)));
    fprintf('  Voltage 2: mean=%.3f, std=%.3f\n', mean(training_data(:,4)), std(training_data(:,4)));
end

