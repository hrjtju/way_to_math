function inferred_params = infer_parameters(observed_data, net, data_mean, data_std)
    % Infer junction parameters from observed dynamics using score function
   
    n_observations = size(observed_data, 1);
    inferred_params = zeros(n_observations, 7);
   
    for i = 1:n_observations
        % Normalize observation
        obs_normalized = (observed_data(i,:) - data_mean) ./ data_std;
       
        % Get score at observation
        dl_obs = dlarray(obs_normalized', 'CB');
        score = extractdata(forward(net, dl_obs))';
       
        % Simple inference based on score patterns
        % In practice, this would use more sophisticated methods
        phase_diff = obs_normalized(1) - obs_normalized(2);
        vel_diff = obs_normalized(3) - obs_normalized(4);
       
        % Heuristic parameter inference (replace with proper Bayesian inference)
        inferred_params(i,1) = 0.1 + 0.05 * score(3); % beta1 from v1 score
        inferred_params(i,2) = 0.1 + 0.05 * score(4); % beta2 from v2 score
        inferred_params(i,5) = 0.05 + 0.1 * abs(score(1)); % kappa1 from phi1 score
        inferred_params(i,6) = 0.05 + 0.1 * abs(score(2)); % kappa2 from phi2 score
       
        % Current bias estimation
        inferred_params(i,3) = 0.3 + 0.4 * tanh(obs_normalized(3)); % i1 from v1
        inferred_params(i,4) = 0.3 + 0.4 * tanh(obs_normalized(4)); % i2 from v2
        inferred_params(i,7) = 0.01; % Fixed noise level
    end
   
    fprintf('Parameter inference completed for %d observations\n', n_observations);
end
