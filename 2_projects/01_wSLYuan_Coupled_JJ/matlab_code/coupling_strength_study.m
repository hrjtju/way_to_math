function coupling_strength_study()
    % Study effect of coupling strength on synchronization
    
    kappa_values = linspace(0, 0.2, 10);
    sync_measures = zeros(length(kappa_values), 3);
    
    for i = 1:length(kappa_values)
        kappa = kappa_values(i);
        params = [0.1, 0.1, 0.5, 0.5, kappa, kappa, 0.01];
        tspan = [0 200];
        init_cond = [0, 0.1, 1.5, 0]; % Different initial phases
        
        [t, phi1, phi2, v1, v2] = simulate_coupled_junctions(params, tspan, init_cond);
        
        % Use last 1000 points for analysis (or adjust if not enough points)
        if length(t) > 1000
            idx = length(t)-1000:length(t);
        else
            idx = 1:length(t);
        end
        
        v1_analysis = v1(idx);
        v2_analysis = v2(idx);
        phi1_analysis = phi1(idx);
        phi2_analysis = phi2(idx);
        
        % Calculate synchronization measures
        phase_diff = mod(phi1_analysis - phi2_analysis + pi, 2*pi) - pi;
        sync_measures(i,1) = std(phase_diff); % Phase locking
        
        % Velocity correlation (manual calculation without corr function)
        sync_measures(i,2) = manual_correlation(v1_analysis, v2_analysis);
        
        sync_measures(i,3) = mean(abs(phase_diff)); % Mean phase difference
    end
    
    % Plot results
    figure;
    subplot(1,3,1);
    plot(kappa_values, sync_measures(:,1), 'bo-', 'LineWidth', 2);
    xlabel('Coupling Strength \kappa'); 
    ylabel('Phase Difference STD');
    title('Phase Locking');
    grid on;
    
    subplot(1,3,2);
    plot(kappa_values, sync_measures(:,2), 'ro-', 'LineWidth', 2);
    xlabel('Coupling Strength \kappa'); 
    ylabel('Velocity Correlation');
    title('Velocity Synchronization');
    grid on;
    ylim([-1.1, 1.1]);
    
    subplot(1,3,3);
    plot(kappa_values, sync_measures(:,3), 'go-', 'LineWidth', 2);
    xlabel('Coupling Strength \kappa'); 
    ylabel('Mean Phase Difference');
    title('Average Phase Separation');
    grid on;
    
    sgtitle('Effect of Coupling Strength on Synchronization');
end

function r = manual_correlation(x, y)
    % Manual calculation of Pearson correlation coefficient
    % Remove any NaN or Inf values
    valid_idx = isfinite(x) & isfinite(y);
    x = x(valid_idx);
    y = y(valid_idx);
    
    if length(x) < 2
        r = NaN;
        return;
    end
    
    % Calculate means
    mean_x = mean(x);
    mean_y = mean(y);
    
    % Calculate covariance and variances
    covariance = sum((x - mean_x) .* (y - mean_y));
    var_x = sum((x - mean_x).^2);
    var_y = sum((y - mean_y).^2);
    
    % Calculate correlation coefficient
    if var_x > 0 && var_y > 0
        r = covariance / sqrt(var_x * var_y);
    else
        r = 0;
    end
    
    % Ensure correlation is within [-1, 1]
    r = max(-1, min(1, r));
end