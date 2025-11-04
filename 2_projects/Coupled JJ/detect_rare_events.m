function detect_rare_events(phi1, phi2, v1, v2)
    % Detect and analyze rare events in junction dynamics
   
    % Phase slip detection (2π jumps)
    phase_slips1 = find(abs(diff(unwrap(phi1))) > pi);
    phase_slips2 = find(abs(diff(unwrap(phi2))) > pi);
   
    % Synchronization events
    phase_diff = mod(phi1 - phi2 + pi, 2*pi) - pi;
    sync_events = find(abs(phase_diff) < 0.1); % Phase difference < 0.1 rad
   
    % Velocity correlation analysis
    window_size = 100;
    corr_coeffs = zeros(length(v1)-window_size, 1);
    for i = 1:length(corr_coeffs)
        idx = i:i+window_size-1;
        corr_coeffs(i) = corr(v1(idx), v2(idx));
    end
   
    % Plot results
    figure('Position', [100, 100, 1000, 800]);
   
    subplot(3,1,1);
    plot(unwrap(phi1), 'b-'); hold on;
    plot(unwrap(phi2), 'r-');
    plot(phase_slips1, unwrap(phi1(phase_slips1)), 'bo', 'MarkerSize', 6);
    plot(phase_slips2, unwrap(phi2(phase_slips2)), 'ro', 'MarkerSize', 6);
    xlabel('Time'); ylabel('Unwrapped Phase');
    legend('\phi_1', '\phi_2', 'Slip J1', 'Slip J2');
    title('Phase Slip Detection');
   
    subplot(3,1,2);
    plot(phase_diff, 'k-'); hold on;
    plot(sync_events, phase_diff(sync_events), 'go', 'MarkerSize', 4);
    xlabel('Time'); ylabel('\phi_1 - \phi_2');
    title('Synchronization Events');
   
    subplot(3,1,3);
    plot(corr_coeffs, 'm-');
    xlabel('Time'); ylabel('Velocity Correlation');
    title('Dynamic Correlation Analysis');
   
    % Statistics
    fprintf('Phase slip statistics:\n');
    fprintf('Junction 1: %d slips\n', length(phase_slips1));
    fprintf('Junction 2: %d slips\n', length(phase_slips2));
    fprintf('Synchronization events: %d\n', length(sync_events));
    fprintf('Mean correlation: %.3f\n', mean(corr_coeffs(isfinite(corr_coeffs))));
end
