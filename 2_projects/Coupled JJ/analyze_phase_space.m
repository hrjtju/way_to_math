function analyze_phase_space()
    % Analyze phase space dynamics for different coupling strengths
   
    params = [0.1, 0.1, 0.5, 0.5, 0.05, 0.05, 0.01];
    tspan = [0 100];
    init_cond = [0, 0.1, 0.2, 0];
   
    [t, phi1, phi2, v1, v2] = simulate_coupled_junctions(params, tspan, init_cond);
   
    % Plot results
    figure('Position', [100, 100, 1200, 800]);
   
    subplot(2,3,1);
    plot(t, phi1, 'b-', t, phi2, 'r-');
    xlabel('Time'); ylabel('Phase'); legend('\phi_1', '\phi_2');
    title('Phase Evolution');
   
    subplot(2,3,2);
    plot(t, v1, 'b-', t, v2, 'r-');
    xlabel('Time'); ylabel('Phase Velocity'); legend('v_1', 'v_2');
    title('Velocity Evolution');
   
    subplot(2,3,3);
    plot(phi1, v1, 'b-'); hold on;
    plot(phi2, v2, 'r-');
    xlabel('Phase'); ylabel('Velocity');
    title('Phase Space Trajectories'); legend('J1', 'J2');
   
    subplot(2,3,4);
    plot(phi1, phi2, 'k-');
    xlabel('\phi_1'); ylabel('\phi_2');
    title('Phase Correlation');
   
    subplot(2,3,5);
    phase_diff = mod(phi1 - phi2 + pi, 2*pi) - pi;
    plot(t, phase_diff, 'g-');
    xlabel('Time'); ylabel('\phi_1 - \phi_2');
    title('Phase Difference');
   
    subplot(2,3,6);
    histogram2(phi1, phi2, 50, 'FaceColor', 'flat');
    colorbar; xlabel('\phi_1'); ylabel('\phi_2');
    title('Joint Phase Distribution');
end
