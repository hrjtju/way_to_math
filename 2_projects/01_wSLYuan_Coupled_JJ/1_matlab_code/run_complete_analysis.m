function run_complete_analysis()
    % Complete analysis pipeline for coupled Josephson junctions
   
    fprintf('=== Coupled Josephson Junction Analysis Pipeline ===\n\n');
   
    % Step 1: Generate training data
    fprintf('1. Generating training data...\n');
    generate_training_data(5000);
   
    % Step 2: Train score network
    fprintf('\n2. Training score network...\n');
    train_score_network();
   
    % Step 3: Load trained network
    fprintf('\n3. Loading trained network...\n');
    load('trained_score_network.mat');
   
    % Step 4: Generate new samples
    fprintf('\n4. Generating new samples with Langevin dynamics...\n');
    [new_samples, trajectory] = langevin_sampling(net, data_mean, data_std, 1000, 0.01);
   
    % Step 5: Compare with simulated data
    fprintf('\n5. Comparing generated and simulated data...\n');
    load('junction_training_data.mat');
   
    figure('Position', [100, 100, 1200, 900]);
   
    subplot(2,3,1);
    histogram2(training_data(:,1), training_data(:,2), 30, 'FaceColor', 'blue');
    xlabel('\phi_1'); ylabel('\phi_2'); title('Simulated Phase Distribution');
    colorbar;
   
    subplot(2,3,2);
    histogram2(new_samples(:,1), new_samples(:,2), 30, 'FaceColor', 'red');
    xlabel('\phi_1'); ylabel('\phi_2'); title('Generated Phase Distribution');
    colorbar;
   
    plot(training_data(:,1), training_data(:,2), 'b.', 'MarkerSize', 1);
    hold on;
    plot(new_samples(:,1), new_samples(:,2), 'r.', 'MarkerSize', 1);
    xlabel('\phi_1'); ylabel('\phi_2');
    legend('Simulated', 'Generated'); title('Phase Space Comparison');
   
    subplot(2,3,4);
    [f1, x1] = ksdensity(training_data(:,3));
    [f2, x2] = ksdensity(new_samples(:,3));
    plot(x1, f1, 'b-', x2, f2, 'r--');
    xlabel('v_1'); ylabel('Density');
    legend('Simulated', 'Generated'); title('Velocity Distribution');
   
    subplot(2,3,5);
    correlation_sim = corr(training_data(:,1:2));
    correlation_gen = corr(new_samples(:,1:2));
    imagesc([correlation_sim, correlation_gen]);
    colorbar; title('Phase Correlation Matrix');
    xticks([1.5, 3.5]); xticklabels({'Sim', 'Gen'});
    yticks([1,2]); yticklabels({'\phi_1', '\phi_2'});
   
    subplot(2,3,6);
    % Calculate KL divergence between distributions
    [f_sim, x_sim] = ksdensity(training_data(:,1));
    [f_gen, x_gen] = ksdensity(new_samples(:,1));
   
    % Interpolate to common grid
    x_common = linspace(min([x_sim, x_gen]), max([x_sim, x_gen]), 100);
    f_sim_interp = interp1(x_sim, f_sim, x_common, 'linear', 0);
    f_gen_interp = interp1(x_gen, f_gen, x_common, 'linear', 0);
   
    % Avoid division by zero
    epsilon = 1e-8;
    f_sim_interp = f_sim_interp + epsilon;
    f_gen_interp = f_gen_interp + epsilon;
   
    % Normalize
    f_sim_interp = f_sim_interp / sum(f_sim_interp);
    f_gen_interp = f_gen_interp / sum(f_gen_interp);
   
    kl_div = sum(f_sim_interp .* log(f_sim_interp ./ f_gen_interp));
   
    plot(x_common, f_sim_interp, 'b-', x_common, f_gen_interp, 'r--');
    xlabel('\phi_1'); ylabel('Density');
    legend('Simulated', 'Generated');
    title(sprintf('Distribution Comparison (KL = %.4f)', kl_div));
   
    fprintf('\n6. Analysis complete!\n');
    fprintf('   KL divergence between distributions: %.4f\n', kl_div);
    fprintf('   Generated %d new samples\n', size(new_samples, 1));
end
