function train_score_network()
    % Train score network using denoising score matching
    % No Deep Learning Toolbox required
    
    % Load training data
    load('junction_training_data.mat');
    data = training_data;
   
    % Normalize data
    data_mean = mean(data, 1);
    data_std = std(data, 1);
    data_std(data_std == 0) = 1; % Avoid division by zero
    data_normalized = (data - data_mean) ./ data_std;
   
    % Create network manually
    input_size = 4;
    layer_sizes = [128, 128, 64, input_size]; % Output same size as input for score
    net = create_manual_network(input_size, layer_sizes);
   
    % Training parameters
    num_epochs = 100;
    batch_size = 64;
    learning_rate = 1e-4;
   
    % Noise schedule
    sigma_min = 0.01;
    sigma_max = 0.5;
   
    num_samples = size(data_normalized, 1);
    
    % Track losses
    losses = zeros(num_epochs, 1);

    for epoch = 1:num_epochs
        total_loss = 0;
        num_batches = 0;
       
        % Shuffle data
        idx = randperm(num_samples);
        data_shuffled = data_normalized(idx, :);
       
        % Mini-batch training
        for batch_start = 1:batch_size:num_samples
            batch_end = min(batch_start + batch_size - 1, num_samples);
            batch_data = data_shuffled(batch_start:batch_end, :);
            batch_size_actual = size(batch_data, 1);
           
            % Random noise levels
            sigma = sigma_min + (sigma_max - sigma_min) * rand(batch_size_actual, 1);
           
            % Add noise to data
            noise = randn(size(batch_data)) .* sigma;
            noisy_data = batch_data + noise;
           
            % Compute target score: -noise/sigma^2
            target_score = -noise ./ (sigma.^2);
           
            % Forward pass
            [pred_score, activations] = manual_forward(net, noisy_data);
           
            % Compute loss (denoising score matching)
            loss = mean((pred_score - target_score).^2, 'all');
           
            % Backward pass
            gradients = manual_backward(net, activations, pred_score, target_score);
           
            % Update network
            net = manual_update(net, gradients, learning_rate);
           
            total_loss = total_loss + loss;
            num_batches = num_batches + 1;
        end
       
        avg_loss = total_loss / num_batches;
        losses(epoch) = avg_loss;
        fprintf('Epoch %d/%d, Loss: %.6f\n', epoch, num_epochs, avg_loss);
       
        if mod(epoch, 20) == 0
            % Save checkpoint
            save(sprintf('score_net_epoch_%d.mat', epoch), 'net', 'data_mean', 'data_std', 'losses');
        end
    end
   
    save('trained_score_network.mat', 'net', 'data_mean', 'data_std', 'losses');
    
    % Plot training loss
    figure;
    plot(1:num_epochs, losses);
    xlabel('Epoch');
    ylabel('Loss');
    title('Training Loss');
    grid on;
end

function net = create_manual_network(input_size, layer_sizes)
    % Create a manual neural network
    net = struct();
    net.num_layers = length(layer_sizes);
    net.weights = cell(1, net.num_layers);
    net.biases = cell(1, net.num_layers);
    
    % Initialize weights and biases
    for i = 1:net.num_layers
        if i == 1
            input_dim = input_size;
        else
            input_dim = layer_sizes(i-1);
        end
        output_dim = layer_sizes(i);
        
        % He initialization for ReLU
        net.weights{i} = randn(output_dim, input_dim) * sqrt(2/input_dim);
        net.biases{i} = zeros(output_dim, 1);
    end
end

function [output, activations] = manual_forward(net, input)
    % Manual forward pass
    % input: batch_size x input_dim
    activations = cell(1, net.num_layers + 1);
    activations{1} = input'; % Convert to input_dim x batch_size
    
    for i = 1:net.num_layers
        % Linear transformation
        z = net.weights{i} * activations{i} + net.biases{i};
        
        % ReLU for hidden layers, linear for output layer
        if i < net.num_layers
            activations{i+1} = max(0, z); % ReLU
        else
            activations{i+1} = z; % Linear output (for score)
        end
    end
    
    output = activations{end}'; % Convert back to batch_size x output_dim
end

function gradients = manual_backward(net, activations, pred, target)
    % Manual backward pass
    % pred, target: batch_size x output_dim
    
    batch_size = size(pred, 1);
    delta = (pred - target)' / batch_size; % output_dim x batch_size
    
    gradients = struct();
    gradients.weights = cell(1, net.num_layers);
    gradients.biases = cell(1, net.num_layers);
    
    for i = net.num_layers:-1:1
        % Gradient for biases
        gradients.biases{i} = sum(delta, 2);
        
        % Gradient for weights
        gradients.weights{i} = delta * activations{i}';
        
        if i > 1
            % Backpropagate through ReLU
            delta = (net.weights{i}' * delta) .* (activations{i} > 0);
        end
    end
end

function net = manual_update(net, gradients, learning_rate)
    % Manual parameter update
    for i = 1:net.num_layers
        net.weights{i} = net.weights{i} - learning_rate * gradients.weights{i};
        net.biases{i} = net.biases{i} - learning_rate * gradients.biases{i};
    end
end