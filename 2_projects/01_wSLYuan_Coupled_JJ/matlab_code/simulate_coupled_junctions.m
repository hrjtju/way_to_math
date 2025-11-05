
function [t, phi1, phi2, v1, v2] = simulate_coupled_junctions(params, tspan, init_cond)
    % Simulate coupled Josephson junctions with thermal noise
    %
    % Inputs:
    %   params = [beta1, beta2, i1, i2, kappa1, kappa2, noise_amp]
    %   tspan  = time vector or [t_start, t_end]
    %   init_cond = [phi1_0, v1_0, phi2_0, v2_0]
    %
    % Outputs:
    %   t - time vector
    %   phi1, phi2 - phase variables
    %   v1, v2 - voltage variables (dphi/dt)
    
    % Parameter unpacking with meaningful names
    beta1 = params(1); beta2 = params(2);
    i1 = params(3); i2 = params(4);
    kappa1 = params(5); kappa2 = params(6);
    noise_amp = params(7);
    
    % Validate inputs
    if length(init_cond) ~= 4
        error('Initial conditions must have 4 elements: [phi1, v1, phi2, v2]');
    end
    
    % Persistent noise for consistent random number generation if needed
    persistent noise_seed;
    if isempty(noise_seed)
        noise_seed = rng('shuffle');
    end

    function dydt = junction_ode(t, y)
        phi1 = y(1); v1 = y(2); 
        phi2 = y(3); v2 = y(4);
       
        % Thermal noise (Gaussian white noise)
        noise1 = noise_amp * randn;
        noise2 = noise_amp * randn;
       
        % Coupled Josephson junction equations
        dphi1 = v1;
        dv1 = i1 + noise1 - beta1*v1 - sin(phi1) + kappa1*(phi2 - phi1);
       
        dphi2 = v2;
        dv2 = i2 + noise2 - beta2*v2 - sin(phi2) + kappa2*(phi1 - phi2);
       
        dydt = [dphi1; dv1; dphi2; dv2];
    end

    % Solve ODE with appropriate solver
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'Stats', 'off');
    
    % Use ode15s for stiff problems, otherwise ode45
    try
        [t, Y] = ode45(@junction_ode, tspan, init_cond, options);
    catch ME
        warning('ode45 failed, trying ode15s: %s', ME.message);
        [t, Y] = ode15s(@junction_ode, tspan, init_cond, options);
    end
   
    % Extract results
    phi1 = Y(:,1); 
    v1 = Y(:,2); 
    phi2 = Y(:,3); 
    v2 = Y(:,4);
end