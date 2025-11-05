
clear; clc;

params = [0.1, 0.1, 0.8, 0.8, 0.05, 0.05, 0.01];
tspan = [0, 100];
init_cond = [0, 0, 0.1, 0];

[t, phi1, phi2, v1, v2] = simulate_coupled_junctions(params, tspan, init_cond);

figure;
subplot(2,2,1);
plot(t, phi1);
title('\phi_1');
xlabel('t'); ylabel('\phi_1');

subplot(2,2,2);
plot(t, phi2);
title('\phi_2');
xlabel('t'); ylabel('\phi_2');

subplot(2,2,3);
plot(t, v1);
title(' v_1');
xlabel('t'); ylabel('v_1');

subplot(2,2,4);
plot(t, v2);
title('v_2');
xlabel('t'); ylabel('v_2');