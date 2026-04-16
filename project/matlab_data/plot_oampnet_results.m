% Publication-style MATLAB plotting script for OAMP vs OAMP-Net results
% Run from matlab_data/ or adjust the file paths below.

clear; clc; close all;

%% Global style
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 13);
set(groot, 'defaultTextFontSize', 13);
set(groot, 'defaultAxesLineWidth', 1.0);
set(groot, 'defaultLineLineWidth', 2.2);
set(groot, 'defaultFigureColor', 'w');

colors.blue   = [0.0000 0.4470 0.7410];
colors.red    = [0.8500 0.3250 0.0980];
colors.green  = [0.4660 0.6740 0.1880];
colors.purple = [0.4940 0.1840 0.5560];
colors.gold   = [0.9290 0.6940 0.1250];
colors.gray   = [0.35   0.35   0.35  ];

markerSize = 7;

%% Load data
ber4 = readtable('ber_4x4_qpsk_rayleigh.csv');
ber8 = readtable('ber_8x8_qpsk_rayleigh.csv');
berCorr = readtable('ber_4x4_qpsk_correlated.csv');
gainTbl = readtable('snr_gain_across_configs.csv');
conv4 = readtable('training_convergence_4x4.csv');
conv8 = readtable('training_convergence_8x8.csv');

%% 1. BER vs SNR
fig1 = figure('Position', [100 100 920 650]);
hold on;
box on;

semilogy(ber4.SNR_dB, ber4.OAMP_BER, ...
    '-o', ...
    'Color', colors.blue, ...
    'MarkerFaceColor', colors.blue, ...
    'MarkerSize', markerSize);

semilogy(ber4.SNR_dB, ber4.OAMPNet_BER, ...
    '-s', ...
    'Color', colors.red, ...
    'MarkerFaceColor', colors.red, ...
    'MarkerSize', markerSize);

semilogy(ber8.SNR_dB, ber8.OAMP_BER, ...
    '--o', ...
    'Color', colors.green, ...
    'MarkerFaceColor', colors.green, ...
    'MarkerSize', markerSize);

semilogy(ber8.SNR_dB, ber8.OAMPNet_BER, ...
    '--s', ...
    'Color', colors.purple, ...
    'MarkerFaceColor', colors.purple, ...
    'MarkerSize', markerSize);

grid on;
ax = gca;
ax.GridAlpha = 0.18;
ax.MinorGridAlpha = 0.10;
ax.YMinorGrid = 'on';
ax.XMinorGrid = 'on';
ax.TickDir = 'out';
ax.Layer = 'top';

xlabel('SNR (dB)', 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontWeight', 'bold');
title('BER vs SNR for OAMP and OAMP-Net', 'FontWeight', 'bold');

legend( ...
    {'4x4 OAMP', '4x4 OAMP-Net', '8x8 OAMP', '8x8 OAMP-Net'}, ...
    'Location', 'southwest', ...
    'Box', 'off' ...
);

xlim([min(ber4.SNR_dB) max(ber4.SNR_dB)]);

%% 2. SNR Gain Across Antenna Configurations
fig2 = figure('Position', [150 140 820 560]);
box on;

b = bar(categorical(gainTbl.Antenna_Config), gainTbl.SNR_Gain_dB, 0.52, ...
    'FaceColor', 'flat', ...
    'EdgeColor', 'none');

b.CData = [
    colors.blue
    colors.red
];

grid on;
ax = gca;
ax.GridAlpha = 0.18;
ax.TickDir = 'out';
ax.Layer = 'top';

xlabel('Antenna Configuration', 'FontWeight', 'bold');
ylabel('SNR Gain at BER = 10^{-3} (dB)', 'FontWeight', 'bold');
title('SNR Gain Across Antenna Configurations', 'FontWeight', 'bold');

for i = 1:height(gainTbl)
    text(i, gainTbl.SNR_Gain_dB(i) + 0.05, ...
        sprintf('%.2f dB', gainTbl.SNR_Gain_dB(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontWeight', 'bold', ...
        'Color', colors.gray);
end

ylim([0, max(gainTbl.SNR_Gain_dB) * 1.25]);

%% 3. Training Convergence Analysis
fig3 = figure('Position', [200 180 960 650]);
hold on;
box on;

plot(conv4.Epoch, conv4.Train_Loss, ...
    '-o', ...
    'Color', colors.blue, ...
    'MarkerFaceColor', colors.blue, ...
    'MarkerSize', 6);

plot(conv4.Epoch, conv4.Val_Loss, ...
    '-s', ...
    'Color', colors.red, ...
    'MarkerFaceColor', colors.red, ...
    'MarkerSize', 6);

plot(conv8.Epoch, conv8.Train_Loss, ...
    '--o', ...
    'Color', colors.green, ...
    'MarkerFaceColor', colors.green, ...
    'MarkerSize', 6);

plot(conv8.Epoch, conv8.Val_Loss, ...
    '--s', ...
    'Color', colors.purple, ...
    'MarkerFaceColor', colors.purple, ...
    'MarkerSize', 6);

grid on;
ax = gca;
ax.GridAlpha = 0.18;
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
ax.TickDir = 'out';
ax.Layer = 'top';

xlabel('Epoch', 'FontWeight', 'bold');
ylabel('MSE Loss', 'FontWeight', 'bold');
title('Training Convergence Analysis', 'FontWeight', 'bold');

legend( ...
    {'4x4 Train', '4x4 Validation', '8x8 Train', '8x8 Validation'}, ...
    'Location', 'northeast', ...
    'Box', 'off' ...
);

xlim([min(conv4.Epoch) max(conv4.Epoch)]);

%% Export high-resolution figures
exportgraphics(fig1, 'ber_vs_snr_publication.png', 'Resolution', 300);
exportgraphics(fig2, 'snr_gain_across_configs_publication.png', 'Resolution', 300);
exportgraphics(fig3, 'training_convergence_publication.png', 'Resolution', 300);

savefig(fig1, 'ber_vs_snr_publication.fig');
savefig(fig2, 'snr_gain_across_configs_publication.fig');
savefig(fig3, 'training_convergence_publication.fig');

%% 4. Rayleigh vs Correlated Channel (4x4)
fig4 = figure('Position', [240 220 920 650]);
hold on;
box on;

semilogy(ber4.SNR_dB, ber4.OAMP_BER, ...
    '-o', 'Color', colors.blue, 'MarkerFaceColor', colors.blue, 'MarkerSize', markerSize);
semilogy(ber4.SNR_dB, ber4.OAMPNet_BER, ...
    '-s', 'Color', colors.red, 'MarkerFaceColor', colors.red, 'MarkerSize', markerSize);
semilogy(berCorr.SNR_dB, berCorr.OAMP_BER, ...
    '--o', 'Color', colors.green, 'MarkerFaceColor', colors.green, 'MarkerSize', markerSize);
semilogy(berCorr.SNR_dB, berCorr.OAMPNet_BER, ...
    '--s', 'Color', colors.purple, 'MarkerFaceColor', colors.purple, 'MarkerSize', markerSize);

grid on;
ax = gca;
ax.GridAlpha = 0.18;
ax.MinorGridAlpha = 0.10;
ax.YMinorGrid = 'on';
ax.XMinorGrid = 'on';
ax.TickDir = 'out';
ax.Layer = 'top';

xlabel('SNR (dB)', 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER)', 'FontWeight', 'bold');
title('4x4 Rayleigh vs Correlated Channel Performance', 'FontWeight', 'bold');
legend({'Rayleigh OAMP','Rayleigh OAMP-Net','Correlated OAMP','Correlated OAMP-Net'}, ...
    'Location', 'southwest', 'Box', 'off');
xlim([min(ber4.SNR_dB) max(ber4.SNR_dB)]);

exportgraphics(fig4, 'rayleigh_vs_correlated_4x4_publication.png', 'Resolution', 300);
savefig(fig4, 'rayleigh_vs_correlated_4x4_publication.fig');
