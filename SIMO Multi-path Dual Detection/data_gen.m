num_cells = 3;
num_antennas_tx = 2; % Transmit antennas
num_antennas_rx = 2; % Receive antennas
modulation_order = 4; % QPSK
c = 3e8; % Speed of light
carrier_freq = 2.4e9; % Carrier frequency
lambda = c/carrier_freq; % Wavelength
d = lambda/2; % Antenna spacing

% Initialize
num_paths = 2; % Number of multipath components
delay = [0, 1e-6]; % Time delay for each path
attenuation = [1, 0.8]; % Attenuation for each path
steering_vector = exp(1i*2*pi*d/lambda*sin((0:num_antennas_tx-1).'*pi/3));

noise_variance = 0.01;

% Transmit Data
num_symbols = 1e4;
data = randi([0 modulation_order-1], num_symbols, num_antennas_tx, num_cells);
qpsk_modulated_data = pskmod(data, modulation_order);

% Simulate MIMO Channel with Time Delay and Attenuation
received_signal = zeros(num_symbols, num_antennas_rx, num_cells);
for k = 1:num_cells
    for p = 1:num_paths
        h = attenuation(p) * steering_vector;
        delayed_signal = [zeros(round(delay(p)*carrier_freq), num_antennas_tx); qpsk_modulated_data(1:end-round(delay(p)*carrier_freq),:,k)];
        received_signal(:,:,k) = received_signal(:,:,k) + delayed_signal * h;
    end
end