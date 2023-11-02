'''The tool chains required for the processing
'''

import numpy as np

def modulation(
    data, # a data symbol list which represents a series of symbols
    mod_scheme = 2, # the modulation scheme, assume to be 2 - QPSK
)-> list:
    
    modulated_symbol = list()
    if mod_scheme == 2:
        mod_mapping = [
            0.707+0.707j,
            0.707-0.707j,
            -0.707+0.707j,
            -0.707-0.707j
        ]
    for _data in data:
        modulated_symbol.append(mod_mapping[_data])

    return(np.array(modulated_symbol))

def steering_vector(N, theta, lambda_, d):
    """
    Compute the steering vector for a ULA.
    
    Parameters:
    - N: Number of transmit antennas.
    - theta: Angle of arrival (AoA) in radians.
    - lambda_: Wavelength.
    - d: Antenna interval (spacing between antennas).

    Returns:
    - s: Steering vector of size M.
    """

    k = 2 * np.pi / lambda_  # Wave number
    n = np.arange(N)  # Antenna index array
    s = np.exp(1j * k * n * d * np.sin(theta))
    return s