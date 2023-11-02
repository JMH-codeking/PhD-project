from utils import *

def signal_gen(
    Nr,
    K = 10000,
    M = 4,
    SNR = 2,
):
    '''generating signal from one single OFDM carrier

    return a noisy signal
    '''
    
    f_c = 3e8
    theta = np.random.rand() * np.pi / 4 # create a 
    lambda_1 = 3e8 / f_c
    d = lambda_1 / 2


    b = np.random.randn() + 1j * np.random.randn()
    tau = np.random.randn() * 1e-9
    data = np.random.randint(0, 4, [M * K])

    
    x_nk = modulation(data)
    x_nk = x_nk.reshape(M, K)
    data_label = data.reshape(M,K)

    steering_vec_r = steering_vector(Nr, theta, lambda_1, d)

    y = np.zeros(shape = (M, K, Nr), dtype=np.complex64)
    y_n = y

    for k in range (K):
        for m in range (M):
            y[m, k] = b * np.exp(-1j * 2*np.pi * m * f_c * tau) * \
                steering_vec_r * x_nk[m, k]
            P_n = abs(y[1, 1]) / (10 ** (SNR/10))
            y_n[m, k] = y[m, k] + np.sqrt(P_n) * \
                (np.random.randn(Nr) + 1j * np.random.randn(Nr))
            
    print (f'{b} - 0')

    return y_n, data_label


def multi_cell_gen(
    cell_num,
    valid_cell = 0, #initialised to be the first cell
    Nr = 4,
    M = 4,
    K = 10000,
    SNR = 2
):
    y_mc = np.zeros(shape = (M, K, Nr),dtype=np.complex64)
    label = np.zeros(shape = (M, K, cell_num))
    for i in range (cell_num):
        _y, _label = signal_gen(
            Nr = Nr, 
            K = K, 
            M = M, 
            SNR = SNR
        )
        y_mc += _y

        if i == valid_cell:
            label = _label
    return y_mc, label


_y, _label = multi_cell_gen(3)

