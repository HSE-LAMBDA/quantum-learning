import numpy as np
from numpy import array, trace, random, linalg
from numpy.linalg import norm

def projectorOnto(vector):
    """ Returns a rank-1 projector onto a given vector
    """
    return np.tensordot(vector, vector.conj(), 0)

def randomPureState(dim):
    """ Generates Haar-random pure state density matrix
    
    dim - dimension of the resulting matrix
    """
    psi = random.randn(dim) + 1j*random.randn(dim)
    proj = projectorOnto(psi)
    proj /= trace(proj)
    return proj

def randomMixedState(dim):
    """ Generates Hilbert - Schmidt distributed random mixed states
    
    dim - dimension of the resulting matrix
    """
    state = random.randn(dim, dim) + 1j*random.randn(dim, dim)
    state = state.dot(state.T.conj())
    state /= trace(state)
    return state

def isMeasurementValid(measurement, eps = 1e-12):
    """ Checks whether a given measurement is valid (i.e. a normalized rank-1 projector)
    
    measurement - the measurement to check
    eps - internal test error
    """
    return ((abs(trace(measurement) - 1.) <= eps) and
            (measurement == measurement.T.conj()).all() and
            (linalg.matrix_rank(measurement) == 1))
           
def isDensityMatrix(rho, eps = 1e-12):
    """ Checks whether a given matrix rho is a density matrix
    (i.e. Hermitian, positive semidefinite with unit trace)
    
    rho - the density matrix to check
    eps - internal test error
    """
    if ((abs(trace(rho) - 1.) > eps) or
        (rho != rho.T.conj()).all()):
        return False
    return (linalg.eigvalsh(rho) >= -eps).all()
    
def bornRule(trueState, measurement):
    """ Calculates probability according to the Born's rule
    
    trueState - the true state's density matrix
    measurement - a normalized rank-1 projector
    """
    return trace(trueState.dot(measurement)).real

def measure(trials, trueState, measurement, checkValidity = False):
    """ Returns measured counts
    
    trials - the number of repetitions of the measurement
    trueState - the true density matrix to measure
    measurement - a normalized rank-1 projector
    """
    if checkValidity:
        if ((not isMeasurementValid(measurement)) or
            (not isDensityMatrix(trueState))):
            raise ValueError('Invalid true state and/or measurement were given')
    p = bornRule(trueState, measurement)
    n = random.binomial(trials, p)
    return array([n, trials - n])
    
def generate_dataset(target_state, train_size, measurements_cnt=10, shuffle=True):
    dim = target_state.shape[0]
    assert(train_size <= dim, 'overdefined system of projectors')
    projectors = [] # E
    for _ in range(train_size):
        proj = randomPureState(dim) # E_m
        projectors.append(proj)
    train_X = []
    train_y = []
    for proj in projectors:
        measurements = measure(measurements_cnt, target_state, proj)
        for _ in range(measurements[0]):
            train_X.append(proj)
            train_y.append(1)
        for _ in range(measurements[1]):
            train_X.append(proj)
            train_y.append(0)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    indices = np.arange(train_X.shape[0])
    if shuffle: np.random.shuffle(indices)
    return train_X[indices], train_y[indices]
    
# Sample qubit true states
trueStates = array([
    [[1., 0.], [0., 0.]], # H state
    [[0.5, 0.5], [0.5, 0.5]], # D state
    [[0.5, -0.5j], [0.5j, 0.5]], # R state
    [[0.7, 0.], [0., 0.3]], # 30% mixed state
    [[0.39759026+0.j, -0.48358514+0.07521739j], [-0.48358514-0.07521739j, 0.60240974+0.j]], # Random pure state #1
    [[0.37719362+0.j, 0.18480147+0.44807032j], [0.18480147-0.44807032j, 0.62280638+0.j]], # Random pure state #2
    [[0.62829064+0.j, 0.13397942-0.30318748j], [0.13397942+0.30318748j, 0.37170936+0.j]], # Random mixed state #1
    [[0.75525869+0.j, 0.27476800-0.2559911j], [0.27476800+0.2559911j, 0.24474131+0.j]] # Random mixed state #2
    ])
