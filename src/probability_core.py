import math
import random
from typing import List, Tuple, Callable, Optional, Union
import numpy as np
from scipy.stats import norm, gamma, beta, nbinom, chi2
from scipy.special import gammaln
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy import sparse

class ProbabilityCore:
    """An enhanced core class for probability calculations, distributions, and simulations with vectorized operations, JIT compilation, and parallel processing."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility.
        
        Args:
            seed: Optional integer seed for random number generators.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def log_factorial(self, n: np.ndarray) -> np.ndarray:
        """Calculate log factorial of n, vectorized using gammaln.
        
        Args:
            n: Array of non-negative integers.
        Returns:
            Array of log factorials.
        Raises:
            ValueError: If any n is negative.
        """
        n = np.asarray(n, dtype=np.int64)
        if np.any(n < 0):
            raise ValueError("Factorial requires non-negative integers")
        return gammaln(n + 1)
    
    def combination(self, n: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Calculate combination (n choose r) using log for large numbers, vectorized.
        
        Args:
            n: Array of total items.
            r: Array of chosen items.
        Returns:
            Array of combinations.
        Raises:
            ValueError: If r < 0 or r > n, or if shapes are incompatible.
        """
        n = np.asarray(n, dtype=np.int64)
        r = np.asarray(r, dtype=np.int64)
        if n.shape != r.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(n, r)
            except ValueError:
                raise ValueError("n and r must have compatible shapes for broadcasting")
        if np.any(r < 0) or np.any(r > n):
            return np.zeros_like(np.broadcast_to(n, (n.shape if n.ndim > r.ndim else r.shape)), dtype=np.float64)
        r = np.minimum(r, n - r)
        log_result = self.log_factorial(n) - self.log_factorial(r) - self.log_factorial(n - r)
        if np.any(log_result > 700):  # Approximate log(1e308) to avoid overflow
            raise ValueError("Combination result too large, potential overflow")
        return np.exp(log_result)
    
    def permutation(self, n: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Calculate permutation (n permute r) using log for large numbers, vectorized.
        
        Args:
            n: Array of total items.
            r: Array of chosen items.
        Returns:
            Array of permutations.
        Raises:
            ValueError: If r < 0 or r > n, or if shapes are incompatible.
        """
        n = np.asarray(n, dtype=np.int64)
        r = np.asarray(r, dtype=np.int64)
        if n.shape != r.shape:
             # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(n, r)
            except ValueError:
                raise ValueError("n and r must have compatible shapes for broadcasting")
        if np.any(r < 0) or np.any(r > n):
            return np.zeros_like(np.broadcast_to(n, (n.shape if n.ndim > r.ndim else r.shape)), dtype=np.float64)
        log_result = self.log_factorial(n) - self.log_factorial(n - r)
        if np.any(log_result > 700):
            raise ValueError("Permutation result too large, potential overflow")
        return np.exp(log_result)
    
    def probability(self, favorable_outcomes: np.ndarray, total_outcomes: np.ndarray) -> np.ndarray:
        """Calculate basic probability P(A) = favorable outcomes / total outcomes, vectorized.
        
        Args:
            favorable_outcomes: Array of favorable outcomes.
            total_outcomes: Array of total outcomes.
        Returns:
            Array of probabilities.
        Raises:
            ValueError: If total_outcomes <= 0, favorable_outcomes < 0, or shapes are incompatible.
        """
        favorable_outcomes = np.asarray(favorable_outcomes, dtype=np.float64)
        total_outcomes = np.asarray(total_outcomes, dtype=np.float64)
        if favorable_outcomes.shape != total_outcomes.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(favorable_outcomes, total_outcomes)
            except ValueError:
                raise ValueError("favorable_outcomes and total_outcomes must have compatible shapes")
        if np.any(total_outcomes <= 0):
            raise ValueError("Total outcomes must be positive")
        if np.any(favorable_outcomes < 0):
            raise ValueError("Favorable outcomes must be non-negative")
        return favorable_outcomes / total_outcomes
    
    def _binomial_worker(self, n_k_p: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Worker function for parallel binomial probability calculation."""
        n, k, p = n_k_p
        return self.binomial_probability(n, k, p)
    
    def binomial_probability(self, n: np.ndarray, k: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Calculate binomial probability P(X=k) for n trials, k successes, and probability p, vectorized.
        
        Args:
            n: Array of number of trials.
            k: Array of number of successes.
            p: Array of success probabilities.
        Returns:
            Array of binomial probabilities.
        Raises:
            ValueError: If p < 0 or p > 1, k < 0 or k > n, or shapes are incompatible.
        """
        n = np.asarray(n, dtype=np.int64)
        k = np.asarray(k, dtype=np.int64)
        p = np.asarray(p, dtype=np.float64)
        if n.shape != k.shape or n.shape != p.shape:
             # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(n, k, p)
            except ValueError:
                raise ValueError("n, k, and p must have compatible shapes for broadcasting")
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("Probability p must be between 0 and 1")
        
        # Create a boolean mask for valid k values
        valid_k = (k >= 0) & (k <= n)
        # Initialize result array with zeros
        result = np.zeros(np.broadcast(n, k, p).shape, dtype=np.float64)
        
        # Only perform calculation where k is valid
        if np.any(valid_k):
            n_valid, k_valid, p_valid = np.broadcast_arrays(n, k, p)
            n_valid, k_valid, p_valid = n_valid[valid_k], k_valid[valid_k], p_valid[valid_k]

            p_valid = np.where(p_valid == 0, 1e-10, p_valid)
            p_valid = np.where(p_valid == 1, 1 - 1e-10, p_valid)
            
            log_prob = self.log_factorial(n_valid) - self.log_factorial(k_valid) - self.log_factorial(n_valid - k_valid)
            log_prob += k_valid * np.log(p_valid) + (n_valid - k_valid) * np.log(1 - p_valid)
            result[valid_k] = np.exp(log_prob)
            
        return result

    def binomial_probability_parallel(self, n: np.ndarray, k: np.ndarray, p: np.ndarray, 
                                     parallel_threshold: Optional[int] = None) -> np.ndarray:
        """Calculate binomial probabilities in parallel for large arrays.
        
        Args:
            n: Array of number of trials.
            k: Array of number of successes.
            p: Array of success probabilities.
            parallel_threshold: Minimum array size to trigger parallel processing (default: based on CPU count).
        Returns:
            Array of binomial probabilities.
        Raises:
            ValueError: If input arrays have incompatible shapes.
        """
        n = np.asarray(n)
        k = np.asarray(k)
        p = np.asarray(p)
        if n.shape != k.shape or n.shape != p.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(n, k, p)
            except ValueError:
                raise ValueError("n, k, and p must have compatible shapes for broadcasting")
        
        # After broadcasting, ensure all are the same shape for chunking
        n, k, p = np.broadcast_arrays(n, k, p)
        
        parallel_threshold = parallel_threshold or max(1000, 10000 // cpu_count())
        if n.size < parallel_threshold:
            return self.binomial_probability(n, k, p)
        
        n_processes = min(cpu_count(), 8)
        # Use np.array_split for even distribution of work
        n_chunks = np.array_split(n, n_processes)
        k_chunks = np.array_split(k, n_processes)
        p_chunks = np.array_split(p, n_processes)
        chunks = list(zip(n_chunks, k_chunks, p_chunks))
        
        with Pool(n_processes) as pool:
            results = pool.map(self._binomial_worker, chunks)
        return np.concatenate(results)

    def poisson_probability(self, k: np.ndarray, lambda_param: np.ndarray) -> np.ndarray:
        """Calculate Poisson probability P(X=k) for k events and rate lambda, vectorized.
        
        Args:
            k: Array of event counts.
            lambda_param: Array of rate parameters.
        Returns:
            Array of Poisson probabilities.
        Raises:
            ValueError: If k < 0, lambda <= 0, or shapes are incompatible.
        """
        k = np.asarray(k, dtype=np.int64)
        lambda_param = np.asarray(lambda_param, dtype=np.float64)
        if k.shape != lambda_param.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(k, lambda_param)
            except ValueError:
                 raise ValueError("k and lambda_param must have compatible shapes")
        if np.any(k < 0):
            raise ValueError("k must be non-negative integers")
        if np.any(lambda_param <= 0):
            raise ValueError("lambda must be positive")
        
        k, lambda_param = np.broadcast_arrays(k, lambda_param)
        lambda_param = np.where(lambda_param == 0, 1e-10, lambda_param)
        log_prob = k * np.log(lambda_param) - lambda_param - self.log_factorial(k)
        return np.exp(log_prob)
    
    def geometric_probability(self, k: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Calculate Geometric probability P(X=k) for first success on k-th trial, vectorized.
        
        Args:
            k: Array of trial numbers.
            p: Array of success probabilities.
        Returns:
            Array of Geometric probabilities.
        Raises:
            ValueError: If k < 1, p < 0 or p > 1, or shapes are incompatible.
        """
        k = np.asarray(k, dtype=np.int64)
        p = np.asarray(p, dtype=np.float64)
        if k.shape != p.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(k, p)
            except ValueError:
                raise ValueError("k and p must have the same shape or be broadcastable")
        if np.any(k < 1):
            raise ValueError("k must be positive integers")
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("Probability p must be between 0 and 1")
        return np.power(1 - p, k - 1) * p
    
    def negative_binomial_probability(self, r: np.ndarray, k: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Calculate Negative Binomial probability P(X=k) for k failures before r-th success, vectorized.
        
        Args:
            r: Array of number of successes.
            k: Array of number of failures.
            p: Array of success probabilities.
        Returns:
            Array of Negative Binomial probabilities.
        Raises:
            ValueError: If r <= 0, k < 0, p < 0 or p > 1, or shapes are incompatible.
        """
        r = np.asarray(r, dtype=np.int64)
        k = np.asarray(k, dtype=np.int64)
        p = np.asarray(p, dtype=np.float64)
        if r.shape != k.shape or r.shape != p.shape:
             # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(r, k, p)
            except ValueError:
                raise ValueError("r, k, and p must have compatible shapes")
        if np.any(r <= 0) or np.any(k < 0):
            raise ValueError("r must be positive, k must be non-negative")
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("Probability p must be between 0 and 1")
        
        r, k, p = np.broadcast_arrays(r, k, p)
        p = np.where(p == 0, 1e-10, p)
        p = np.where(p == 1, 1 - 1e-10, p)
        log_prob = self.log_factorial(r + k - 1) - self.log_factorial(k) - self.log_factorial(r - 1)
        log_prob += r * np.log(p) + k * np.log(1 - p)
        return np.exp(log_prob)
    
    def chi2_pdf(self, x: np.ndarray, df: np.ndarray) -> np.ndarray:
        """Calculate Chi-Squared probability density function, vectorized.
        
        Args:
            x: Array of values.
            df: Array of degrees of freedom.
        Returns:
            Array of Chi-Squared PDF values.
        Raises:
            ValueError: If x < 0, df <= 0, or shapes are incompatible.
        """
        x = np.asarray(x, dtype=np.float64)
        df = np.asarray(df, dtype=np.int64)
        if x.shape != df.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(x, df)
            except ValueError:
                raise ValueError("x and df must have compatible shapes")
        if np.any(x < 0) or np.any(df <= 0):
            raise ValueError("x must be non-negative, df must be positive")
        return chi2.pdf(x, df=df)
    
    def gamma_pdf(self, x: np.ndarray, shape: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Calculate Gamma probability density function, vectorized.
        
        Args:
            x: Array of values.
            shape: Array of shape parameters (k).
            scale: Array of scale parameters (theta).
        Returns:
            Array of Gamma PDF values.
        Raises:
            ValueError: If x < 0, shape <= 0, scale <= 0, or shapes are incompatible.
        """
        x = np.asarray(x, dtype=np.float64)
        shape = np.asarray(shape, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        if x.shape != shape.shape or x.shape != scale.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(x, shape, scale)
            except ValueError:
                raise ValueError("x, shape, and scale must have compatible shapes")
        if np.any(x < 0) or np.any(shape <= 0) or np.any(scale <= 0):
            raise ValueError("x, shape, and scale must be positive")
        return gamma.pdf(x, a=shape, scale=scale)
    
    def beta_pdf(self, x: np.ndarray, alpha: np.ndarray, beta_param: np.ndarray) -> np.ndarray:
        """Calculate Beta probability density function, vectorized.
        
        Args:
            x: Array of values in [0, 1].
            alpha: Array of alpha parameters.
            beta_param: Array of beta parameters.
        Returns:
            Array of Beta PDF values.
        Raises:
            ValueError: If x < 0, x > 1, alpha <= 0, beta <= 0, or shapes are incompatible.
        """
        x = np.asarray(x, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        beta_param = np.asarray(beta_param, dtype=np.float64)
        if x.shape != alpha.shape or x.shape != beta_param.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(x, alpha, beta_param)
            except ValueError:
                raise ValueError("x, alpha, and beta must have compatible shapes")
        if np.any(x < 0) or np.any(x > 1) or np.any(alpha <= 0) or np.any(beta_param <= 0):
            raise ValueError("x must be in [0, 1], alpha and beta must be positive")
        return beta.pdf(x, a=alpha, b=beta_param)
    
    def normal_pdf(self, x: np.ndarray, mean: np.ndarray = 0, std_dev: np.ndarray = 1) -> np.ndarray:
        """Calculate probability density function for normal distribution, vectorized.
        
        Args:
            x: Array of values.
            mean: Array of means.
            std_dev: Array of standard deviations.
        Returns:
            Array of Normal PDF values.
        Raises:
            ValueError: If std_dev <= 0 or shapes are incompatible.
        """
        x = np.asarray(x, dtype=np.float64)
        mean = np.asarray(mean, dtype=np.float64)
        std_dev = np.asarray(std_dev, dtype=np.float64)
        if x.shape != mean.shape or x.shape != std_dev.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(x, mean, std_dev)
            except ValueError:
                raise ValueError("x, mean, and std_dev must have compatible shapes")
        if np.any(std_dev <= 0):
            raise ValueError("Standard deviation must be positive")
        return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.square((x - mean) / std_dev))
    
    def normal_cdf(self, x: np.ndarray, mean: np.ndarray = 0, std_dev: np.ndarray = 1) -> np.ndarray:
        """Calculate cumulative distribution function for normal distribution, vectorized.
        
        Args:
            x: Array of values.
            mean: Array of means.
            std_dev: Array of standard deviations.
        Returns:
            Array of Normal CDF values.
        Raises:
            ValueError: If std_dev <= 0 or shapes are incompatible.
        """
        x = np.asarray(x, dtype=np.float64)
        mean = np.asarray(mean, dtype=np.float64)
        std_dev = np.asarray(std_dev, dtype=np.float64)
        if x.shape != mean.shape or x.shape != std_dev.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(x, mean, std_dev)
            except ValueError:
                raise ValueError("x, mean, and std_dev must have compatible shapes")
        if np.any(std_dev <= 0):
            raise ValueError("Standard deviation must be positive")
        return norm.cdf(x, loc=mean, scale=std_dev)
    
    def exponential_pdf(self, x: np.ndarray, lambda_param: np.ndarray) -> np.ndarray:
        """Calculate probability density function for exponential distribution, vectorized.
        
        Args:
            x: Array of values.
            lambda_param: Array of rate parameters.
        Returns:
            Array of Exponential PDF values.
        Raises:
            ValueError: If x < 0, lambda <= 0, or shapes are incompatible.
        """
        x = np.asarray(x, dtype=np.float64)
        lambda_param = np.asarray(lambda_param, dtype=np.float64)
        if x.shape != lambda_param.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(x, lambda_param)
            except ValueError:
                raise ValueError("x and lambda_param must have compatible shapes")
        if np.any(x < 0) or np.any(lambda_param <= 0):
            raise ValueError("x must be non-negative and lambda must be positive")
        return lambda_param * np.exp(-lambda_param * x)

    def exponential_cdf(self, x: np.ndarray, lambda_param: np.ndarray) -> np.ndarray:
        """Calculate cumulative distribution function for exponential distribution, vectorized.
        
        Args:
            x: Array of values.
            lambda_param: Array of rate parameters.
        Returns:
            Array of Exponential CDF values.
        Raises:
            ValueError: If x < 0, lambda <= 0, or shapes are incompatible.
        """
        x = np.asarray(x, dtype=np.float64)
        lambda_param = np.asarray(lambda_param, dtype=np.float64)
        if x.shape != lambda_param.shape:
            # --- MODIFIED: Allow broadcasting ---
            try:
                np.broadcast(x, lambda_param)
            except ValueError:
                raise ValueError("x and lambda_param must have compatible shapes")
        if np.any(x < 0) or np.any(lambda_param <= 0):
            raise ValueError("x must be non-negative and lambda must be positive")
        return 1 - np.exp(-lambda_param * x)
    
    def uniform_random(self, a: float, b: float, size: int = 1) -> np.ndarray:
        """Generate random number(s) from uniform distribution in range [a, b].
        
        Args:
            a: Lower bound.
            b: Upper bound.
            size: Number of samples.
        Returns:
            Array of random samples.
        Raises:
            ValueError: If a >= b.
        """
        if a >= b:
            raise ValueError("a must be less than b")
        return np.random.uniform(a, b, size)
    
    def bernoulli_trial(self, p: float, size: int = 1) -> np.ndarray:
        """Perform Bernoulli trial(s) with success probability p.
        
        Args:
            p: Success probability.
            size: Number of trials.
        Returns:
            Array of trial outcomes (0 or 1).
        Raises:
            ValueError: If p < 0 or p > 1.
        """
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1")
        return np.random.binomial(1, p, size)
    
    def normal_random(self, mean: float = 0, std_dev: float = 1, size: int = 1) -> np.ndarray:
        """Generate random number(s) from normal distribution.
        
        Args:
            mean: Mean of the distribution.
            std_dev: Standard deviation.
            size: Number of samples.
        Returns:
            Array of random samples.
        Raises:
            ValueError: If std_dev <= 0.
        """
        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive")
        return np.random.normal(loc=mean, scale=std_dev, size=size)
    
    def binomial_random(self, n: int, p: float, size: int = 1) -> np.ndarray:
        """Generate random number(s) from binomial distribution.
        
        Args:
            n: Number of trials.
            p: Success probability.
            size: Number of samples.
        Returns:
            Array of random samples.
        Raises:
            ValueError: If n < 0 or p < 0 or p > 1.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1")
        return np.random.binomial(n, p, size)
    
    def gamma_random(self, shape: float, scale: float, size: int = 1) -> np.ndarray:
        """Generate random number(s) from Gamma distribution.
        
        Args:
            shape: Shape parameter (k).
            scale: Scale parameter (theta).
            size: Number of samples.
        Returns:
            Array of random samples.
        Raises:
            ValueError: If shape <= 0 or scale <= 0.
        """
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be positive")
        return np.random.gamma(shape, scale, size)
    
    def beta_random(self, alpha: float, beta_param: float, size: int = 1) -> np.ndarray:
        """Generate random number(s) from Beta distribution.
        
        Args:
            alpha: Alpha parameter.
            beta_param: Beta parameter.
            size: Number of samples.
        Returns:
            Array of random samples.
        Raises:
            ValueError: If alpha <= 0 or beta <= 0.
        """
        if alpha <= 0 or beta_param <= 0:
            raise ValueError("alpha and beta must be positive")
        return np.random.beta(alpha, beta_param, size)
    
    # --- MODIFIED: Worker now only needs a seed. ---
    def _monte_carlo_worker(self, args: Tuple[int, Callable[[], float], int]) -> np.ndarray:
        """Worker function for parallel Monte Carlo simulation."""
        trials, event_func, seed = args
        np.random.seed(seed)  # Seed this process's instance of numpy.random
        return np.array([event_func() for _ in range(trials)])
    
    # --- MODIFIED: Uses SeedSequence for robust parallel seeding. ---
    def monte_carlo_simulation(self, trials: int, event_func: Callable[[], float], parallel: bool = False, 
                               parallel_threshold: Optional[int] = None) -> float:
        """Perform Monte Carlo simulation to estimate expected value, with optional parallel processing.
        
        Args:
            trials: Number of trials.
            event_func: Function returning a numerical outcome. MUST be self-contained for parallel execution.
            parallel: Whether to use parallel processing.
            parallel_threshold: Minimum trials to trigger parallel processing (default: based on CPU count).
        Returns:
            Expected value of the event.
        Raises:
            ValueError: If trials <= 0.
        """
        if not isinstance(trials, int) or trials <= 0:
            raise ValueError("Number of trials must be a positive integer")
        
        parallel_threshold = parallel_threshold or max(1000, 10000 // cpu_count())
        if not parallel or trials < parallel_threshold:
            outcomes = np.array([event_func() for _ in range(trials)])
            return np.mean(outcomes)
        
        n_processes = min(cpu_count(), 8)
        trials_per_process = trials // n_processes
        
        # Use SeedSequence to create statistically independent child seeds
        sg = np.random.SeedSequence()
        child_seeds = sg.spawn(n_processes)
        
        tasks = []
        for i in range(n_processes):
            # Distribute remainder trials to the first few processes
            num_trials = trials_per_process + 1 if i < trials % n_processes else trials_per_process
            if num_trials > 0:
                tasks.append((num_trials, event_func, child_seeds[i]))

        with Pool(processes=n_processes) as pool:
            results = pool.map(self._monte_carlo_worker, tasks)
        
        outcomes = np.concatenate(results)
        return np.mean(outcomes)
    
    def markov_chain(self, transition_matrix: List[List[float]], initial_state: int, steps: int, 
                     track_evolution: bool = False, sparse_matrix: Optional[bool] = None) -> Union[List[float], List[List[float]]]:
        """Simulate the probability distribution of a Markov chain over time.
        
        Args:
            transition_matrix: Square matrix of transition probabilities.
            initial_state: Starting state index.
            steps: Number of steps to simulate.
            track_evolution: Whether to return state probabilities at each step.
            sparse_matrix: Whether to use sparse matrix (default: auto-detect based on density).
        Returns:
            Final state probabilities or list of state probabilities if track_evolution is True.
        Raises:
            ValueError: If matrix is not square, rows don’t sum to 1, or initial_state is invalid.
        """
        n_states = len(transition_matrix)
        if not all(isinstance(row, list) and len(row) == n_states for row in transition_matrix):
            raise ValueError("Transition matrix must be square")
        trans_matrix = np.array(transition_matrix, dtype=np.float64)
        if not np.all(np.abs(np.sum(trans_matrix, axis=1) - 1.0) < 1e-10):
            raise ValueError("Each row in transition matrix must sum to approximately 1")
        if not (0 <= initial_state < n_states):
            raise ValueError("Invalid initial state")
        
        if sparse_matrix is None:
            density = np.count_nonzero(trans_matrix) / trans_matrix.size
            sparse_matrix = density < 0.1
        
        if sparse_matrix:
            trans_matrix_pow = sparse.csr_matrix(trans_matrix)
        else:
            trans_matrix_pow = trans_matrix
            
        state_probs = np.zeros(n_states, dtype=np.float64)
        state_probs[initial_state] = 1.0
        evolution = [state_probs.tolist()] if track_evolution else None
        
        # More efficient calculation using matrix power for the final state
        if not track_evolution and steps > 1:
            final_trans_matrix = trans_matrix_pow
            for _ in range(steps - 1):
                final_trans_matrix = final_trans_matrix.dot(trans_matrix_pow)
            state_probs = state_probs.dot(final_trans_matrix)
        else:
             for _ in range(steps):
                state_probs = state_probs.dot(trans_matrix_pow)
                if track_evolution:
                    # For sparse, convert back to dense for the list
                    evolution.append(state_probs.toarray().flatten().tolist() if sparse_matrix else state_probs.tolist())
        
        return evolution if track_evolution else (state_probs.toarray().flatten().tolist() if sparse_matrix else state_probs.tolist())
        
    # --- NEW METHOD ---
    def simulate_markov_path(self, transition_matrix: List[List[float]], initial_state: int, steps: int) -> List[int]:
        """Simulate a single random path (trajectory) through a Markov chain.
        
        Args:
            transition_matrix: Square matrix of transition probabilities.
            initial_state: Starting state index.
            steps: Number of steps to simulate.
        Returns:
            A list of integers representing the sequence of states visited.
        Raises:
            ValueError: If matrix is not square, rows don’t sum to 1, or initial_state is invalid.
        """
        n_states = len(transition_matrix)
        if not all(len(row) == n_states for row in transition_matrix):
            raise ValueError("Transition matrix must be square")
        trans_matrix = np.array(transition_matrix, dtype=np.float64)
        if not np.all(np.abs(np.sum(trans_matrix, axis=1) - 1.0) < 1e-10):
            raise ValueError("Each row in transition matrix must sum to approximately 1")
        if not (0 <= initial_state < n_states):
            raise ValueError("Invalid initial state")

        path = [initial_state]
        current_state = initial_state
        possible_states = np.arange(n_states)
        
        for _ in range(steps):
            probabilities = trans_matrix[current_state]
            next_state = np.random.choice(possible_states, p=probabilities)
            path.append(next_state)
            current_state = next_state
            
        return path

    def mean(self, data: np.ndarray) -> float:
        """Calculate mean of a dataset using NumPy.
        
        Args:
            data: Array of values.
        Returns:
            Mean of the dataset.
        Raises:
            ValueError: If dataset is empty.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.size == 0:
            raise ValueError("Dataset cannot be empty")
        return float(np.mean(data))
    
    def variance(self, data: np.ndarray) -> float:
        """Calculate variance of a dataset using NumPy.
        
        Args:
            data: Array of values.
        Returns:
            Variance of the dataset.
        Raises:
            ValueError: If dataset is empty.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.size == 0:
            raise ValueError("Dataset cannot be empty")
        return float(np.var(data))
    
    def standard_deviation(self, data: np.ndarray) -> float:
        """Calculate standard deviation of a dataset using NumPy.
        
        Args:
            data: Array of values.
        Returns:
            Standard deviation of the dataset.
        Raises:
            ValueError: If dataset is empty.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.size == 0:
            raise ValueError("Dataset cannot be empty")
        return float(np.std(data))
    
    # --- MODIFIED: Added plot_type argument and more plotting options. ---
    def plot_distribution(self, dist_type: str, params: dict, x_range: Tuple[float, float], 
                         points: int = 100, ax: Optional[plt.Axes] = None, 
                         plot_type: str = 'pdf', save_path: Optional[str] = None, **plot_kwargs):
        """Plot probability density/mass or cumulative distribution function.
        
        Args:
            dist_type: "normal", "exponential", "geometric", "poisson", "gamma", "beta", "negative_binomial", "chi2".
            params: Dictionary of distribution parameters.
            x_range: Tuple of (min, max) for x-axis.
            points: Number of points to plot for continuous distributions.
            ax: Optional matplotlib Axes to plot on.
            plot_type: 'pdf' for PDF/PMF or 'cdf' for CDF.
            save_path: Optional file path to save the plot.
            **plot_kwargs: Additional plotting options (e.g., color, linestyle).
        Returns:
            Matplotlib Axes object.
        Raises:
            ValueError: If dist_type or plot_type is unsupported or parameters are invalid.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots()

        dist_type = dist_type.lower()
        plot_type = plot_type.lower()
        label_suffix = "PMF" if dist_type in ["geometric", "poisson", "negative_binomial"] else "PDF"
        
        if plot_type == 'cdf':
            label_suffix = 'CDF'

        x = np.linspace(x_range[0], x_range[1], points)
        
        # --- Discrete Distributions ---
        if dist_type in ["geometric", "poisson", "negative_binomial"]:
            x = np.arange(max(0, int(x_range[0])), int(x_range[1]) + 1)
            if dist_type == "geometric":
                p = params.get("p", 0.5)
                y = self.geometric_probability(x, p) if plot_type != 'cdf' else nbinom.cdf(x, 1, p)
            elif dist_type == "poisson":
                lambda_val = params.get("lambda", 1)
                y = self.poisson_probability(x, lambda_val) if plot_type != 'cdf' else poisson.cdf(x, lambda_val)
            elif dist_type == "negative_binomial":
                r, p = params.get("r", 1), params.get("p", 0.5)
                y = self.negative_binomial_probability(r, x, p) if plot_type != 'cdf' else nbinom.cdf(x, r, p)
            ax.stem(x, y, label=f"{dist_type.capitalize()} {label_suffix}", **plot_kwargs)

        # --- Continuous Distributions ---
        elif dist_type in ["normal", "exponential", "gamma", "beta", "chi2"]:
            if dist_type == "normal":
                mean, std = params.get("mean", 0), params.get("std_dev", 1)
                y = self.normal_pdf(x, mean, std) if plot_type == 'pdf' else self.normal_cdf(x, mean, std)
            elif dist_type == "exponential":
                lambda_val = params.get("lambda", 1)
                y = self.exponential_pdf(x, lambda_val) if plot_type == 'pdf' else self.exponential_cdf(x, lambda_val)
            elif dist_type == "gamma":
                shape, scale = params.get("shape", 1), params.get("scale", 1)
                y = self.gamma_pdf(x, shape, scale) if plot_type == 'pdf' else gamma.cdf(x, a=shape, scale=scale)
            elif dist_type == "beta":
                alpha, beta_val = params.get("alpha", 1), params.get("beta", 1)
                y = self.beta_pdf(x, alpha, beta_val) if plot_type == 'pdf' else beta.cdf(x, a=alpha, b=beta_val)
            elif dist_type == "chi2":
                df = params.get("df", 1)
                y = self.chi2_pdf(x, df) if plot_type == 'pdf' else chi2.cdf(x, df)
            ax.plot(x, y, label=f"{dist_type.capitalize()} {label_suffix}", **plot_kwargs)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

        ax.set_xlabel("x")
        ax.set_ylabel("Probability Density" if plot_type == 'pdf' else 'Cumulative Probability')
        ax.set_title(f"{dist_type.capitalize()} Distribution")
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if standalone:
            plt.show()
            
        return ax

# --- MODIFIED: Updated example usage to reflect changes ---
if __name__ == "__main__":
    prob_core = ProbabilityCore(seed=42)
    
    # Example 1: Parallel binomial probability with broadcasting
    n = np.array([20, 30])
    k = np.arange(0, 20).reshape(10, 2)
    p = 0.5
    # n (shape 2,) and k (shape 10,2) will broadcast to (10,2)
    binomial_probs = prob_core.binomial_probability_parallel(n, k, p)
    print(f"Parallel binomial probabilities (shape: {binomial_probs.shape})")

    # Example 2: Self-contained event function for parallel Monte Carlo
    # This function must not rely on the `prob_core` instance
    def dice_sum_is_seven():
        # Use numpy's random module directly, which will be seeded by the worker
        die1 = np.random.randint(1, 7)
        die2 = np.random.randint(1, 7)
        return 1.0 if die1 + die2 == 7 else 0.0
    
    monte_carlo_prob = prob_core.monte_carlo_simulation(100000, dice_sum_is_seven, parallel=True)
    print(f"\nParallel Monte Carlo (w/ SeedSequence) prob of dice sum = 7: {monte_carlo_prob:.4f}")
    
    # Example 3: Markov chain path simulation (NEW)
    transition_matrix = [
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ]
    markov_path = prob_core.simulate_markov_path(transition_matrix, initial_state=0, steps=15)
    print(f"\nSimulated Markov path: {markov_path}")
    
    # Example 4: Plotting PDF and CDF on the same axes (MODIFIED)
    fig, ax = plt.subplots(figsize=(10, 6))
    prob_core.plot_distribution("normal", {"mean": 0, "std_dev": 1}, (-4, 4), ax=ax, 
                                plot_type='pdf', color="blue", linestyle="-", label="Normal PDF")
    prob_core.plot_distribution("normal", {"mean": 0, "std_dev": 1}, (-4, 4), ax=ax, 
                                plot_type='cdf', color="cyan", linestyle="--", label="Normal CDF")
    ax.legend() # Re-call legend to show custom labels
    plt.savefig("normal_pdf_cdf.png")
    print("\nSaved plot 'normal_pdf_cdf.png'")
    plt.show()