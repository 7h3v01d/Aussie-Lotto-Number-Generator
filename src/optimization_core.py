import pulp
from constraint import Problem
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Protocol
from abc import ABC, abstractmethod
import random
import uuid
import json
from joblib import Parallel, delayed
import platform
import time
from copy import deepcopy
import logging
from threading import Thread, Event
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Thread-based timeout decorator
def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            event = Event()
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
                finally:
                    event.set()
            thread = Thread(target=target)
            thread.daemon = True  # Ensure thread terminates with main program
            thread.start()
            if not event.wait(seconds):
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

# Interfaces for external cores
class LogicCoreInterface(ABC):
    @abstractmethod
    def get_rules(self) -> List[Any]:
        """Return a list of rules/constraints for optimization problems."""
        pass

class ProbabilityCoreInterface(ABC):
    @abstractmethod
    def get_probabilities(self) -> Dict[str, float]:
        """Return a dictionary of state probabilities."""
        pass

    @abstractmethod
    def transition_model(self, state: Any, action: Callable) -> Any:
        """Return the next state given the current state and action."""
        pass

class OptimizationCore:
    solvers = {}

    @classmethod
    def register_solver(cls, problem_type: str, solver: Callable, solver_params: Optional[Dict] = None, required_params: List[str] = None):
        """
        Register a new solver for a problem type.

        Args:
            problem_type: Name of the problem type.
            solver: Callable solver function.
            solver_params: Optional dictionary of solver-specific parameters.
            required_params: List of required parameter keys for the solver.

        Raises:
            ValueError: If solver is not callable, problem_type is invalid, or required_params are invalid.
        """
        if not callable(solver):
            raise ValueError(f"Solver for {problem_type} must be callable")
        if not problem_type:
            raise ValueError("Problem type cannot be empty")
        if required_params and not all(isinstance(p, str) for p in required_params):
            raise ValueError("Required parameters must be strings")
        cls.solvers[problem_type] = {
            'solver': solver,
            'params': solver_params or {},
            'required_params': required_params or [],
            'has_cache': hasattr(solver, 'cache_clear')
        }
        logger.debug(f"Registered solver for {problem_type} with required params: {required_params}, cacheable: {cls.solvers[problem_type]['has_cache']}")

    def __init__(self, logic_core: Optional[LogicCoreInterface] = None, 
                 probability_core: Optional[ProbabilityCoreInterface] = None, 
                 seed: Optional[int] = None):
        """
        Initialize OptimizationCore with optional LogicCore and ProbabilityCore.

        Args:
            logic_core: Instance of LogicCoreInterface for rule-based constraints.
            probability_core: Instance of ProbabilityCoreInterface for probabilistic modeling.
            seed: Optional seed for reproducible random number generation.
        """
        self.logic_core = logic_core
        self.probability_core = probability_core
        self.rng = np.random.default_rng(seed)
        self.validate_logic_core()
        logger.debug("OptimizationCore initialized with seed: %s", seed)

    def validate_logic_core(self):
        """
        Validate LogicCore rules for compatibility with solvers.

        Raises:
            ValueError: If LogicCore rules are invalid or incompatible.
        """
        if self.logic_core:
            rules = self.logic_core.get_rules()
            for rule in rules:
                if not isinstance(rule, (tuple, Callable)):
                    logger.error("Invalid LogicCore rule format: %s", rule)
                    raise ValueError(f"Invalid LogicCore rule format: {rule}")
                if isinstance(rule, tuple) and len(rule) == 3:
                    coeff_dict, op, bound = rule
                    if not isinstance(coeff_dict, dict) or op not in {'<=', '>=', '='} or not isinstance(bound, (int, float)):
                        logger.error("Invalid linear programming rule: %s", rule)
                        raise ValueError(f"Invalid linear programming rule: {rule}")
            logger.debug("LogicCore validated successfully: %d rules", len(rules))

    def clear_cache(self, problem_type: Optional[str] = None):
        """
        Clear LRU cache for specified or all cached methods to manage memory.

        Args:
            problem_type: Optional problem type to clear specific cache ('linear', 'csp', 'bayesian', or custom).

        Raises:
            ValueError: If problem_type is specified but invalid.
        """
        cleared = False
        if problem_type is None:
            self.linear_programming.cache_clear()
            self.constraint_satisfaction.cache_clear()
            self.bayesian_decision.cache_clear()
            for solver_type, solver_info in self.solvers.items():
                if solver_info.get('has_cache', False):
                    solver_info['solver'].cache_clear()
                    logger.info(f"Cleared LRU cache for custom solver: {solver_type}")
            cleared = True
        else:
            if problem_type == 'linear':
                self.linear_programming.cache_clear()
                cleared = True
            elif problem_type == 'csp':
                self.constraint_satisfaction.cache_clear()
                cleared = True
            elif problem_type == 'bayesian':
                self.bayesian_decision.cache_clear()
                cleared = True
            elif problem_type in self.solvers and self.solvers[problem_type].get('has_cache', False):
                self.solvers[problem_type]['solver'].cache_clear()
                cleared = True
            else:
                logger.warning("No cacheable solver found for problem type: %s", problem_type)
                raise ValueError(f"No cacheable solver found for problem type: {problem_type}")
        if cleared:
            logger.info("Cleared LRU cache for %s", problem_type if problem_type else "all solvers")
        else:
            logger.warning("No caches cleared; no matching problem type found")

    @lru_cache(maxsize=128)
    @timeout(60)
    def linear_programming(self, objective: Tuple[Tuple[str, float], ...], 
                         constraints: Tuple[Tuple[Dict[str, float], str, float], ...], 
                         integer: bool = False, sense: str = 'maximize', 
                         var_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None) -> Dict[str, float]:
        """
        Solve a linear or integer programming problem.

        Args:
            objective: Coefficients of variables in objective function as tuple of (var, coeff) pairs.
            constraints: Tuple of (coefficients, operator, bound) where operator is '<=', '>=', or '='.
            integer: If True, use integer programming.
            sense: Optimization direction ('maximize' or 'minimize').
            var_bounds: Dictionary of variable names to (lowBound, upBound) tuples.

        Returns:
            Optimal variable assignments.

        Raises:
            ValueError: If inputs are invalid, variables are missing, or no optimal solution is found.
        """
        start_time = time.time()
        objective_dict = dict(objective)
        if not objective_dict:
            logger.error("Objective function cannot be empty")
            raise ValueError("Objective function cannot be empty")
        if sense not in ['maximize', 'minimize']:
            logger.error("Invalid sense: %s", sense)
            raise ValueError("Sense must be 'maximize' or 'minimize'")
        valid_operators = {'<=', '>=', '='}
        for coeff_dict, op, bound in constraints:
            if not coeff_dict:
                logger.error("Constraint coefficients cannot be empty")
                raise ValueError("Constraint coefficients cannot be empty")
            if op not in valid_operators:
                logger.error("Invalid operator %s, must be one of %s", op, valid_operators)
                raise ValueError(f"Invalid operator {op}, must be one of {valid_operators}")
            for var in coeff_dict:
                if var not in objective_dict:
                    logger.error("Variable %s in constraint not found in objective", var)
                    raise ValueError(f"Variable {var} in constraint not found in objective")

        prob = pulp.LpProblem("Linear_Programming", pulp.LpMaximize if sense == 'maximize' else pulp.LpMinimize)
        var_type = pulp.LpInteger if integer else pulp.LpContinuous
        var_bounds = var_bounds or {}
        variables = {
            var: pulp.LpVariable(var, lowBound=var_bounds.get(var, (0, None))[0], 
                               upBound=var_bounds.get(var, (0, None))[1], cat=var_type)
            for var in objective_dict
        }

        prob += pulp.lpSum([coeff * variables[var] for var, coeff in objective])
        for coeff_dict, op, bound in constraints:
            expr = pulp.lpSum([coeff * variables[var] for var, coeff in coeff_dict.items()])
            if op == '<=':
                prob += expr <= bound
            elif op == '>=':
                prob += expr >= bound
            elif op == '=':
                prob += expr == bound

        prob.solve()
        if pulp.LpStatus[prob.status] != 'Optimal':
            logger.error("No optimal solution found: %s", pulp.LpStatus[prob.status])
            raise ValueError(f"No optimal solution found: {pulp.LpStatus[prob.status]}. "
                           f"Possible cause: {'Infeasible constraints' if pulp.LpStatus[prob.status] == 'Infeasible' else 'Unbounded problem'}")
        result = {var: variables[var].varValue for var in variables}
        logger.debug("Linear programming completed in %.2f seconds: %s", time.time() - start_time, result)
        return result

    @lru_cache(maxsize=128)
    @timeout(60)
    def constraint_satisfaction(self, variables: Tuple[Tuple[str, Tuple[Any, ...]], ...], 
                              constraints: Tuple[Callable[[Dict[str, Any]], bool], ...]) -> Tuple[Dict[str, Any], ...]:
        """
        Solve a constraint satisfaction problem.

        Args:
            variables: Tuple of (variable name, domain) pairs, where domain is a tuple of possible values.
            constraints: Tuple of constraint functions that take a variable assignment and return True/False.

        Returns:
            Tuple of satisfying assignments.

        Raises:
            ValueError: If inputs are invalid, constraints are not callable, or no solutions exist.
        """
        start_time = time.time()
        variables_dict = dict(variables)
        if not variables_dict:
            logger.error("Variables dictionary cannot be empty")
            raise ValueError("Variables dictionary cannot be empty")
        for var, domain in variables_dict.items():
            if not domain:
                logger.error("Domain for variable %s cannot be empty", var)
                raise ValueError(f"Domain for variable {var} cannot be empty")
        for constraint in constraints:
            if not callable(constraint):
                logger.error("All constraints must be callable")
                raise ValueError("All constraints must be callable")

        problem = Problem()
        for var, domain in variables_dict.items():
            problem.addVariable(var, list(domain))
            problem.setSolver(Problem.MinConflictsSolver())  # Use constraint propagation for efficiency

        for constraint in constraints:
            problem.addConstraint(constraint, list(variables_dict.keys()))

        solutions = problem.getSolutions()
        if not solutions:
            logger.warning("No solutions found for the constraint satisfaction problem")
            raise ValueError("No solutions found for the constraint satisfaction problem. "
                           "Consider relaxing constraints or expanding variable domains.")
        logger.debug("CSP completed in %.2f seconds, found %d solutions", time.time() - start_time, len(solutions))
        return tuple(solutions)

    @lru_cache(maxsize=128)
    @timeout(30)
    def bayesian_decision(self, actions: Tuple[str, ...], states: Tuple[str, ...], 
                         utilities: Tuple[Tuple[Tuple[str, str], float], ...], 
                         probabilities: Dict[str, float], use_cache: bool = True) -> Tuple[str, Dict[str, float]]:
        """
        Bayesian decision theory for expected utility maximization.

        Args:
            actions: Tuple of possible actions.
            states: Tuple of possible states.
            utilities: Tuple of ((action, state), utility) pairs.
            probabilities: Dictionary of state to probability.
            use_cache: Whether to use LRU cache for this computation.

        Returns:
            Tuple of (best action, expected utilities dictionary).

        Raises:
            ValueError: If inputs are invalid or probabilities don't sum to ~1.
        """
        start_time = time.time()
        if not use_cache:
            self.bayesian_decision.cache_clear()

        if not actions or not states:
            logger.error("Actions and states cannot be empty")
            raise ValueError("Actions and states cannot be empty")
        if not probabilities:
            logger.error("Probabilities cannot be empty")
            raise ValueError("Probabilities cannot be empty")
        prob_sum = sum(probabilities.values())
        if not 0.99 <= prob_sum <= 1.01:
            logger.error("Probabilities must sum to approximately 1, got %s", prob_sum)
            raise ValueError(f"Probabilities must sum to approximately 1, got {prob_sum}")
        utilities_dict = dict(utilities)
        for action in actions:
            for state in states:
                if (action, state) not in utilities_dict:
                    logger.error("Utility for (%s, %s) not defined", action, state)
                    raise ValueError(f"Utility for ({action}, {state}) not defined")

        expected_utilities = {}
        for action in actions:
            eu = sum(probabilities[state] * utilities_dict[(action, state)] for state in states)
            expected_utilities[action] = eu

        best_action = max(expected_utilities, key=expected_utilities.get)
        logger.debug("Bayesian decision completed in %.2f seconds: %s", time.time() - start_time, best_action)
        return best_action, expected_utilities

    def visualize_results(self, data: Dict[str, float], title: str, x_label: str, y_label: str, 
                         chart_type: str = 'bar', custom_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a Chart.js-compatible JSON configuration for visualizing results.

        Args:
            data: Dictionary of labels to values (e.g., actions to utilities or variables to values).
            title: Title of the chart.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            chart_type: Type of chart ('bar', 'line', 'pie').
            custom_config: Optional dictionary to merge with default Chart.js options.

        Returns:
            Dictionary containing Chart.js configuration and raw data.

        Raises:
            ValueError: If chart_type is invalid or data is empty.
        """
        start_time = time.time()
        valid_chart_types = {'bar', 'line', 'pie'}
        if chart_type not in valid_chart_types:
            logger.error("Invalid chart type: %s", chart_type)
            raise ValueError(f"Chart type must be one of {valid_chart_types}")
        if not data:
            logger.error("Visualization data cannot be empty")
            raise ValueError("Visualization data cannot be empty")

        def generate_colors(n):
            base_colors = ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF"]
            return [base_colors[i % len(base_colors)] for i in range(n)]

        default_config = {
            "type": chart_type,
            "data": {
                "labels": list(data.keys()),
                "datasets": [{
                    "label": y_label,
                    "data": list(data.values()),
                    "backgroundColor": ["#36A2EB"] if chart_type != 'pie' else generate_colors(len(data)),
                    "borderColor": ["#1F77B4"] if chart_type != 'pie' else generate_colors(len(data)),
                    "borderWidth": 1 if chart_type != 'pie' else 0,
                    "fill": chart_type == 'line'
                }]
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": x_label}},
                    "y": {"title": {"display": True, "text": y_label}, "beginAtZero": True}
                } if chart_type != 'pie' else {},
                "plugins": {"title": {"display": True, "text": title}},
                "legend": {"position": "top"} if chart_type == 'pie' else {}
            }
        }

        if custom_config:
            def deep_merge(default, custom):
                result = deepcopy(default)
                for key, value in custom.items():
                    if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = deepcopy(value)
                return result
            try:
                default_config = deep_merge(default_config, custom_config)
            except Exception as e:
                logger.warning("Invalid custom_config, using default: %s", str(e))
                default_config = default_config

        result = {"chart_config": default_config, "raw_data": data}
        logger.debug("Visualization generated in %.2f seconds for %s chart", time.time() - start_time, chart_type)
        return result

    def _simulate_action(self, action: Callable, initial_state: Any, 
                       reward_func: Callable[[Any], float], 
                       transition_model: Callable[[Any, Callable], Any], 
                       horizon: int, gamma: float = 1.0, timeout: float = 30.0) -> Optional[float]:
        """
        Helper function for Monte Carlo simulation of a single action.

        Args:
            action: Action function to simulate.
            initial_state: Initial state for simulation.
            reward_func: Function to compute reward from state.
            transition_model: Function to compute next state.
            horizon: Number of simulation steps.
            gamma: Discount factor for future rewards (0 to 1).
            timeout: Maximum time (seconds) per simulation.

        Returns:
            Total discounted reward for the simulation, or None if simulation fails.

        Raises:
            TimeoutError: If simulation exceeds the time limit.
        """
        @timeout(timeout)
        def run_simulation():
            total_reward = 0
            current_state = deepcopy(initial_state)
            for t in range(horizon):
                try:
                    next_state = transition_model(current_state, action)
                    total_reward += (gamma ** t) * reward_func(next_state)
                    current_state = next_state
                except Exception as e:
                    logger.warning("Simulation failed for action %s: %s", id(action), str(e))
                    return None
            return total_reward
        return run_simulation()

    def monte_carlo_planning(self, initial_state: Any, actions: List[Callable[[Any], Any]], 
                            action_names: List[str], reward_func: Callable[[Any], float], 
                            transition_model: Callable[[Any, Callable], Any], 
                            num_simulations: int = 1000, horizon: int = 10, 
                            n_jobs: int = -1, gamma: float = 1.0, timeout: float = 30.0,
                            progress_callback: Optional[Callable[[float], None]] = None,
                            batch_size: int = 100) -> Tuple[Callable, float, Dict[str, float]]:
        """
        Reinforcement planning using Monte Carlo simulations with parallel processing and batching.

        Args:
            initial_state: Initial state of the system.
            actions: List of action functions that take a state and return an action value.
            action_names: List of descriptive names for actions.
            reward_func: Function that takes a state and returns a reward.
            transition_model: Function that takes (state, action) and returns next state.
            num_simulations: Number of Monte Carlo simulations.
            horizon: Planning horizon (number of steps).
            n_jobs: Number of parallel jobs (-1 for all available cores).
            gamma: Discount factor for future rewards (0 to 1).
            timeout: Maximum time (seconds) per simulation.
            progress_callback: Optional callback to report simulation progress (0 to 1).
            batch_size: Number of simulations per batch for memory efficiency.

        Returns:
            Tuple of (best action, average reward, dictionary of action names to average rewards).

        Raises:
            ValueError: If inputs are invalid or no valid simulations complete.
        """
        start_time = time.time()
        if not actions or not callable(reward_func) or not callable(transition_model):
            logger.error("Invalid inputs: actions, reward_func, and transition_model must be valid")
            raise ValueError("Actions, reward_func, and transition_model must be valid")
        if len(actions) != len(action_names):
            logger.error("Number of action names (%d) does not match number of actions (%d)", len(action_names), len(actions))
            raise ValueError(f"Number of action names ({len(action_names)}) does not match number of actions ({len(actions)})")
        if num_simulations <= 0 or horizon <= 0:
            logger.error("num_simulations and horizon must be positive")
            raise ValueError("num_simulations and horizon must be positive")
        if n_jobs == 0 or n_jobs < -1:
            logger.error("n_jobs must be -1 or positive")
            raise ValueError("n_jobs must be -1 or positive")
        if not 0 <= gamma <= 1:
            logger.error("gamma must be between 0 and 1")
            raise ValueError("gamma must be between 0 and 1")
        if timeout <= 0:
            logger.error("timeout must be positive")
            raise ValueError("timeout must be positive")
        if batch_size <= 0:
            logger.error("batch_size must be positive")
            raise ValueError("batch_size must be positive")

        action_rewards = {name: [] for name in action_names}
        action_map = dict(zip(action_names, actions))
        errors = []

        if platform.system() != "Emscripten":
            for action_name, action in action_map.items():
                for batch_start in range(0, num_simulations, batch_size):
                    batch_end = min(batch_start + batch_size, num_simulations)
                    batch_size_actual = batch_end - batch_start
                    try:
                        results = Parallel(n_jobs=n_jobs)(
                            delayed(self._simulate_action)(action, initial_state, reward_func, transition_model, horizon, gamma, timeout)
                            for _ in range(batch_size_actual)
                        )
                        action_rewards[action_name].extend([r for r in results if r is not None])
                        if progress_callback:
                            progress_callback((batch_end + sum(len(r) for r in action_rewards.values())) / (num_simulations * len(actions)))
                    except Exception as e:
                        errors.append(f"{action_name} batch {batch_start}-{batch_end}: {str(e)}")
        else:
            total_simulations = num_simulations * len(actions)
            completed = 0
            for batch_start in range(0, num_simulations, batch_size):
                batch_end = min(batch_start + batch_size, num_simulations)
                for action_name, action in action_map.items():
                    for _ in range(batch_end - batch_start):
                        reward = self._simulate_action(action, initial_state, reward_func, transition_model, horizon, gamma, timeout)
                        if reward is not None:
                            action_rewards[action_name].append(reward)
                        completed += 1
                        if progress_callback:
                            progress_callback(completed / total_simulations)

        if errors:
            logger.warning("%d simulation errors occurred: %s", len(errors), '; '.join(errors))

        avg_rewards = {name: np.mean(rewards) if rewards else float('nan') for name, rewards in action_rewards.items()}
        valid_rewards = {k: v for k, v in avg_rewards.items() if not np.isnan(v)}
        if not valid_rewards:
            logger.error("No valid rewards computed from simulations")
            raise ValueError("No valid rewards computed from simulations")
        best_action_name = max(valid_rewards, key=valid_rewards.get)
        logger.debug("Monte Carlo planning completed in %.2f seconds: best action %s", time.time() - start_time, best_action_name)
        return action_map[best_action_name], valid_rewards[best_action_name], valid_rewards

    def validate_params(self, problem_type: str, params: Dict) -> None:
        """
        Validate parameters for a given problem type.

        Args:
            problem_type: Type of optimization.
            params: Parameters to validate.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        solver_info = self.solvers.get(problem_type)
        if solver_info and solver_info['required_params']:
            missing = [p for p in solver_info['required_params'] if p not in params]
            if missing:
                logger.error("Missing required parameters for %s: %s", problem_type, missing)
                raise ValueError(f"Missing required parameters for {problem_type}: {missing}")

    def integrated_decision(self, problem_type: str, params: Dict) -> Any:
        """
        Integrated decision-making using LogicCore and ProbabilityCore.

        Args:
            problem_type: Type of optimization ('linear', 'csp', 'bayesian', 'monte_carlo', or custom).
            params: Parameters specific to the problem type.

        Returns:
            Tuple of:
            - For 'linear': (variable assignments, chart data)
            - For 'csp': (solutions list, None)
            - For 'bayesian': (best action, chart data)
            - For 'monte_carlo': (best action, average reward, chart data)
            - For custom: (solver result, chart data)

        Raises:
            ValueError: If problem type is invalid or required parameters are missing.
        """
        start_time = time.time()
        if problem_type not in self.solvers and problem_type not in ['linear', 'csp', 'bayesian', 'monte_carlo']:
            logger.error("Unknown problem type: %s", problem_type)
            raise ValueError(f"Unknown problem type: {problem_type}")
        self.validate_params(problem_type, params)

        if self.logic_core and hasattr(self.logic_core, 'get_rules') and problem_type in ['linear', 'csp']:
            params['constraints'] = self.logic_core.get_rules() + params.get('constraints', [])

        if self.probability_core and problem_type in ['bayesian', 'monte_carlo']:
            if hasattr(self.probability_core, 'get_probabilities'):
                params['probabilities'] = self.probability_core.get_probabilities()
            if hasattr(self.probability_core, 'transition_model'):
                params['transition_model'] = self.probability_core.transition_model

        if problem_type == 'linear':
            result = self.linear_programming(
                tuple(params['objective'].items()), tuple(params['constraints']), 
                params.get('integer', False), params.get('sense', 'maximize'),
                params.get('var_bounds')
            )
            chart_data = self.visualize_results(result, "Linear Programming Results", "Variables", "Values", 
                                              params.get('chart_type', 'bar'), params.get('custom_config'))
            logger.debug("Linear programming decision completed in %.2f seconds", time.time() - start_time)
            return result, chart_data
        elif problem_type == 'csp':
            result = self.constraint_satisfaction(
                tuple((k, tuple(v)) for k, v in params['variables'].items()), tuple(params['constraints'])
            )
            logger.debug("CSP decision completed in %.2f seconds", time.time() - start_time)
            return result, None
        elif problem_type == 'bayesian':
            best_action, expected_utilities = self.bayesian_decision(
                tuple(params['actions']), tuple(params['states']), 
                tuple(params['utilities'].items()), params['probabilities'],
                params.get('use_cache', True)
            )
            chart_data = self.visualize_results(expected_utilities, "Expected Utilities of Actions", "Actions", "Expected Utility",
                                              params.get('chart_type', 'bar'), params.get('custom_config'))
            logger.debug("Bayesian decision completed in %.2f seconds", time.time() - start_time)
            return best_action, chart_data
        elif problem_type == 'monte_carlo':
            best_action, avg_reward, avg_rewards = self.monte_carlo_planning(
                params['initial_state'], params['actions'], params.get('action_names', [f"Action_{i}" for i in range(len(params['actions']))]),
                params['reward_func'], params['transition_model'], params.get('num_simulations', 1000),
                params.get('horizon', 10), params.get('n_jobs', -1), params.get('gamma', 1.0), params.get('timeout', 30.0),
                params.get('progress_callback'), params.get('batch_size', 100)
            )
            chart_data = self.visualize_results(
                avg_rewards, "Monte Carlo Average Rewards", "Actions", "Average Reward",
                params.get('chart_type', 'bar'), params.get('custom_config')
            )
            logger.debug("Monte Carlo decision completed in %.2f seconds", time.time() - start_time)
            return best_action, avg_reward, chart_data
        else:
            solver_info = self.solvers.get(problem_type)
            if solver_info:
                result = solver_info['solver'](self, **{**solver_info['params'], **params})
                chart_data = self.visualize_results(
                    result if isinstance(result, dict) else {'Result': result},
                    f"{problem_type.capitalize()} Results", "Items", "Values",
                    params.get('chart_type', 'bar'), params.get('custom_config')
                )
                logger.debug("Custom solver decision completed in %.2f seconds", time.time() - start_time)
                return result, chart_data
            logger.error("No solver registered for problem type: %s", problem_type)
            raise ValueError(f"No solver registered for problem type: {problem_type}")

    def hybrid_optimization(self, primary_params: Dict, secondary_params: Dict, 
                           state_mapping: Callable[[Any, Any], Any]) -> Tuple[Any, Any]:
        """
        Perform hybrid optimization by combining two solvers.

        Args:
            primary_params: Parameters for the primary solver (e.g., linear, csp).
            secondary_params: Parameters for the secondary solver (e.g., monte_carlo).
            state_mapping: Function to map primary solver output to secondary solver input state.

        Returns:
            Tuple of (primary result, secondary result).

        Raises:
            ValueError: If solver types or state mapping are invalid.
        """
        start_time = time.time()
        primary_type = primary_params.get('problem_type')
        secondary_type = secondary_params.get('problem_type')
        if not primary_type or not secondary_type:
            logger.error("Both primary and secondary parameters must specify problem_type")
            raise ValueError("Both primary and secondary parameters must specify problem_type")
        self.validate_params(primary_type, primary_params)
        self.validate_params(secondary_type, secondary_params)

        primary_result, _ = self.integrated_decision(primary_type, primary_params)
        try:
            mapped_state = state_mapping(primary_result, secondary_params.get('initial_state'))
        except Exception as e:
            logger.error("State mapping failed: %s", str(e))
            raise ValueError(f"State mapping failed: {str(e)}")
        
        if secondary_type == 'monte_carlo' and not isinstance(mapped_state, dict):
            logger.error("Mapped state for Monte Carlo must be a dictionary")
            raise ValueError("Mapped state for Monte Carlo must be a dictionary")
        
        secondary_params['initial_state'] = mapped_state
        secondary_result = self.integrated_decision(secondary_type, secondary_params)
        logger.debug("Hybrid optimization completed in %.2f seconds: primary=%s, secondary=%s", 
                     time.time() - start_time, primary_type, secondary_type)
        return primary_result, secondary_result

    def sensitivity_analysis(self, problem_type: str, base_params: Dict, 
                            param_key: str, param_values: List[Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Perform sensitivity analysis by varying a parameter and collecting results.

        Args:
            problem_type: Type of optimization.
            base_params: Base parameters for the solver.
            param_key: Parameter to vary (e.g., 'probabilities', 'objective').
            param_values: List of values to test for the parameter.

        Returns:
            Tuple of (results dictionary, list of errors).

        Raises:
            ValueError: If inputs are invalid or parameter key is not found.
        """
        start_time = time.time()
        if problem_type not in self.solvers and problem_type not in ['linear', 'csp', 'bayesian', 'monte_carlo']:
            logger.error("Unknown problem type: %s", problem_type)
            raise ValueError(f"Unknown problem type: {problem_type}")
        if param_key not in base_params:
            logger.error("Parameter key %s not found in base parameters", param_key)
            raise ValueError(f"Parameter key {param_key} not found in base parameters")

        results = {}
        errors = []
        for i, value in enumerate(param_values):
            params = deepcopy(base_params)
            params[param_key] = value
            try:
                result = self.integrated_decision(problem_type, params)
                results[f"Run_{i}"] = result
            except Exception as e:
                errors.append(f"Run {i} with {param_key}={value}: {str(e)}")
                logger.warning("Sensitivity analysis run failed: %s", str(e))

        if problem_type in ['linear', 'bayesian', 'monte_carlo'] and results:
            data = {}
            for k, v in results.items():
                if problem_type == 'bayesian':
                    data[k] = v[1]['raw_data'][v[0]]  # Best action's utility
                elif problem_type == 'monte_carlo':
                    data[k] = v[1]  # Average reward
                else:  # linear
                    data[k] = sum(v[0].values())  # Sum of variable values
            chart_data = self.visualize_results(
                data, f"Sensitivity Analysis: {param_key}", "Runs", 
                "Utility" if problem_type == 'bayesian' else "Reward" if problem_type == 'monte_carlo' else "Objective Value",
                base_params.get('chart_type', 'line'), base_params.get('custom_config')
            )
            results['chart_data'] = chart_data

        logger.debug("Sensitivity analysis completed in %.2f seconds with %d runs, %d errors", 
                     time.time() - start_time, len(param_values), len(errors))
        return results, errors

    def generate_test_cases(self, problem_type: str, case_type: str = 'all') -> List[Dict[str, Any]]:
        """
        Generate test case parameters for a given problem type.

        Args:
            problem_type: Type of optimization ('linear', 'csp', 'bayesian', 'monte_carlo').
            case_type: Type of test cases to return ('normal', 'edge', 'stress', 'all').

        Returns:
            List of test case dictionaries.

        Raises:
            ValueError: If problem_type or case_type is invalid.
        """
        start_time = time.time()
        if problem_type not in ['linear', 'csp', 'bayesian', 'monte_carlo']:
            logger.error("Unknown problem type: %s", problem_type)
            raise ValueError(f"Unknown problem type: {problem_type}")
        if case_type not in ['normal', 'edge', 'stress', 'all']:
            logger.error("Unknown case type: %s", case_type)
            raise ValueError(f"Unknown case type: {case_type}")

        test_cases = []
        if problem_type == 'linear':
            normal_case = {
                'problem_type': 'linear',
                'objective': {'x1': 4, 'x2': 3},
                'constraints': [
                    ({'x1': 1, 'x2': 1}, '<=', 5),
                    ({'x1': 2, 'x2': 1}, '<=', 8)
                ],
                'integer': False,
                'sense': 'maximize',
                'var_bounds': {'x1': (0, 10), 'x2': (0, 5)},
                'chart_type': 'bar'
            }
            edge_case = {
                'problem_type': 'linear',
                'objective': {'x1': 1},
                'constraints': [({'x1': 1}, '>=', 10), ({'x1': 1}, '<=', 5)],
                'integer': False,
                'sense': 'maximize',
                'chart_type': 'bar'
            }
            stress_case = {
                'problem_type': 'linear',
                'objective': {f'x{i}': 1 for i in range(50)},
                'constraints': [({f'x{i}': 1 for i in range(50)}, '<=', 100)],
                'integer': False,
                'sense': 'maximize',
                'var_bounds': {f'x{i}': (0, 10) for i in range(50)},
                'chart_type': 'bar'
            }
            test_cases = (
                [normal_case] if case_type == 'normal' else
                [edge_case] if case_type == 'edge' else
                [stress_case] if case_type == 'stress' else
                [normal_case, edge_case, stress_case]
            )
        elif problem_type == 'csp':
            normal_case = {
                'problem_type': 'csp',
                'variables': {'task1': [1, 2, 3], 'task2': [1, 2, 3], 'task3': [1, 2, 3]},
                'constraints': [
                    lambda t1, t2, t3: t1 != t2,
                    lambda t1, t2, t3: t2 != t3,
                    lambda t1, t2, t3: t1 != t3
                ]
            }
            edge_case = {
                'problem_type': 'csp',
                'variables': {'task1': [1], 'task2': [1]},
                'constraints': [lambda t1, t2: t1 != t2]
            }
            mid_case = {
                'problem_type': 'csp',
                'variables': {f'task{i}': list(range(5)) for i in range(10)},
                'constraints': [lambda **kwargs: len(set(kwargs.values())) == len(kwargs)]
            }
            stress_case = {
                'problem_type': 'csp',
                'variables': {f'task{i}': list(range(10)) for i in range(15)},
                'constraints': [lambda **kwargs: len(set(kwargs.values())) == len(kwargs)]
            }
            test_cases = (
                [normal_case] if case_type == 'normal' else
                [edge_case] if case_type == 'edge' else
                [stress_case] if case_type == 'stress' else
                [normal_case, edge_case, mid_case, stress_case]
            )
        elif problem_type == 'bayesian':
            normal_case = {
                'problem_type': 'bayesian',
                'actions': ['invest_stock', 'invest_bond', 'wait'],
                'states': ['market_up', 'market_down'],
                'utilities': {
                    ('invest_stock', 'market_up'): 100,
                    ('invest_stock', 'market_down'): -50,
                    ('invest_bond', 'market_up'): 50,
                    ('invest_bond', 'market_down'): 20,
                    ('wait', 'market_up'): 0,
                    ('wait', 'market_down'): 0
                },
                'probabilities': {'market_up': 0.6, 'market_down': 0.4},
                'chart_type': 'bar',
                'use_cache': False
            }
            edge_case = {
                'problem_type': 'bayesian',
                'actions': ['invest'],
                'states': ['market_up'],
                'utilities': {('invest', 'market_up'): 100},
                'probabilities': {'market_up': 1.0},
                'chart_type': 'bar',
                'use_cache': False
            }
            stress_case = {
                'problem_type': 'bayesian',
                'actions': [f'action{i}' for i in range(10)],
                'states': [f'state{i}' for i in range(10)],
                'utilities': {(f'action{i}', f'state{j}'): (i + 1) * (j + 1) * 10 for i in range(10) for j in range(10)},
                'probabilities': {f'state{i}': 1/10 for i in range(10)},
                'chart_type': 'bar',
                'use_cache': False
            }
            test_cases = (
                [normal_case] if case_type == 'normal' else
                [edge_case] if case_type == 'edge' else
                [stress_case] if case_type == 'stress' else
                [normal_case, edge_case, stress_case]
            )
        elif problem_type == 'monte_carlo':
            def reward_func(state): return -abs(state['inventory'] - 50)
            def transition_model(state, action):
                new_state = state.copy()
                demand = self.rng.normal(50, 10)
                new_state['inventory'] = max(0, new_state['inventory'] + action(new_state) - demand)
                return new_state
            def failing_transition_model(state, action):
                raise ValueError("Simulated transition failure")
            normal_case = {
                'problem_type': 'monte_carlo',
                'initial_state': {'inventory': 50},
                'actions': [lambda s: 20, lambda s: 10, lambda s: 0],
                'action_names': ['Order_20', 'Order_10', 'Order_0'],
                'reward_func': reward_func,
                'transition_model': transition_model,
                'num_simulations': 1000,
                'horizon': 5,
                'n_jobs': 2,
                'gamma': 0.9,
                'timeout': 30.0,
                'batch_size': 100,
                'chart_type': 'bar'
            }
            edge_case = {
                'problem_type': 'monte_carlo',
                'initial_state': {'inventory': 50},
                'actions': [lambda s: 0],
                'action_names': ['Order_0'],
                'reward_func': reward_func,
                'transition_model': transition_model,
                'num_simulations': 10,
                'horizon': 1,
                'n_jobs': 1,
                'gamma': 1.0,
                'timeout': 5.0,
                'batch_size': 10,
                'chart_type': 'bar'
            }
            stress_case = {
                'problem_type': 'monte_carlo',
                'initial_state': {'inventory': 50},
                'actions': [lambda s: i * 10 for i in range(8)],
                'action_names': [f'Order_{i*10}' for i in range(8)],
                'reward_func': reward_func,
                'transition_model': transition_model,
                'num_simulations': 5000,
                'horizon': 15,
                'n_jobs': 4,
                'gamma': 0.95,
                'timeout': 60.0,
                'batch_size': 500,
                'chart_type': 'bar'
            }
            test_cases = (
                [normal_case] if case_type == 'normal' else
                [edge_case] if case_type == 'edge' else
                [stress_case] if case_type == 'stress' else
                [normal_case, edge_case, stress_case]
            )
        logger.info("Generated %d test cases for %s in %.2f seconds", len(test_cases), problem_type, time.time() - start_time)
        return test_cases

# Register default solvers
OptimizationCore.register_solver('linear', OptimizationCore.linear_programming, required_params=['objective', 'constraints'])
OptimizationCore.register_solver('csp', OptimizationCore.constraint_satisfaction, required_params=['variables', 'constraints'])
OptimizationCore.register_solver('bayesian', OptimizationCore.bayesian_decision, required_params=['actions', 'states', 'utilities', 'probabilities'])
OptimizationCore.register_solver('monte_carlo', OptimizationCore.monte_carlo_planning, 
                                required_params=['initial_state', 'actions', 'reward_func', 'transition_model'])

# Example usage
if __name__ == "__main__":
    optimizer = OptimizationCore(seed=42)

    # Linear Programming Example
    lp_params = {
        'problem_type': 'linear',
        'objective': {'x1': 4, 'x2': 3},
        'constraints': [
            ({'x1': 1, 'x2': 1}, '<=', 5),
            ({'x1': 2, 'x2': 1}, '<=', 8)
        ],
        'integer': False,
        'sense': 'maximize',
        'var_bounds': {'x1': (0, 10), 'x2': (0, 5)},
        'chart_type': 'bar',
        'custom_config': {'options': {'scales': {'y': {'beginAtZero': True}}}}
    }
    lp_result, lp_chart = optimizer.integrated_decision('linear', lp_params)
    print("Linear Programming Result:", lp_result)
    print("Linear Programming Chart (Chart.js JSON):", json.dumps(lp_chart['chart_config'], indent=2))

    # CSP Example
    csp_params = {
        'problem_type': 'csp',
        'variables': {'task1': [1, 2, 3], 'task2': [1, 2, 3], 'task3': [1, 2, 3]},
        'constraints': [
            lambda t1, t2, t3: t1 != t2,
            lambda t1, t2, t3: t2 != t3,
            lambda t1, t2, t3: t1 != t3
        ]
    }
    csp_result, _ = optimizer.integrated_decision('csp', csp_params)
    print("CSP Result (Scheduling):", csp_result)

    # Bayesian Decision Example
    bayesian_params = {
        'problem_type': 'bayesian',
        'actions': ['invest_stock', 'invest_bond', 'wait'],
        'states': ['market_up', 'market_down'],
        'utilities': {
            ('invest_stock', 'market_up'): 100,
            ('invest_stock', 'market_down'): -50,
            ('invest_bond', 'market_up'): 50,
            ('invest_bond', 'market_down'): 20,
            ('wait', 'market_up'): 0,
            ('wait', 'market_down'): 0
        },
        'probabilities': {'market_up': 0.6, 'market_down': 0.4},
        'chart_type': 'pie',
        'custom_config': {'options': {'legend': {'position': 'right'}}},
        'use_cache': False
    }
    bayesian_result, bayesian_chart = optimizer.integrated_decision('bayesian', bayesian_params)
    print("Bayesian Decision Result:", bayesian_result)
    print("Bayesian Chart (Chart.js JSON):", json.dumps(bayesian_chart['chart_config'], indent=2))

    # Monte Carlo Planning Example
    def reward_func(state): return -abs(state['inventory'] - 50)
    def transition_model(state, action):
        new_state = state.copy()
        demand = optimizer.rng.normal(50, 10)
        new_state['inventory'] = max(0, new_state['inventory'] + action(new_state) - demand)
        return new_state

    mc_params = {
        'problem_type': 'monte_carlo',
        'initial_state': {'inventory': 50},
        'actions': [lambda s: 20, lambda s: 10, lambda s: 0],
        'action_names': ['Order_20', 'Order_10', 'Order_0'],
        'reward_func': reward_func,
        'transition_model': transition_model,
        'num_simulations': 1000,
        'horizon': 5,
        'n_jobs': 2,
        'gamma': 0.9,
        'timeout': 30.0,
        'batch_size': 100,
        'chart_type': 'bar',
        'progress_callback': lambda p: print(f"Monte Carlo Progress: {p*100:.1f}%")
    }
    mc_result, mc_reward, mc_chart = optimizer.integrated_decision('monte_carlo', mc_params)
    print("Monte Carlo Planning Result (Order Quantity):", mc_result({'inventory': 50}), f"Avg Reward: {mc_reward}")
    print("Monte Carlo Chart (Chart.js JSON):", json.dumps(mc_chart['chart_config'], indent=2))

    # Hybrid Optimization Example
    def state_mapping(lp_result, mc_initial_state):
        return {'inventory': sum(lp_result.values())}
    hybrid_result = optimizer.hybrid_optimization(lp_params, mc_params, state_mapping)
    print("Hybrid Optimization Result:", hybrid_result)

    # Sensitivity Analysis Example
    sensitivity_results, sensitivity_errors = optimizer.sensitivity_analysis(
        'bayesian', bayesian_params, 'probabilities', 
        [{'market_up': p, 'market_down': 1-p} for p in [0.5, 0.6, 0.7, 0.8]]
    )
    print("Sensitivity Analysis Results:", sensitivity_results)
    print("Sensitivity Analysis Errors:", sensitivity_errors)
    if 'chart_data' in sensitivity_results:
        print("Sensitivity Analysis Chart (Chart.js JSON):", json.dumps(sensitivity_results['chart_data']['chart_config'], indent=2))

    # Test Case Generation Example
    for problem_type in ['linear', 'csp', 'bayesian', 'monte_carlo']:
        print(f"\nGenerated {problem_type.capitalize()} Test Cases:")
        test_cases = optimizer.generate_test_cases(problem_type, case_type='all')
        for i, case in enumerate(test_cases):
            print(f"Test Case {i+1}:", case)
            try:
                result = optimizer.integrated_decision(problem_type, case)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error: {str(e)}")