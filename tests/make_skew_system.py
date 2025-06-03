"""
Test the skew system generation pipeline
NOTE: this is not a unit test, just a script to make skew systems for testing
TODO: consider converting a slimmed down version of this into a unit test
"""

import logging
from functools import partial
from itertools import permutations
from multiprocessing import Pool
from typing import Callable

import numpy as np

import dysts.flows as flows
from dysts.attractor import (
    check_boundedness,
    check_lyapunov_exponent,
    check_not_fixed_point,
    check_not_limit_cycle,
    check_not_linear,
    check_power_spectrum,
    check_stationarity,
    check_zero_one_test,
)
from dysts.base import DynSys, SkewProduct
from dysts.coupling import (
    RandomAdditiveCouplingMap,
)
from dysts.generator import DynSysSampler
from dysts.sampling import (
    InstabilityEvent,
    OnAttractorInitCondSampler,
    SignedGaussianParamSampler,
    TimeLimitEvent,
    TimeStepEvent,
)
from dysts.systems import get_attractor_list


def default_attractor_tests(tests_to_use: list[str] | None = None) -> list[Callable]:
    """Builds default attractor tests to check for each trajectory ensemble"""
    default_tests = {
        "check_not_linear": partial(check_not_linear, r2_threshold=0.99, eps=1e-10),
        "check_boundedness": partial(check_boundedness, threshold=1e4, max_zscore=15),
        "check_not_fixed_point": partial(
            check_not_fixed_point, atol=1e-3, tail_prop=0.1
        ),
        "check_zero_one_test": partial(
            check_zero_one_test, threshold=0.2, strategy="score"
        ),
        "check_not_limit_cycle": partial(
            check_not_limit_cycle,
            tolerance=1e-3,
            min_prop_recurrences=0.1,
            min_counts_per_rtime=200,
            min_block_length=50,
            enforce_endpoint_recurrence=True,
        ),
        "check_power_spectrum": partial(
            check_power_spectrum, rel_peak_height=1e-5, rel_prominence=1e-5, min_peaks=4
        ),
        "check_lyapunov_exponent": partial(check_lyapunov_exponent, traj_len=200),
        "check_stationarity": partial(check_stationarity, p_value=0.05),
    }

    if tests_to_use is None:
        tests_to_use = list(default_tests.keys())

    return [test for name, test in default_tests.items() if name in tests_to_use]


def additive_coupling_map_factory(
    driver_name: str,
    response_name: str,
    stats_cache: dict[str, dict[str, np.ndarray]],
    transform_scales: bool = True,
    randomize_driver_indices: bool = True,
    normalization_strategy: str = "flow_rms",
    random_seed: int = 0,
) -> Callable[[int, int], RandomAdditiveCouplingMap]:
    """
    Initialize a random additive coupling map for a skew-product dynamical system
    """
    driver_stats = stats_cache[driver_name]
    response_stats = stats_cache[response_name]
    if normalization_strategy == "mean_amp_response":
        # NOTE: response_stats actually returns reciprocals of stats i.e. stiffness / amplitude
        mean_amp_response = 1 / response_stats.get("mean_amp", 1.0)
        inv_mean_amp_driver = driver_stats.get("mean_amp", 1.0)
        driver_scale = mean_amp_response * inv_mean_amp_driver
        response_scale = 1.0
    else:
        driver_scale = driver_stats.get(normalization_strategy, 1.0)
        response_scale = response_stats.get(normalization_strategy, 1.0)

    return partial(
        RandomAdditiveCouplingMap,
        driver_scale=driver_scale,
        response_scale=response_scale,
        transform_scales=transform_scales,
        randomize_driver_indices=randomize_driver_indices,
        random_seed=random_seed,
    )


def sample_skew_systems(
    systems: list[str], num_pairs: int, random_seed: int = 0
) -> list[tuple[str, str]]:
    system_pairs = list(permutations(systems, 2))
    rng = np.random.default_rng(random_seed)
    sampled_pairs = rng.choice(
        len(system_pairs), size=min(num_pairs, len(system_pairs)), replace=False
    )
    return [system_pairs[i] for i in sampled_pairs]


def init_skew_system(
    driver_name: str, response_name: str, coupling_map_fn: Callable, **kwargs
) -> DynSys:
    """
    Initialize a skew-product dynamical system with a driver and response system

    Args:
        driver_name: name of the driver system
        response_name: name of the response system
        coupling_map_fn: function for initializing the coupling map
        kwargs: additional arguments for the SkewProduct constructor
    """
    driver = getattr(flows, driver_name)()
    response = getattr(flows, response_name)()
    coupling_map = coupling_map_fn(driver.dimension, response.dimension)
    return SkewProduct(
        driver=driver, response=response, coupling_map=coupling_map, **kwargs
    )


def filter_and_split_skew_systems(
    skew_pairs: list[tuple[str, str]],
    test_split: float = 0.2,
    train_systems: list[str] | None = None,
    test_systems: list[str] | None = None,
    coupling_map_type: str = "additive",
    coupling_map_kwargs: dict | None = None,
    skew_system_kwargs: dict | None = None,
) -> tuple[list[str], list[str]]:
    """Sample skew systems from all pairs of non-skew systems and split into train/test

    Args:
        skew_pairs: List of skew system pairs to sample from
        test_split: Fraction of systems to use for testing
        random_seed: Random seed for reproducibility
        train_systems: Optional list of system names to use for training
        test_systems: Optional list of system names to use for testing

    Returns:
        Tuple of (train_systems, test_systems) where each is a list of initialized
        skew product systems
    """
    coupling_map_kwargs = coupling_map_kwargs or {}
    skew_system_kwargs = skew_system_kwargs or {}

    split_idx = int(len(skew_pairs) * (1 - test_split))
    train_pairs, test_pairs = skew_pairs[:split_idx], skew_pairs[split_idx:]

    # if provided, filter out pairs from train and test pairs that contain systems
    # that are not in the train or test sets, then recombine to update valid train/test pairs
    def is_valid_pair(pair: tuple[str, str], filter_list: list[str] | None) -> bool:
        return (
            True
            if filter_list is None
            else all(system in filter_list for system in pair)
        )

    valid_train_pairs = filter(
        lambda pair: is_valid_pair(pair, train_systems), train_pairs
    )
    valid_test_pairs = filter(
        lambda pair: is_valid_pair(pair, test_systems), test_pairs
    )
    invalid_train_pairs = filter(
        lambda pair: not is_valid_pair(pair, train_systems), train_pairs
    )
    invalid_test_pairs = filter(
        lambda pair: not is_valid_pair(pair, test_systems), test_pairs
    )
    train_pairs = list(valid_train_pairs) + list(invalid_test_pairs)
    test_pairs = list(valid_test_pairs) + list(invalid_train_pairs)

    coupling_map_factory = {
        "additive": additive_coupling_map_factory,
    }[coupling_map_type]

    systems = {}
    for split, skew_pairs in [("train", train_pairs), ("test", test_pairs)]:
        systems[split] = [
            init_skew_system(
                driver,
                response,
                coupling_map_fn=coupling_map_factory(
                    driver, response, **coupling_map_kwargs
                ),
                **skew_system_kwargs,
            )
            for driver, response in skew_pairs
        ]

    return systems["train"], systems["test"]


def _compute_system_stats(
    system: str,
    n: int,
    num_periods: int,
    transient: int,
    atol: float,
    rtol: float,
    stiffness: float = 1.0,
) -> tuple[str, dict[str, np.ndarray]]:
    """
    Compute RMS scale and amplitude for a single system's trajectory.
    Returns the reciprocals of the computed stats, with a stiffness factor applied
    """
    sys = getattr(flows, system)()
    ts, traj = sys.make_trajectory(
        n, pts_per_period=n // num_periods, return_times=True, atol=atol, rtol=rtol
    )
    assert traj is not None, f"{system} should be integrable"
    ts, traj = ts[transient:], traj[transient:]
    flow_rms = np.sqrt(
        np.mean([np.asarray(sys(x, t)) ** 2 for x, t in zip(traj, ts)], axis=0)
    )
    mean_amp = np.mean(np.abs(traj), axis=0)
    system_stats = {
        "flow_rms": stiffness / flow_rms,
        "mean_amp": stiffness / mean_amp,
    }
    return system, system_stats


def init_trajectory_stats_cache(
    systems: list[str],
    traj_length: int,
    num_periods: int,
    traj_transient: float,
    atol: float,
    rtol: float,
    multiprocess_kwargs: dict = {},
) -> dict[str, dict[str, np.ndarray]]:
    """Initialize a cache of vector field RMS scales and amplitudes for each system using multiprocessing"""
    _compute_stats_worker = partial(
        _compute_system_stats,
        n=traj_length,
        num_periods=num_periods,
        transient=int(traj_length * traj_transient),
        atol=atol,
        rtol=rtol,
    )
    with Pool(**multiprocess_kwargs) as pool:
        results = pool.map(_compute_stats_worker, systems)
    return dict(results)


def test_skew_system_generation(
    num_points: int = 640,
    num_periods: int = 12,
    num_ics: int = 1,
    num_param_perturbations: int = 1,
    test_split: float = 0.5,
    atol: float = 1e-6,  # reduced precision for testing
    rtol: float = 1e-5,  # reduced precision for testing
    transient: float = 0.2,
    param_scale: float = 1.0,
    instability_threshold: float = 1e4,
    max_duration: int = 15,
    min_step: float = 1e-6,  # reduced precision for testing
    verbose: bool = False,
    silence_integration_errors: bool = True,
    standardize: bool = True,
    split_coords: bool = False,
    selected_attractor_tests: list[str] = [
        "check_not_linear",
        "check_boundedness",
        "check_not_fixed_point",
        "check_power_spectrum",
    ],
    multiprocessing: bool = True,
    num_skew_pairs: int = 2,
    rseed: int = 42,
):
    """Test the generation of skew product systems and their trajectories"""
    # Get test systems. NOTE: we only use the first 4 systems for testing
    system_names = get_attractor_list(sys_class="continuous")

    # logger.info(f"Systems: {system_names}")

    # Initialize events
    time_limit_event = partial(
        TimeLimitEvent,
        max_duration=max_duration,
        verbose=verbose,
    )
    instability_event = partial(
        InstabilityEvent,
        threshold=instability_threshold,
        verbose=verbose,
    )
    time_step_event = partial(
        TimeStepEvent,
        min_step=min_step,
        verbose=verbose,
    )
    event_fns = [time_limit_event, instability_event, time_step_event]

    # Initialize samplers
    param_sampler = SignedGaussianParamSampler(
        random_seed=rseed,
        scale=param_scale,
        verbose=verbose,
    )
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=num_points,
        reference_traj_n_periods=num_periods,
        reference_traj_transient=transient,
        reference_traj_atol=atol,
        reference_traj_rtol=rtol,
        recompute_standardization=standardize,
        random_seed=rseed,
        events=event_fns,
        silence_integration_errors=silence_integration_errors,
        verbose=verbose,
    )

    # Initialize system sampler
    sys_sampler = DynSysSampler(
        rseed=rseed,
        num_periods=[num_periods],
        num_points=num_points,
        num_ics=num_ics,
        num_param_perturbations=num_param_perturbations,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=event_fns,
        verbose=verbose,
        split_coords=split_coords,
        attractor_tests=default_attractor_tests(tests_to_use=selected_attractor_tests),
        validator_transient_frac=transient,
        save_failed_trajs=False,
        multiprocess_kwargs={},
    )

    # sample skew system train/test splits
    skew_pairs = sample_skew_systems(system_names, num_skew_pairs, random_seed=rseed)
    logger.info(f"Making {len(skew_pairs)} skew pairs: {skew_pairs}")

    # Initialize stats cache
    base_systems = set(sys for pair in skew_pairs for sys in pair)
    logger.info(f"Initializing trajectory scale cache for {len(base_systems)} systems")

    stats_cache = init_trajectory_stats_cache(
        list(base_systems),
        num_points,
        num_periods,
        transient,
        atol=atol,
        rtol=rtol,
        multiprocess_kwargs={},
    )

    # Split into train/test
    train_systems, test_systems = filter_and_split_skew_systems(
        skew_pairs,
        test_split=test_split,
        coupling_map_type="additive",
        coupling_map_kwargs={
            "stats_cache": stats_cache,
            "transform_scales": True,
            "randomize_driver_indices": True,
            "normalization_strategy": "flow_rms",
            "random_seed": rseed,
        },
    )

    logger.info(f"Train systems: {train_systems}")
    logger.info(f"Test systems: {test_systems}")

    # Test that we have both train and test systems
    assert len(train_systems) > 0, "No training systems generated"
    assert len(test_systems) > 0, "No test systems generated"

    for split, systems in [("train", train_systems), ("test", test_systems)]:
        sys_sampler.sample_ensembles(
            systems=systems,
            save_dir=None,
            split=split,
            split_failures=f"failed_attractors_{split}",
            samples_process_interval=1,
            save_params_dir=None,
            save_traj_stats_dir=None,
            standardize=standardize,
            use_multiprocessing=multiprocessing,
            reset_attractor_validator=True,
            silent_errors=silence_integration_errors,
            atol=atol,
            rtol=rtol,
            use_tqdm=False,
        )

        summary_dict = sys_sampler.save_summary()
        logger.info(f"Summary for {split} split: {summary_dict}")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    test_skew_system_generation()
