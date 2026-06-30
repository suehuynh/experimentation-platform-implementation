"""
Experimentation Platform: Hash Assignment.

This module provides stateless, deterministic user-to-variant mapping 
for A/B testing using consistent hashing and fractional traffic allocation.
"""

from dataclasses import dataclass
from typing import Protocol, Dict, List, Tuple, Any
from typing import runtime_checkable
import math
import time


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantSpec:
    """
    Defines a single experiment variant and its allocated traffic proportion.

    Attributes:
        name (str): Unique identifier for the variant (e.g., 'control', 'treatment_v1').
        percentage (float): Traffic allocation represented as a fraction between 0.0 and 1.0.
    """
    name: str
    percentage: float


@dataclass(frozen=True)
class ExperimentConfig:
    """
    An immutable configuration representing traffic distribution rules for an experiment.

    Traffic distribution is managed via an ordered bucket architecture. The sequence 
    of variants dictates the cumulative cutoff boundaries. To prevent user migration 
    during traffic ramps, the order of variants must remain fixed across updates.

    Attributes:
        experiment_id (str): Globally unique identifier for the experiment.
        salt (str): Entropy modifier combined with the hash input to eliminate cross-experiment 
            assignment correlation (preventing selection bias across overlapping experiments).
        variants (tuple[VariantSpec, ...]): An ordered tuple of variant specifications.
    """
    experiment_id: str
    salt: str
    variants: Tuple[VariantSpec, ...]

    def validate(self) -> None:
        """
        Validates the integrity of the experiment configuration.

        Raises:
            AssertionError: If total traffic allocation does not approximate 100%, 
                if negative allocations exist, or if variant names are duplicated.
        """
        current_pct = 0.0
        seen_variants = set()
        
        for variant in self.variants:
            assert variant.percentage >= 0, f"Variant '{variant.name}' cannot have a negative allocation."
            assert variant.name not in seen_variants, f"Duplicate variant name detected: '{variant.name}'."
            current_pct += variant.percentage
            seen_variants.add(variant.name)
            
        assert math.isclose(current_pct, 1.0, rel_tol=1e-9), f"Total traffic allocation must equal 1.0, got {current_pct}."

    def cumulative_cutoffs(self) -> List[Tuple[str, float, float]]:
        """
        Computes the absolute, sequential intervals for fractional user mapping.

        Returns:
            List[Tuple[str, float, float]]: A list of tuples containing the variant name, 
                lower bound (inclusive), and upper bound (exclusive).
                Example: [("control", 0.0, 0.5), ("treatment", 0.5, 1.0)]
        """
        running_total = 0.0
        cutoffs = []
        for variant in self.variants:
            lower = running_total
            running_total += variant.percentage
            upper = running_total
            cutoffs.append((variant.name, lower, upper))
            
        assert math.isclose(running_total, 1.0, rel_tol=1e-9)
        return cutoffs

    def __post_init__(self) -> None:
        """Executes automated data integrity guards immediately following object instantiation."""
        self.validate()


# ---------------------------------------------------------------------------
# 2. Hashing Architecture
# ---------------------------------------------------------------------------
@runtime_checkable
class HashFunction(Protocol):
    """
    Structural protocol defining an interchangeable hashing strategy.

    Allows seamless substitution of underlying hashing backends (e.g., MurmurHash3, 
    xxHash, or deterministic mock implementations) without introducing tight coupling.

    Attributes:
        max_value (int): The upper limit (inclusive) of the hash space domain, 
            used downstream for interval normalization.
    """
    max_value: int

    def __call__(self, input_str: str) -> int:
        """Computes the unsigned integer hash value of a given string payload."""
        ...


def murmur_hash(input_str: str) -> int:
    """
    Computes an unsigned 32-bit MurmurHash3 value.

    Guarantees cross-runtime and cross-platform deterministic output with high 
    avalanche properties, ensuring uniform distribution across the keyspace.

    Args:
        input_str (str): The raw string identifier to hash.

    Returns:
        int: A uniformly distributed 32-bit unsigned integer.
    """
    import mmh3  # Imported inline to highlight third-party dependency encapsulation
    return mmh3.hash(input_str.encode("utf-8"), signed=False)

# Expose the cryptographic space upper bound for dynamic normalization
murmur_hash.max_value = (2**32) - 1


# ---------------------------------------------------------------------------
# 3. Assignment Strategy
# ---------------------------------------------------------------------------

class HashAssignment:
    """
    Stateless evaluation engine mapping experimental units to specific configurations.

    Design Architectural Controls:
        1. Pure Functions: Assignments depend strictly on inputs, making the pipeline 
           thread-safe, highly cacheable, and easily parallelizable.
        2. Interval Normalization: The variant search-space maps uniformly to [0.0, 1.0), 
           decoupling underlying partition scales from business traffic configurations.
        3. Compound Key Isolation: Salt configurations isolate specific variants across 
           separate test parameters, ensuring zero statistical correlation.
    """

    def __init__(self, hash_fn: HashFunction = murmur_hash):
        """
        Initializes the assignment engine with an injected hashing backend.

        Args:
            hash_fn (HashFunction): An object or function satisfying the HashFunction 
                protocol. Defaults to `murmur_hash`.
        """
        if not hasattr(hash_fn, "max_value"):
            raise ValueError(
                f"{hash_fn} must define a 'max_value' attribute"
            )
        self._hash_fn = hash_fn

    def _to_unit_interval(self, user_id: str, config: ExperimentConfig) -> float:
        """
        Projects an entity identifier uniformly into a continuous range between [0, 1).

        Args:
            user_id (str): Unique experimental unit identifier (e.g., account UUID).
            config (ExperimentConfig): Target experiment metadata structure.

        Returns:
            float: Normalized distribution point within the range [0.0, 1.0).
        """
        composite_key = f"{user_id}:{config.experiment_id}:{config.salt}"
        raw_hash = self._hash_fn(composite_key)
        return raw_hash / (self._hash_fn.max_value+ 1)

    def assign(self, user_id: str, config: ExperimentConfig) -> str:
        """
        Maps an experimental unit deterministically to an experiment variant.

        Args:
            user_id (str): Unique experimental unit identifier.
            config (ExperimentConfig): Target experiment configuration parameters.

        Returns:
            str: Name of the assigned experiment variant.

        Raises:
            AssertionError: If total variation spaces fail to encapsulate the evaluation point.
        """
        point = self._to_unit_interval(user_id, config)
        for name, lower, upper in config.cumulative_cutoffs():
            if lower <= point < upper:
                return name
        
        raise AssertionError(
            f"Evaluation threshold violation: assigned point {point} is unmapped. "
            "Ensure cumulative variations map perfectly to 1.0 total threshold."
        )

    def assign_batch(self, user_ids: List[str], config: ExperimentConfig) -> Dict[str, str]:
        """
        Batch processes assignments for multiple experimental units.

        Highly performant wrapper suitable for handling batch predictions, local service 
        pipelines, or high-throughput simulation runs (such as synthetic A/A validations).

        Args:
            user_ids (List[str]): Collection of experimental unit identifiers.
            config (ExperimentConfig): Core experiment parameters.

        Returns:
            Dict[str, str]: Map of user identifiers to their resolved variant names.
        """
        return {user_id: self.assign(user_id, config) for user_id in user_ids}


# ---------------------------------------------------------------------------
# 4. Persistence & Audit Layer
# ---------------------------------------------------------------------------

class AssignmentLogger:
    """
    Decoupled logger handling operational state recording for user assignments.

    Separating state recording from the assignment engine preserves the purity of 
    the core hashing logic, enabling isolated integration testing and flexible persistence 
    backends (such as data streams, relational systems, or distributed object storage).
    """

    def __init__(self, assigner: HashAssignment):
        """
        Initializes the state auditor wrapping an active assignment engine.

        Args:
            assigner (HashAssignment): The downstream evaluation instance.
        """
        self._assigner = assigner
        self._log: List[Dict[str, Any]] = []

    def assign_and_log(self, user_id: str, config: ExperimentConfig) -> str:
        """
        Evaluates a user assignment variant and logs the resulting event metadata.

        Args:
            user_id (str): Unique experimental unit identifier.
            config (ExperimentConfig): Targeted execution metadata profile.

        Returns:
            str: Name of the assigned experiment variant.
        """
        variant = self._assigner.assign(user_id, config)

        event_payload = {
            "user_id": user_id,
            "experiment_id": config.experiment_id,
            "variant": variant,
            "timestamp": time.time_ns()
        }

        self._log.append(event_payload)
        return variant

    def flush_logs(self) -> List[Dict[str, Any]]:
        """
        Extracts all captured assignment events and clears the internal buffer.

        Prevents unbounded in-memory memory growth in production environments,
        allowing a background runner or consumer to easily offload logs into long-term 
        storage (e.g., a Pandas DataFrame, an SQLite database, or a Kafka stream).

        Returns:
            List[Dict[str, Any]]: A list of event payloads recorded since the last flush.
        """
        records = self._log
        self._log = []
        return records