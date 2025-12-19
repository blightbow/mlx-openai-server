"""Mock implementations for distributed testing.

These mocks simulate MLX distributed primitives for testing code structure
and call patterns without requiring actual multi-node hardware.

IMPORTANT: FakeDistributed produces WRONG results for actual communication
verification. Use only for testing code structure and call patterns, not
correctness of distributed algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeGroup:
    """Mock distributed group matching mx.distributed.Group interface.

    Provides rank() and size() methods that return configured values.
    """

    _rank: int = 0
    _size: int = 4

    def rank(self) -> int:
        """Return the rank of this process in the group."""
        return self._rank

    def size(self) -> int:
        """Return the total number of processes in the group."""
        return self._size


@dataclass
class FakeDistributed:
    """Mock MLX distributed module for testing.

    Simulates distributed operations without actual communication.
    All collective operations are no-ops that return predictable values.

    Parameters
    ----------
    rank : int
        The rank to assign to this mock process (default: 0)
    world_size : int
        The total number of processes to simulate (default: 4)

    Examples
    --------
    >>> fake = FakeDistributed(rank=0, world_size=2)
    >>> group = fake.init()
    >>> group.rank()
    0
    >>> group.size()
    2
    """

    rank: int = 0
    world_size: int = 4
    _group: FakeGroup = field(init=False)
    _init_called: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize the fake group."""
        self._group = FakeGroup(_rank=self.rank, _size=self.world_size)

    def init(self, backend: str = "fake") -> FakeGroup:
        """Mock mx.distributed.init().

        Parameters
        ----------
        backend : str
            Ignored; for API compatibility.

        Returns
        -------
        FakeGroup
            A mock group with configured rank and size.
        """
        self._init_called = True
        return self._group

    def all_sum(self, x: Any) -> Any:
        """Mock all_sum - returns input scaled by world_size.

        This simulates the result of summing identical values across ranks.
        NOT accurate for real distributed verification.
        """
        return x * self._group.size()

    def all_gather(self, x: Any) -> list[Any]:
        """Mock all_gather - returns list of copies.

        This simulates gathering identical values from all ranks.
        NOT accurate for real distributed verification.
        """
        return [x] * self._group.size()

    def send(self, x: Any, dst: int) -> None:
        """Mock send - no-op."""

    def recv_like(self, x: Any, src: int) -> Any:
        """Mock recv_like - returns input unchanged."""
        return x

    @property
    def was_initialized(self) -> bool:
        """Check if init() was called."""
        return self._init_called


def create_mock_distributed(
    mocker: Any,
    rank: int = 0,
    world_size: int = 4,
) -> FakeDistributed:
    """Create and install a FakeDistributed mock using pytest-mock.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        The pytest-mock mocker fixture.
    rank : int
        The rank to assign (default: 0).
    world_size : int
        The world size to simulate (default: 4).

    Returns
    -------
    FakeDistributed
        The installed mock, for assertions.

    Examples
    --------
    >>> def test_pipeline_init(mocker):
    ...     fake = create_mock_distributed(mocker, rank=0, world_size=2)
    ...     # Code under test calls mx.distributed.init()
    ...     assert fake.was_initialized
    """
    fake = FakeDistributed(rank=rank, world_size=world_size)
    mocker.patch("mlx.core.distributed.init", fake.init)
    mocker.patch("mlx.core.distributed.all_sum", fake.all_sum)
    mocker.patch("mlx.core.distributed.all_gather", fake.all_gather)
    mocker.patch("mlx.core.distributed.send", fake.send)
    mocker.patch("mlx.core.distributed.recv_like", fake.recv_like)
    return fake
