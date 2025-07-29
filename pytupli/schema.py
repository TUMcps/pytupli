from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from enum import Enum, StrEnum, auto
from typing import Any, Dict, List, Optional, Self
import uuid
import numpy as np

from pydantic import BaseModel, Field, model_validator


class RESOURCE_TYPE(StrEnum):
    ARTIFACT = auto()
    BENCHMARK = auto()
    EPISODE = auto()
    USER = auto()
    GROUP = auto()
    ROLE = auto()


class RIGHT(StrEnum):
    """
    Enum for defining rights in the system.
    """

    BENCHMARK_CREATE = auto()
    BENCHMARK_READ = auto()
    BENCHMARK_DELETE = auto()

    ARTIFACT_CREATE = auto()
    ARTIFACT_READ = auto()
    ARTIFACT_DELETE = auto()

    EPISODE_CREATE = auto()
    EPISODE_READ = auto()
    EPISODE_DELETE = auto()

    USER_CREATE = auto()
    USER_READ = auto()
    USER_UPDATE = auto()
    USER_DELETE = auto()

    GROUP_CREATE = auto()
    GROUP_READ = auto()
    GROUP_UPDATE = auto()
    GROUP_DELETE = auto()

    ROLE_MANAGEMENT = auto()


RIGHT_BUNDLE_CONTENT_READ = [RIGHT.ARTIFACT_READ, RIGHT.BENCHMARK_READ, RIGHT.EPISODE_READ]

RIGHT_BUNDLE_CONTENT_CREATE = [RIGHT.ARTIFACT_CREATE, RIGHT.BENCHMARK_CREATE, RIGHT.EPISODE_CREATE]

RIGHT_BUNDLE_CONTENT_DELETE = [RIGHT.ARTIFACT_DELETE, RIGHT.BENCHMARK_DELETE, RIGHT.EPISODE_DELETE]


class DEFAULT_ROLE(StrEnum):
    """
    Enum for defining roles in the system.
    """

    ADMIN = auto()
    CONTENT_ADMIN = auto()
    USER_ADMIN = auto()
    GROUP_ADMIN = auto()
    CONTRIBUTOR = auto()
    MEMBER = auto()
    GLOBAL_MEMBER = auto()
    GUEST = auto()


class DBItem(BaseModel):
    id: str
    created_by: str
    published_in: list[str] = []
    created_at: str

    @classmethod
    def create_new(cls, **data) -> DBItem:
        return cls(
            id=uuid.uuid4().hex,
            created_at=datetime.datetime.now().isoformat(),
            published_in=[data.get('created_by')],
            **data,
        )


class ArtifactMetadata(BaseModel):
    name: str
    description: str = ''
    # Artifacts can be associated with a specific benchmark or stand for themselves, e.g., if they are used by multiple benchmarks
    benchmark_id: str | None = None
    # Allow for additional arbitrary metadata
    extra: Dict[str, Any] = {}


class BenchmarkMetadata(BaseModel):
    name: str
    description: str = ''
    difficulty: str | None = None
    version: str | None = None
    # Allow for additional arbitrary metadata
    extra: Dict[str, Any] = {}


class BenchmarkQuery(BaseModel):
    hash: str
    serialized: str
    metadata: BenchmarkMetadata


# server side / response models


class ArtifactMetadataItem(DBItem, ArtifactMetadata):
    hash: str
    encoding: str = 'utf-8'


class BenchmarkHeader(DBItem):
    hash: str
    metadata: BenchmarkMetadata


class Benchmark(BenchmarkHeader):
    serialized: str
    # json encoding of the configured env (data sources are referenced by their hash)


class RLTuple(BaseModel):
    state: Any
    action: Any
    reward: Any
    info: dict[str, Any] = {}
    terminal: bool = False
    timeout: bool = False

    @classmethod
    def from_env_step(
        cls,
        obs: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> RLTuple:
        """
        Creates an RLTuple from environment step outputs.
        This method can be overridden to handle different types of observations and actions.

        Args:
            obs: The observation from the environment
            action: The action taken
            reward: The reward received
            terminated: Whether the episode terminated
            truncated: Whether the episode was truncated
            info: Additional information from the environment step

        Returns:
            RLTuple: A new tuple instance
        """
        # Convert numpy arrays to lists for serialization
        if isinstance(obs, np.ndarray):
            obs = obs.tolist()
        else:
            raise ValueError(f'Unsupported observation type: {type(obs)}. Expected numpy array.')
        if isinstance(action, (np.ndarray, np.integer)):
            action = action.tolist()
        else:
            raise ValueError(
                f'Unsupported action type: {type(action)}. Expected numpy array or int.'
            )

        return cls(
            state=obs,
            action=action,
            reward=reward,
            terminal=terminated,
            timeout=truncated,
            info=info,
        )


class Episode(BaseModel):
    benchmark_id: str
    metadata: dict[str, Any]
    tuples: list[RLTuple]


class EpisodeHeader(DBItem):
    benchmark_id: str
    metadata: dict[str, Any]
    n_tuples: int
    terminated: bool
    timeout: bool


class EpisodeItem(Episode, EpisodeHeader):
    @classmethod
    def create_new(cls, **data) -> EpisodeItem:
        return cls(
            id=uuid.uuid4().hex,
            created_at=datetime.datetime.now().isoformat(),
            published_in=[data.get('created_by')],
            n_tuples=len(data['tuples']),
            terminated=data['tuples'][-1]['terminal'],
            timeout=data['tuples'][-1]['timeout'],
            **data,
        )


class EpisodeMetadataCallback(ABC):
    """Abstract base class for episode metadata callbacks.

    This class defines the interface for callbacks that generate metadata for an
    episode based on the tuples. Implementations can store internal state and use it to
    generate metadata that will be stored once the end of an episode is reached.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal state of the callback."""
        pass

    @abstractmethod
    def __call__(self, tuple: RLTuple) -> dict[str, Any]:
        """Generate metadata for the episode.

        Args:
            tuple: The current tuple being processed

        Returns:
            A dictionary containing metadata for the episode
        """
        pass


######################
# Filtering
######################


class FilterType(str, Enum):
    EQ = 'EQ'  # equal to
    GEQ = 'GEQ'  # greater than or equal to
    LEQ = 'LEQ'  # less than or equal to
    GT = 'GT'  # greater than
    LT = 'LT'  # less than
    NE = 'NE'  # not for not equal to
    IN = 'IN'  # value is in array
    AND = 'AND'
    OR = 'OR'


class BaseFilter(BaseModel):
    type: FilterType = FilterType.OR
    key: Optional[str] = None
    value: Optional[Any] = None
    filters: Optional[List[BaseFilter]] = []

    @model_validator(mode='after')
    def validate_filter_attributes(self) -> Self:
        if self.type not in [FilterType.AND, FilterType.OR]:
            if self.key is None or self.value is None:
                raise ValueError(f'{self.type} filter must have key and value set')

        return self

    def __and__(self, other: BaseFilter) -> BaseFilter:
        """Combine two filters with logical AND using & operator."""
        if self.type == FilterType.AND:
            # If this filter is already an AND filter, add the other filter to its filters
            return FilterAND(filters=self.filters + [other])
        elif other.type == FilterType.AND:
            # If other filter is an AND filter, add this filter to its filters
            return FilterAND(filters=[self] + other.filters)
        else:
            # Create a new AND filter with both filters
            return FilterAND(filters=[self, other])

    def __or__(self, other: BaseFilter) -> BaseFilter:
        """Combine two filters with logical OR using | operator."""
        if self.type == FilterType.OR:
            # If this filter is already an OR filter, add the other filter to its filters
            return FilterOR(filters=self.filters + [other])
        elif other.type == FilterType.OR:
            # If other filter is an OR filter, add this filter to its filters
            return FilterOR(filters=[self] + other.filters)
        else:
            # Create a new OR filter with both filters
            return FilterOR(filters=[self, other])

    def to_params_dict(self, params: dict = {}) -> dict[str, Any]:
        """Convert the filter to a dictionary suitable for query parameters.
        Keys are added to the provided oiptional params dictionary."""
        return {**params, **self.model_dump(exclude_none=True)}

    def apply_prefix(self, prefix: str):
        """Apply a prefix to the key and all subfilters."""
        if self.key:
            self.key = f'{prefix}.{self.key}'
        if self.filters:
            for f in self.filters:
                f.apply_prefix(prefix)


class FilterEQ(BaseFilter):
    type: FilterType = FilterType.EQ


class FilterGEQ(BaseFilter):
    type: FilterType = FilterType.GEQ


class FilterLEQ(BaseFilter):
    type: FilterType = FilterType.LEQ


class FilterGT(BaseFilter):
    type: FilterType = FilterType.GT


class FilterLT(BaseFilter):
    type: FilterType = FilterType.LT


class FilterNE(BaseFilter):
    type: FilterType = FilterType.NE


class FilterIN(BaseFilter):
    type: FilterType = FilterType.IN


class FilterAND(BaseFilter):
    type: FilterType = FilterType.AND
    filters: List[BaseFilter]


class FilterOR(BaseFilter):
    type: FilterType = FilterType.OR
    filters: List[BaseFilter]

    @classmethod
    def from_list(cls, items: list[BaseModel], on_key: str, from_key: str) -> FilterOR:
        """
        Create an OR filter with one EQ filter for each item in the list.
        The key must be a field in each item.

        Args:
            items (list[BaseModel]): The list of items to construct the filter from.
            key (str): The key to filter on.

        Returns:
            BaseFilter: The filter object.
        """
        return cls(
            filters=[
                FilterEQ(type=FilterType.EQ, key=on_key, value=getattr(item, from_key))
                for item in items
            ]
        )


class EpisodesListRequest(BaseFilter):
    include_tuples: bool = False


######################
# User Management
######################


class User(BaseModel):
    username: str
    password: str
    memberships: list[Membership] = []  # List of group memberships


class Membership(BaseModel):
    group: str  # Group name
    roles: list[str]  # Roles in the group, e.g., ['member', 'admin']


class UserOut(BaseModel):
    username: str
    memberships: list[Membership] = []  # List of group memberships


class UserRole(BaseModel):
    role: str
    rights: list[RIGHT]
    description: str | None = None


class Token(BaseModel):
    token: str
    token_type: str


class UpdateUserQuery(BaseModel):
    username: str


class UserCredentials(UpdateUserQuery):
    password: str = Field(
        ..., write_only=True
    )  # Write-only field hides the password in the response


class Group(BaseModel):
    name: str
    description: str | None = None


class GroupMembership(BaseModel):
    user: str
    roles: list[str] | None = None


class GroupMembershipQuery(BaseModel):
    group_name: str
    members: list[GroupMembership]


class GroupWithMembers(BaseModel):
    name: str
    description: str | None = None
    members: list[str] = []  # List of users in the group
