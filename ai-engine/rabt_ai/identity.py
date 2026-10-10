"""Registered identities, their contacts, permissions and voice profiles.

Identity is asserted by the authenticated integration backend (``account_id``) and checked
here against the registry. Nothing in a chat message can change who the user is.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rabt_ai.errors import RabtError
from rabt_shared import AccountId, ErrorCode, Role, ToolName


class _ContactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: AccountId
    display_name: str = Field(min_length=1, max_length=64)
    relationship: str = Field(min_length=1, max_length=32)


class _AccountConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=64)
    role: Role
    voice_profile: str
    allowed_tools: list[ToolName]
    about: str = Field(default="", max_length=300)
    preferences: dict[str, str | int | float | bool] = Field(default_factory=dict)
    contacts: list[_ContactConfig] = Field(default_factory=list)


class _VoiceProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: AccountId
    engine: str
    voice: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    lang: str = "en-us"
    description: str = ""


class _RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accounts: dict[AccountId, _AccountConfig]
    voice_profiles: dict[str, _VoiceProfileConfig]


@dataclass(frozen=True)
class Contact:
    account_id: str
    display_name: str
    relationship: str


@dataclass(frozen=True)
class VoiceProfile:
    profile_id: str
    account_id: str
    engine: str
    voice: str
    speed: float
    lang: str
    description: str


@dataclass(frozen=True)
class Identity:
    account_id: str
    display_name: str
    role: Role
    voice_profile: str
    allowed_tools: frozenset[ToolName]
    about: str
    contacts: tuple[Contact, ...]
    preferences: Mapping[str, Any] = field(default_factory=dict)

    def find_contact(self, ref: str) -> Contact | None:
        """Match a contact by account_id or display name (case-insensitive)."""
        key = ref.strip().lower().lstrip("@")
        for contact in self.contacts:
            if key in (contact.account_id, contact.display_name.lower()):
                return contact
        return None


class IdentityRegistry:
    def __init__(self, identities: Mapping[str, Identity], voice_profiles: Mapping[str, VoiceProfile]):
        self._identities = dict(identities)
        self._voice_profiles = dict(voice_profiles)

    @classmethod
    def load(cls, path: Path) -> IdentityRegistry:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        config = _RegistryConfig.model_validate(raw)

        identities: dict[str, Identity] = {}
        for account_id, acc in config.accounts.items():
            identities[account_id] = Identity(
                account_id=account_id,
                display_name=acc.display_name,
                role=acc.role,
                voice_profile=acc.voice_profile,
                allowed_tools=frozenset(acc.allowed_tools),
                about=acc.about,
                contacts=tuple(Contact(**c.model_dump()) for c in acc.contacts),
                preferences=dict(acc.preferences),
            )
        profiles = {
            pid: VoiceProfile(profile_id=pid, **vp.model_dump()) for pid, vp in config.voice_profiles.items()
        }

        for ident in identities.values():
            for contact in ident.contacts:
                if contact.account_id not in identities:
                    raise ValueError(f"{ident.account_id}: contact {contact.account_id!r} is not registered")
                if contact.account_id == ident.account_id:
                    raise ValueError(f"{ident.account_id}: an account cannot list itself as a contact")
            profile = profiles.get(ident.voice_profile)
            if profile is None or profile.account_id != ident.account_id:
                raise ValueError(f"{ident.account_id}: voice profile {ident.voice_profile!r} is not theirs")
        for pid, profile in profiles.items():
            if profile.account_id not in identities:
                raise ValueError(f"voice profile {pid!r} belongs to unknown account {profile.account_id!r}")
        return cls(identities, profiles)

    @property
    def identities(self) -> Mapping[str, Identity]:
        return self._identities

    @property
    def voice_profiles(self) -> Mapping[str, VoiceProfile]:
        return self._voice_profiles

    def require(self, account_id: str, role: Role | None = None) -> Identity:
        identity = self._identities.get(account_id)
        if identity is None:
            raise RabtError(ErrorCode.UNKNOWN_ACCOUNT, f"account {account_id!r} is not registered")
        if role is not None and role != identity.role:
            raise RabtError(
                ErrorCode.ROLE_MISMATCH,
                f"account {account_id!r} is registered as {identity.role.value!r}, not {role.value!r}",
            )
        return identity

    def voice_profile_for(self, account_id: str, profile_id: str | None) -> VoiceProfile:
        """Return the requested profile only if it belongs to ``account_id`` (default: their own)."""
        identity = self.require(account_id)
        wanted = profile_id or identity.voice_profile
        profile = self._voice_profiles.get(wanted)
        if profile is None or profile.account_id != account_id:
            raise RabtError(
                ErrorCode.VOICE_PROFILE_FORBIDDEN,
                f"voice profile {wanted!r} is not authorized for account {account_id!r}",
            )
        return profile
