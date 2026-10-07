from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def workspace_root_from_app() -> Path:
    return Path(__file__).resolve().parents[3]


def get_workspace_root() -> Path:
    return Path(os.environ.get("MARK_WORKSPACE_ROOT", workspace_root_from_app())).resolve()


def get_generator_root() -> Path:
    """Return the generator project root used for local, ignored settings."""

    configured = os.environ.get("MARK_GENERATOR_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[1]


def get_openai_base_url() -> str | None:
    value = os.environ.get("MARK_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not value or not value.strip():
        value = _read_setting_from_local_files(
            {"MARK_OPENAI_BASE_URL", "OPENAI_BASE_URL"},
        )
    return value.strip().rstrip("/") if value and value.strip() else None


def get_openai_image_model() -> str:
    value = os.environ.get("MARK_OPENAI_IMAGE_MODEL")
    if not value or not value.strip():
        value = _read_setting_from_local_files({"MARK_OPENAI_IMAGE_MODEL"})
    return (value or "gpt-image-2").strip() or "gpt-image-2"


def get_openai_image_quality() -> str:
    value = os.environ.get("MARK_OPENAI_IMAGE_QUALITY")
    if not value or not value.strip():
        value = _read_setting_from_local_files({"MARK_OPENAI_IMAGE_QUALITY"})
    return (value or "high").strip() or "high"


def get_xai_base_url() -> str:
    value = os.environ.get("MARK_XAI_BASE_URL", "https://api.x.ai/v1")
    return value.strip().rstrip("/") or "https://api.x.ai/v1"


def get_xai_image_model() -> str:
    return (
        os.environ.get("MARK_XAI_IMAGE_MODEL", "grok-imagine-image").strip()
        or "grok-imagine-image"
    )


def get_grok_secret_path() -> Path:
    configured = os.environ.get("MARK_GROK_SECRET_FILE")
    if configured:
        return Path(configured).resolve()
    return get_generator_root() / ".env.grok.local"


def get_lineart_provider_config_path() -> Path:
    configured = os.environ.get("MARK_LINEART_PROVIDER_CONFIG")
    if configured:
        return Path(configured).resolve()
    return get_generator_root() / ".lineart-provider.local.json"


def _read_setting_from_local_files(names: set[str]) -> str | None:
    """Read one non-secret setting from ignored local env files."""

    root = get_generator_root()
    for path in (root / ".env.openai.local", root / ".env.gateway.local"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in names:
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    return None


def _read_key_from_local_file(path: Path, key_name: str) -> str | None:
    """Read one key from a local ignored env file without logging its value."""

    try:
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != key_name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value or None
    except (OSError, UnicodeError):
        return None
    return None


def _read_all_keys_from_local_file(path: Path, key_names: set[str]) -> list[str]:
    """Read one or more keys from a local ignored env file without logging values."""

    found: list[str] = []
    seen: set[str] = set()
    try:
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            # Accept XAI_API_KEY, XAI_API_KEYS, XAI_API_KEY_1, GROK_API_KEY, ...
            base = name
            if name[:-1].rstrip("_").endswith("KEY") and name[-1].isdigit():
                base = name.rsplit("_", 1)[0]
            if name not in key_names and base not in key_names and not any(
                name.startswith(prefix) for prefix in key_names
            ):
                # also allow XAI_API_KEY_1 style when prefix XAI_API_KEY matches
                if not any(name == prefix or name.startswith(prefix + "_") or name == prefix + "S" for prefix in key_names):
                    continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            for part in value.replace(";", ",").split(","):
                key = part.strip()
                if not key or key in seen:
                    continue
                if any(char.isspace() for char in key) or "=" in key:
                    continue
                seen.add(key)
                found.append(key)
    except (OSError, UnicodeError):
        return found
    return found


def get_xai_api_keys() -> list[str]:
    """Return all configured Grok/xAI keys (env multi-key first, then local file)."""

    keys: list[str] = []
    seen: set[str] = set()
    for name in ("XAI_API_KEYS", "GROK_API_KEYS", "XAI_API_KEY", "GROK_API_KEY"):
        value = os.environ.get(name)
        if not value or not value.strip():
            continue
        for part in value.replace(";", ",").replace("\n", ",").split(","):
            key = part.strip()
            if not key or key in seen:
                continue
            if any(char.isspace() for char in key) or "=" in key:
                continue
            seen.add(key)
            keys.append(key)
    for key in _read_all_keys_from_local_file(
        get_grok_secret_path(),
        {"XAI_API_KEY", "XAI_API_KEYS", "GROK_API_KEY", "GROK_API_KEYS"},
    ):
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def get_xai_api_key() -> str | None:
    """Resolve one Grok/xAI key for compatibility callers."""

    keys = get_xai_api_keys()
    return keys[0] if keys else None


def get_openai_api_keys() -> list[str]:
    """Return all configured OpenAI-compatible keys."""

    keys: list[str] = []
    seen: set[str] = set()
    for name in ("OPENAI_API_KEYS", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if not value or not value.strip():
            continue
        for part in value.replace(";", ",").replace("\n", ",").split(","):
            key = part.strip()
            if not key or key in seen:
                continue
            if any(char.isspace() for char in key) or "=" in key:
                continue
            seen.add(key)
            keys.append(key)
    for key in _read_all_keys_from_local_file(
        get_generator_root() / ".env.openai.local",
        {"OPENAI_API_KEY", "OPENAI_API_KEYS"},
    ):
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def get_openai_api_key() -> str | None:
    keys = get_openai_api_keys()
    return keys[0] if keys else None


_LINEART_PROVIDER_ALIASES = {
    "openai": "openai_image",
    "gpt": "openai_image",
    "openai-image": "openai_image",
    "openai_image": "openai_image",
    "grok": "grok_image",
    "xai": "grok_image",
    "grok-image": "grok_image",
    "grok_image": "grok_image",
    "opencv-v2": "opencv_v2",
    "opencv_v2": "opencv_v2",
    "opencv-v3": "opencv_v3",
    "opencv_v3": "opencv_v3",
    "mock": "mock",
}


def canonical_lineart_provider(name: str | None) -> str:
    normalized = (name or "openai_image").strip().lower()
    try:
        return _LINEART_PROVIDER_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_LINEART_PROVIDER_ALIASES.values())))
        raise ValueError(f"Unsupported lineart provider: {name}. Allowed: {allowed}") from exc


def _read_lineart_provider_selection() -> dict[str, Any]:
    path = get_lineart_provider_config_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, UnicodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def get_active_lineart_provider() -> str:
    selection = _read_lineart_provider_selection()
    selected = selection.get("provider") or os.environ.get("MARK_LINEART_PROVIDER")
    try:
        return canonical_lineart_provider(str(selected or "openai_image"))
    except ValueError:
        return "openai_image"


def get_lineart_provider_model(provider: str) -> str | None:
    canonical = canonical_lineart_provider(provider)
    selection = _read_lineart_provider_selection()
    if selection.get("provider") == canonical and isinstance(selection.get("model"), str):
        value = str(selection["model"]).strip()
        if value:
            # Migrate the previous default transparently.  A stale ignored
            # selection file must not silently keep sending requests to the
            # old Grok model after the workbench default changes.
            if canonical == "grok_image" and value == "grok-imagine-image-quality":
                return get_xai_image_model()
            return value
    if canonical == "grok_image":
        return get_xai_image_model()
    if canonical == "openai_image":
        return get_openai_image_model()
    return None


def _provider_config(provider: str) -> dict[str, object]:
    provider = canonical_lineart_provider(provider)
    if provider == "grok_image":
        return {
            "id": provider,
            "label": "Grok Imagine",
            "configured": bool(get_xai_api_keys()),
            "key_count": len(get_xai_api_keys()),
            "base_url": get_xai_base_url(),
            "model": get_lineart_provider_model(provider),
            "quality": None,
            "using_gateway": False,
            "required_operation": "images.edits",
            "key_exposed": False,
        }
    if provider == "openai_image":
        base_url = get_openai_base_url()
        return {
            "id": provider,
            "label": "OpenAI / 兼容网关",
            "configured": bool(get_openai_api_keys()),
            "key_count": len(get_openai_api_keys()),
            "base_url": base_url or "https://api.openai.com/v1",
            "model": get_lineart_provider_model(provider),
            "quality": get_openai_image_quality(),
            "using_gateway": base_url is not None,
            "required_operation": "images.edit",
            "key_exposed": False,
        }
    return {
        "id": provider,
        "label": provider,
        "configured": True,
        "base_url": None,
        "model": None,
        "quality": None,
        "using_gateway": False,
        "required_operation": "local",
        "key_exposed": False,
    }


def get_lineart_provider_status() -> dict[str, object]:
    active = get_active_lineart_provider()
    active_config = _provider_config(active)
    openai_config = _provider_config("openai_image")
    grok_config = _provider_config("grok_image")
    return {
        # ``provider`` remains a compatibility field for the old diagnostics
        # endpoint; the canonical routing value is ``active_provider``.
        "provider": "openai_compatible" if active == "openai_image" else active,
        "active_provider": active,
        "configured": bool(active_config["configured"]),
        "base_url": active_config["base_url"],
        "using_gateway": bool(active_config["using_gateway"]),
        "model": active_config["model"],
        "quality": active_config["quality"],
        "required_operation": active_config["required_operation"],
        "key_exposed": False,
        "grok_key_configured": bool(grok_config["configured"]),
        "grok_key_count": int(grok_config.get("key_count") or 0),
        "openai_key_count": int(openai_config.get("key_count") or 0),
        "grok_secret_file": ".env.grok.local",
        "openai_secret_file": ".env.openai.local",
        "available_models": {
            "openai_image": [
                get_openai_image_model(),
                "gpt-image-2",
                "gpt-image-1",
                "gpt-image-1-mini",
            ],
            "grok_image": [
                get_xai_image_model(),
                "grok-imagine-image",
            ],
        },
        "providers": [openai_config, grok_config],
    }


def get_openai_provider_status() -> dict[str, object]:
    """Backward-compatible name used by older callers and tests."""

    return get_lineart_provider_status()


def save_lineart_provider_selection(provider: str, model: str | None = None) -> dict[str, object]:
    canonical = canonical_lineart_provider(provider)
    if canonical not in {"openai_image", "grok_image"}:
        raise ValueError("Only OpenAI and Grok image providers can be selected from the workbench.")
    if model is not None:
        model = model.strip()
        if not model or len(model) > 160 or any(char.isspace() for char in model):
            raise ValueError("Model name must be a single non-empty token.")
    payload = {"provider": canonical}
    if model:
        payload["model"] = model
    path = get_lineart_provider_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as temp:
        json.dump(payload, temp, ensure_ascii=False, indent=2)
        temp.write("\n")
        temp_path = Path(temp.name)
    temp_path.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return get_lineart_provider_status()


def save_grok_api_key(api_key: str, *, append: bool = False) -> dict[str, object]:
    """Save a Grok key to an ignored project-local file without returning it."""

    value = (api_key or "").strip()
    if not value:
        raise ValueError("Grok API key cannot be empty.")
    if any(char.isspace() for char in value) or "=" in value:
        raise ValueError("Grok API key must be one uninterrupted token.")
    path = get_grok_secret_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = get_xai_api_keys() if append else []
    keys: list[str] = []
    seen: set[str] = set()
    for key in existing + [value]:
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    with NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as temp:
        if len(keys) == 1:
            temp.write(f"XAI_API_KEY={keys[0]}\n")
        else:
            temp.write("XAI_API_KEYS=" + ",".join(keys) + "\n")
        temp_path = Path(temp.name)
    try:
        os.chmod(temp_path, 0o600)
    except OSError:
        pass
    temp_path.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    try:
        from app.services.api_key_pool import reset_key_pools

        reset_key_pools()
    except Exception:
        pass
    return {
        "saved": True,
        "configured": bool(get_xai_api_keys()),
        "key_count": len(get_xai_api_keys()),
        "secret_file": ".env.grok.local",
        "key_exposed": False,
    }


def save_openai_api_key(api_key: str, *, append: bool = False) -> dict[str, object]:
    """Save an OpenAI-compatible key to an ignored local file without returning it."""

    value = (api_key or "").strip()
    if not value:
        raise ValueError("OpenAI API key cannot be empty.")
    if any(char.isspace() for char in value) or "=" in value:
        raise ValueError("OpenAI API key must be one uninterrupted token.")
    path = get_generator_root() / ".env.openai.local"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = get_openai_api_keys() if append else []
    keys = []
    seen = set()
    for key in existing + [value]:
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    with NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as temp:
        if len(keys) == 1:
            temp.write(f"OPENAI_API_KEY={keys[0]}\n")
        else:
            temp.write("OPENAI_API_KEYS=" + ",".join(keys) + "\n")
        temp_path = Path(temp.name)
    try:
        os.chmod(temp_path, 0o600)
    except OSError:
        pass
    temp_path.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    try:
        from app.services.api_key_pool import reset_key_pools

        reset_key_pools()
    except Exception:
        pass
    return {
        "saved": True,
        "configured": bool(get_openai_api_keys()),
        "key_count": len(get_openai_api_keys()),
        "secret_file": ".env.openai.local",
        "key_exposed": False,
    }
