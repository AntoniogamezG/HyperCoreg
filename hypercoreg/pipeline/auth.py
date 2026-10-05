"""CDSE authentication and credential helpers.

This module is the migration target for auth/session logic previously hosted
in the monolithic runtime.
"""

from __future__ import annotations

import base64
import getpass
import json
import logging
import os
import sys
import threading
import time
from time import sleep
from typing import Any, Callable, Dict, Optional, Tuple

import requests
from requests.auth import AuthBase

from hypercoreg.utils import CDSEAuthenticationError

logger = logging.getLogger("COREG_PROCESSING")

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
HTTP_TIMEOUT_S = 30
HTTP_AUTH_RETRY_ATTEMPTS = 2
HTTP_RETRY_BACKOFF_BASE_S = 1.0
HTTP_RETRY_BACKOFF_FACTOR = 2.0
HTTP_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
TOKEN_REFRESH_MARGIN_S = 300.0

PromptUserpassFn = Callable[[], Tuple[str, str, Optional[str]]]
TokenFactoryFn = Callable[[], str]
_CDSE_CREDENTIAL_CACHE_LOCK = threading.Lock()
# NOTE: the TOTP *code* is never cached (it is single-use and expires within
# ~30 s). Only a boolean "totp_required" flag is remembered so that later
# reconnects know a fresh code must be obtained instead of replaying one.
_CDSE_CREDENTIAL_CACHE: Dict[str, Any] = {
    "username": None,
    "password": None,
    "totp_required": False,
}


def _fmt_issue(scope: str, message: str) -> str:
    return f"[{scope}] {message}"


def _retry_delay_seconds(retry_index: int) -> float:
    idx = max(1, int(retry_index))
    return float(HTTP_RETRY_BACKOFF_BASE_S * (HTTP_RETRY_BACKOFF_FACTOR ** (idx - 1)))


def _is_retryable_http_status(status_code: int) -> bool:
    return int(status_code) in HTTP_RETRYABLE_STATUS_CODES


def _cache_cdse_public_credentials(
    username: Optional[str],
    password: Optional[str],
    totp: Optional[str] = None,
) -> None:
    """Cache username/password for reconnects.

    The ``totp`` argument is accepted for backwards compatibility but the code
    itself is never stored; only the fact that the account uses 2FA is kept.
    """
    user = str(username or "").strip()
    pwd = str(password or "")
    totp_required = bool(str(totp).strip()) if totp is not None else False
    with _CDSE_CREDENTIAL_CACHE_LOCK:
        _CDSE_CREDENTIAL_CACHE["username"] = user or None
        _CDSE_CREDENTIAL_CACHE["password"] = pwd if pwd else None
        _CDSE_CREDENTIAL_CACHE["totp_required"] = totp_required


def _get_cached_cdse_public_credentials() -> Optional[Tuple[str, str, Optional[str]]]:
    with _CDSE_CREDENTIAL_CACHE_LOCK:
        user = str(_CDSE_CREDENTIAL_CACHE.get("username") or "").strip()
        pwd = _CDSE_CREDENTIAL_CACHE.get("password") or ""
    if user and pwd:
        # The third element (TOTP code) is always None: codes are never cached.
        return user, str(pwd), None
    return None


def _cached_cdse_account_requires_totp() -> bool:
    with _CDSE_CREDENTIAL_CACHE_LOCK:
        return bool(_CDSE_CREDENTIAL_CACHE.get("totp_required"))


def _clear_cached_cdse_public_credentials() -> None:
    with _CDSE_CREDENTIAL_CACHE_LOCK:
        _CDSE_CREDENTIAL_CACHE["username"] = None
        _CDSE_CREDENTIAL_CACHE["password"] = None
        _CDSE_CREDENTIAL_CACHE["totp_required"] = False


def _resolve_cdse_credentials_file_path() -> str:
    override = (os.environ.get("HYPERCOREG_CREDENTIALS_FILE") or "").strip()
    if override:
        return os.path.normpath(os.path.expanduser(override))
    return os.path.normpath(os.path.expanduser(os.path.join("~", ".hypercoreg", "credentials.json")))


def _read_cdse_credentials_file() -> Optional[Dict[str, Any]]:
    path = _resolve_cdse_credentials_file_path()
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        logger.warning(_fmt_issue("AUTH", f"Failed to read CDSE credentials file ({path}): {exc}"))
        return None

    if not isinstance(payload, dict):
        logger.warning(_fmt_issue("AUTH", f"CDSE credentials file must contain a JSON object: {path}"))
        return None

    nested = payload.get("cdse")
    if isinstance(nested, dict):
        return nested
    return payload


def _extract_cdse_credential_value(source: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _resolve_configured_public_credentials(
    *,
    include_env: bool = True,
    include_file: bool = True,
    file_creds: Optional[Dict[str, Any]] = None,
    creds_path: Optional[str] = None,
) -> Optional[Tuple[str, str, Optional[str], str]]:
    if include_env:
        username = (os.environ.get("CDSE_USERNAME") or "").strip()
        password = os.environ.get("CDSE_PASSWORD") or ""
        if username and password:
            totp = (os.environ.get("CDSE_TOTP") or "").strip() or None
            return username, password, totp, "environment"
        if username or password:
            logger.warning(
                _fmt_issue(
                    "AUTH",
                    "Incomplete CDSE username/password in environment; set both "
                    "CDSE_USERNAME and CDSE_PASSWORD.",
                )
            )

    if not include_file:
        return None

    if file_creds is None:
        file_creds = _read_cdse_credentials_file()
    if not isinstance(file_creds, dict):
        return None
    if creds_path is None:
        creds_path = _resolve_cdse_credentials_file_path()

    file_username = _extract_cdse_credential_value(file_creds, "username", "CDSE_USERNAME")
    file_password = _extract_cdse_credential_value(file_creds, "password", "CDSE_PASSWORD")
    if file_username and file_password:
        file_totp = _extract_cdse_credential_value(file_creds, "totp", "CDSE_TOTP") or None
        return file_username, file_password, file_totp, f"credentials file: {creds_path}"
    if file_username or file_password:
        logger.warning(
            _fmt_issue(
                "AUTH",
                "Incomplete CDSE username/password in credentials file; expected "
                "both username and password.",
            )
        )
    return None


class _BearerAuth(AuthBase):
    def __init__(self, token: str):
        self.token = token

    def __call__(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        return request


def _decode_jwt_exp(access_token: str) -> Optional[float]:
    parts = str(access_token or "").split(".")
    if len(parts) < 2:
        return None
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode((payload_b64 + padding).encode("ascii")))
    except Exception:
        return None
    exp = payload.get("exp")
    try:
        return float(exp)
    except (TypeError, ValueError):
        return None


class _RefreshableBearerAuth(AuthBase):
    def __init__(
        self,
        token_factory: TokenFactoryFn,
        *,
        source_name: str,
        refresh_margin_s: float = TOKEN_REFRESH_MARGIN_S,
    ):
        self._token_factory = token_factory
        self._source_name = str(source_name or "CDSE")
        self._refresh_margin_s = max(0.0, float(refresh_margin_s))
        self._lock = threading.Lock()
        self.token: Optional[str] = None
        self.expires_at: Optional[float] = None

    def _refresh_locked(self) -> str:
        token = str(self._token_factory() or "").strip()
        if not token:
            raise CDSEAuthenticationError(
                _fmt_issue("AUTH", f"{self._source_name} token refresh returned no token.")
            )
        self.token = token
        self.expires_at = _decode_jwt_exp(token)
        if self.expires_at is None:
            factory_exp = getattr(self._token_factory, "access_expires_at", None)
            if isinstance(factory_exp, (int, float)):
                self.expires_at = float(factory_exp)
        if self.expires_at is not None:
            logger.debug(
                "Refreshed CDSE token from %s; expires at %s UTC.",
                self._source_name,
                datetime_from_timestamp_utc(self.expires_at),
            )
        else:
            logger.debug(
                "Refreshed CDSE token from %s; token expiry is not available.",
                self._source_name,
            )
        return token

    def _needs_refresh_locked(self) -> bool:
        if not self.token:
            return True
        if self.expires_at is None:
            return False
        return (self.expires_at - time.time()) <= self._refresh_margin_s

    def force_refresh(self) -> str:
        with self._lock:
            return self._refresh_locked()

    def get_token(self) -> str:
        with self._lock:
            if self._needs_refresh_locked():
                return self._refresh_locked()
            return str(self.token)

    def __call__(self, request):
        request.headers["Authorization"] = f"Bearer {self.get_token()}"
        return request


def datetime_from_timestamp_utc(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _create_cdse_bearer_session(access_token: str) -> requests.Session:
    token = (access_token or "").strip()
    if not token:
        raise ValueError(_fmt_issue("AUTH", "Empty CDSE bearer token provided."))
    session = requests.Session()
    session.auth = _BearerAuth(token)
    session.headers.update({"Accept": "application/json"})
    return session


def _create_cdse_refreshable_bearer_session(
    token_factory: TokenFactoryFn,
    *,
    source_name: str,
) -> requests.Session:
    session = requests.Session()
    refreshable_auth = _RefreshableBearerAuth(token_factory, source_name=source_name)
    session.auth = refreshable_auth
    session.headers.update({"Accept": "application/json"})
    refreshable_auth.force_refresh()
    return session


def _force_refresh_cdse_session(session: Any) -> bool:
    auth_obj = getattr(session, "auth", None)
    refresh = getattr(auth_obj, "force_refresh", None)
    if not callable(refresh):
        return False
    refresh()
    return True


def _request_cdse_access_token(payload: Dict[str, str], flow_name: str) -> str:
    """Return only the access token string (backwards-compatible API)."""
    return str(_request_cdse_token_response(payload, flow_name)["access_token"])


def _request_cdse_token_response(payload: Dict[str, str], flow_name: str) -> Dict[str, Any]:
    """POST ``payload`` to the CDSE token endpoint and return the JSON response.

    The returned dict always contains a non-empty ``access_token`` and may
    contain ``expires_in``, ``refresh_token`` and ``refresh_expires_in``.
    Non-retryable HTTP errors (e.g. 401 wrong password) are raised immediately
    without resending credentials.
    """
    attempts = max(1, int(HTTP_AUTH_RETRY_ATTEMPTS))
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(TOKEN_URL, data=payload, timeout=HTTP_TIMEOUT_S)
            if response.status_code >= 400:
                if _is_retryable_http_status(response.status_code) and attempt < attempts:
                    delay_s = _retry_delay_seconds(attempt)
                    logger.warning(
                        _fmt_issue(
                            "AUTH",
                            f"{flow_name} token retryable HTTP {response.status_code} "
                            f"(attempt {attempt}/{attempts}); retrying in {delay_s:.1f}s.",
                        )
                    )
                    sleep(delay_s)
                    continue
                msg = response.text[:500]
                raise CDSEAuthenticationError(
                    _fmt_issue(
                        "AUTH",
                        f"{flow_name} token generation failed ({response.status_code}): {msg}",
                    )
                )

            body = response.json()
            token = body.get("access_token") if isinstance(body, dict) else None
            if not token:
                raise CDSEAuthenticationError(
                    _fmt_issue("AUTH", f"{flow_name} token response missing access_token.")
                )
            return dict(body)
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts:
                delay_s = _retry_delay_seconds(attempt)
                logger.warning(
                    _fmt_issue(
                        "AUTH",
                        f"{flow_name} token request failed (attempt {attempt}/{attempts}): {exc}; "
                        f"retrying in {delay_s:.1f}s.",
                    )
                )
                sleep(delay_s)
                continue
            raise CDSEAuthenticationError(
                _fmt_issue("AUTH", f"{flow_name} token request failed: {exc}")
            ) from exc
        except CDSEAuthenticationError:
            # A non-retryable rejection (e.g. wrong password); do not resend credentials.
            raise
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                delay_s = _retry_delay_seconds(attempt)
                logger.warning(
                    _fmt_issue(
                        "AUTH",
                        f"{flow_name} token generation failed (attempt {attempt}/{attempts}): {exc}; "
                        f"retrying in {delay_s:.1f}s.",
                    )
                )
                sleep(delay_s)
                continue
            if isinstance(exc, CDSEAuthenticationError):
                raise
            raise CDSEAuthenticationError(
                _fmt_issue("AUTH", f"{flow_name} token generation failed: {exc}")
            ) from exc

    if last_error is not None:
        raise CDSEAuthenticationError(
            _fmt_issue("AUTH", f"{flow_name} token generation failed: {last_error}")
        )
    raise CDSEAuthenticationError(
        _fmt_issue("AUTH", f"{flow_name} token generation failed unexpectedly.")
    )


def _generate_cdse_public_access_token(
    username: str,
    password: str,
    totp: Optional[str] = None,
) -> str:
    payload = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    if totp:
        payload["totp"] = str(totp).strip()
    return _request_cdse_access_token(payload, flow_name="cdse-public")


def _generate_cdse_client_access_token(client_id: str, client_secret: str) -> str:
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    return _request_cdse_access_token(payload, flow_name="client-credentials")


TotpProviderFn = Callable[[], Optional[str]]


def _expiry_from_seconds(value: Any, now: float) -> Optional[float]:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        # Keycloak reports 0 for "no expiry"/offline tokens; treat as unknown.
        return None
    return now + seconds


class _CDSEPublicTokenFactory:
    """Token factory for the ``cdse-public`` password flow.

    * First call: ``grant_type=password`` (+ ``totp`` if supplied). The TOTP
      code is consumed and dropped immediately; it is never replayed.
    * Later calls: ``grant_type=refresh_token`` while the refresh token is
      valid (no password/TOTP sent).
    * If the refresh fails or the refresh token has expired: fall back to the
      password grant. For 2FA accounts a *fresh* code is requested from
      ``totp_provider``; if none is available a clear
      :class:`CDSEAuthenticationError` is raised.
    """

    client_id = "cdse-public"
    flow_name = "cdse-public"

    def __init__(
        self,
        username: str,
        password: str,
        totp: Optional[str] = None,
        *,
        totp_provider: Optional[TotpProviderFn] = None,
    ):
        self._username = username
        self._password = password
        first_totp = str(totp).strip() if totp is not None else ""
        self._pending_totp: Optional[str] = first_totp or None
        self.totp_required = bool(first_totp)
        self._totp_provider = totp_provider
        self._refresh_token: Optional[str] = None
        self._refresh_expires_at: Optional[float] = None
        self.access_expires_at: Optional[float] = None
        self._initial_grant_done = False
        self._lock = threading.Lock()

    def __repr__(self) -> str:  # never expose secrets
        return f"<_CDSEPublicTokenFactory user={self._username!r} totp_required={self.totp_required}>"

    def _store_response(self, body: Dict[str, Any]) -> str:
        now = time.time()
        self.access_expires_at = _expiry_from_seconds(body.get("expires_in"), now)
        refresh_token = str(body.get("refresh_token") or "").strip()
        if refresh_token:
            self._refresh_token = refresh_token
            self._refresh_expires_at = _expiry_from_seconds(body.get("refresh_expires_in"), now)
        else:
            self._refresh_token = None
            self._refresh_expires_at = None
        return str(body["access_token"])

    def _refresh_token_usable(self) -> bool:
        if not self._refresh_token:
            return False
        if self._refresh_expires_at is None:
            return True
        # Leave a small safety margin so the request does not race expiry.
        return (self._refresh_expires_at - time.time()) > 5.0

    def _refresh_grant(self) -> str:
        payload = {
            "client_id": self.client_id,
            "grant_type": "refresh_token",
            "refresh_token": str(self._refresh_token),
        }
        body = _request_cdse_token_response(payload, flow_name=f"{self.flow_name} refresh")
        return self._store_response(body)

    def _obtain_fresh_totp(self) -> str:
        code: Optional[str] = None
        if self._totp_provider is not None:
            code = str(self._totp_provider() or "").strip() or None
        if not code:
            raise CDSEAuthenticationError(
                _fmt_issue(
                    "AUTH",
                    "CDSE session expired and the account requires a TOTP (2FA) code, "
                    "but no fresh code is available. TOTP codes are single-use and are "
                    "never replayed; log in again with a new code.",
                )
            )
        return code

    def _password_grant(self, totp: Optional[str]) -> str:
        payload = {
            "client_id": self.client_id,
            "grant_type": "password",
            "username": self._username,
            "password": self._password,
        }
        if totp:
            payload["totp"] = totp
        body = _request_cdse_token_response(payload, flow_name=self.flow_name)
        return self._store_response(body)

    def __call__(self) -> str:
        with self._lock:
            if not self._initial_grant_done:
                # Consume the caller-supplied TOTP exactly once, even on failure.
                totp, self._pending_totp = self._pending_totp, None
                self._initial_grant_done = True
                return self._password_grant(totp)

            if self._refresh_token_usable():
                try:
                    return self._refresh_grant()
                except CDSEAuthenticationError as exc:
                    logger.info(
                        _fmt_issue(
                            "AUTH",
                            f"CDSE refresh-token grant failed ({exc}); falling back to password grant.",
                        )
                    )
                    self._refresh_token = None
                    self._refresh_expires_at = None

            totp = self._obtain_fresh_totp() if self.totp_required else None
            return self._password_grant(totp)


def _create_cdse_public_session(
    username: str,
    password: str,
    totp: Optional[str] = None,
    *,
    totp_provider: Optional[TotpProviderFn] = None,
) -> requests.Session:
    factory = _CDSEPublicTokenFactory(
        username,
        password,
        totp,
        totp_provider=totp_provider,
    )
    return _create_cdse_refreshable_bearer_session(factory, source_name="cdse-public")


def _create_cdse_client_session(client_id: str, client_secret: str) -> requests.Session:
    return _create_cdse_refreshable_bearer_session(
        lambda: _generate_cdse_client_access_token(
            client_id=client_id,
            client_secret=client_secret,
        ),
        source_name="client-credentials",
    )


def _create_cdse_session_from_environment() -> Optional[requests.Session]:
    public_creds = _resolve_configured_public_credentials(include_env=True, include_file=False)
    if public_creds is not None:
        username, password, totp, source = public_creds
        logger.info("Using CDSE username/password from %s.", source)
        sess = _create_cdse_public_session(username=username, password=password, totp=totp)
        _cache_cdse_public_credentials(username=username, password=password, totp=totp)
        return sess

    client_id = (os.environ.get("CDSE_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("CDSE_CLIENT_SECRET") or "").strip()
    if client_id and client_secret:
        logger.info("Using CDSE client credentials from environment.")
        return _create_cdse_client_session(client_id=client_id, client_secret=client_secret)
    if client_id or client_secret:
        logger.warning(
            _fmt_issue(
                "AUTH",
                "Incomplete CDSE client credentials in environment; set both "
                "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET.",
            )
        )

    access_token = (os.environ.get("CDSE_ACCESS_TOKEN") or "").strip()
    if access_token:
        logger.info("Using non-renewable CDSE bearer token from environment (CDSE_ACCESS_TOKEN).")
        return _create_cdse_bearer_session(access_token)

    file_creds = _read_cdse_credentials_file()
    if isinstance(file_creds, dict):
        creds_path = _resolve_cdse_credentials_file_path()

        public_file_creds = _resolve_configured_public_credentials(
            include_env=False,
            include_file=True,
            file_creds=file_creds,
            creds_path=creds_path,
        )
        if public_file_creds is not None:
            file_username, file_password, file_totp, source = public_file_creds
            logger.info("Using CDSE username/password from %s.", source)
            sess = _create_cdse_public_session(
                username=file_username,
                password=file_password,
                totp=file_totp,
            )
            _cache_cdse_public_credentials(
                username=file_username,
                password=file_password,
                totp=file_totp,
            )
            return sess

        file_client_id = _extract_cdse_credential_value(
            file_creds,
            "client_id",
            "CDSE_CLIENT_ID",
        )
        file_client_secret = _extract_cdse_credential_value(
            file_creds,
            "client_secret",
            "CDSE_CLIENT_SECRET",
        )
        if file_client_id and file_client_secret:
            logger.info("Using CDSE client credentials from credentials file: %s", creds_path)
            return _create_cdse_client_session(
                client_id=file_client_id,
                client_secret=file_client_secret,
            )
        if file_client_id or file_client_secret:
            logger.warning(
                _fmt_issue(
                    "AUTH",
                    "Incomplete CDSE client credentials in credentials file; expected "
                    "both client_id and client_secret.",
                )
            )

        file_access_token = _extract_cdse_credential_value(
            file_creds,
            "access_token",
            "CDSE_ACCESS_TOKEN",
            "cdse_access_token",
            "token",
        )
        if file_access_token:
            logger.info(
                "Using non-renewable CDSE bearer token from credentials file: %s",
                creds_path,
            )
            return _create_cdse_bearer_session(file_access_token)

    return None


def _prompt_cdse_userpass_cli(max_prompt_attempts: int = 2) -> Tuple[str, str, Optional[str]]:
    stdin = getattr(sys, "stdin", None)
    if stdin is None or not callable(getattr(stdin, "isatty", None)) or not bool(stdin.isatty()):
        raise CDSEAuthenticationError(
            _fmt_issue(
                "AUTH",
                "No interactive terminal available for CDSE username/password prompt. "
                "Set CDSE_USERNAME/CDSE_PASSWORD (or CDSE_CLIENT_ID/CDSE_CLIENT_SECRET), "
                "or define them in ~/.hypercoreg/credentials.json.",
            )
        )

    attempts = max(1, int(max_prompt_attempts))
    for attempt in range(1, attempts + 1):
        try:
            user = input("CDSE username: ").strip()
            pwd = getpass.getpass("CDSE password: ")
            totp = input("CDSE TOTP (optional): ").strip() or None
        except EOFError as exc:
            raise CDSEAuthenticationError(
                _fmt_issue(
                    "AUTH",
                    "CDSE credential prompt reached EOF (non-interactive input stream). "
                    "Set CDSE_USERNAME/CDSE_PASSWORD (or CDSE_CLIENT_ID/CDSE_CLIENT_SECRET), "
                    "or define them in ~/.hypercoreg/credentials.json.",
                )
            ) from exc
        except KeyboardInterrupt as exc:
            raise CDSEAuthenticationError(
                _fmt_issue("AUTH", "CDSE login cancelled by user.")
            ) from exc

        if user and pwd:
            return user, pwd, totp

        logger.warning(
            _fmt_issue(
                "AUTH",
                f"Username and password are required (attempt {attempt}/{attempts}).",
            )
        )

    raise CDSEAuthenticationError(_fmt_issue("AUTH", "CDSE username/password were not provided."))


def _prompt_cdse_userpass_gui() -> Tuple[str, str, Optional[str]]:
    import tkinter as tk
    from tkinter import messagebox

    parent = tk._default_root
    owns_root = False
    if parent is None:
        parent = tk.Tk()
        parent.withdraw()
        owns_root = True

    dialog = tk.Toplevel(parent)
    dialog.title("CDSE Access Token Login")
    dialog.geometry("480x260")
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - 240
    y = (dialog.winfo_screenheight() // 2) - 130
    dialog.geometry(f"480x260+{x}+{y}")

    result = {"ok": False, "username": "", "password": "", "totp": ""}

    tk.Label(dialog, text="Copernicus Data Space Login", font=("Arial", 12, "bold")).pack(pady=(12, 8))
    tk.Label(dialog, text="Used to generate a temporary cdse-public access token.", fg="gray").pack(pady=(0, 10))

    frame = tk.Frame(dialog)
    frame.pack(fill="x", padx=20)

    tk.Label(frame, text="Username:", width=14, anchor="w").grid(row=0, column=0, pady=5, sticky="w")
    user_entry = tk.Entry(frame, width=38)
    user_entry.grid(row=0, column=1, pady=5, sticky="w")

    tk.Label(frame, text="Password:", width=14, anchor="w").grid(row=1, column=0, pady=5, sticky="w")
    pass_entry = tk.Entry(frame, width=38, show="*")
    pass_entry.grid(row=1, column=1, pady=5, sticky="w")

    tk.Label(frame, text="TOTP (optional):", width=14, anchor="w").grid(row=2, column=0, pady=5, sticky="w")
    totp_entry = tk.Entry(frame, width=20)
    totp_entry.grid(row=2, column=1, pady=5, sticky="w")

    btns = tk.Frame(dialog)
    btns.pack(pady=16)

    def on_ok() -> None:
        user = user_entry.get().strip()
        pwd = pass_entry.get()
        if not user or not pwd:
            messagebox.showerror("Missing credentials", "Username and password are required.", parent=dialog)
            return
        result["ok"] = True
        result["username"] = user
        result["password"] = pwd
        result["totp"] = totp_entry.get().strip()
        dialog.destroy()

    def on_cancel() -> None:
        dialog.destroy()

    tk.Button(btns, text="OK", width=12, command=on_ok).pack(side="left", padx=8)
    tk.Button(btns, text="Cancel", width=12, command=on_cancel).pack(side="left", padx=8)
    dialog.bind("<Return>", lambda _evt: on_ok())
    pass_entry.bind("<Return>", lambda _evt: on_ok())
    user_entry.focus_set()

    parent.wait_window(dialog)
    if owns_root:
        try:
            parent.destroy()
        except Exception:
            pass

    if not result["ok"]:
        raise CDSEAuthenticationError(_fmt_issue("AUTH", "CDSE login cancelled by user."))

    return result["username"], result["password"], (result["totp"] or None)


def _request_cdse_userpass(
    allow_gui_prompt: bool,
    prompt_userpass_fn: Optional[PromptUserpassFn] = None,
) -> Tuple[str, str, Optional[str]]:
    if allow_gui_prompt:
        if prompt_userpass_fn is not None:
            return prompt_userpass_fn()
        if threading.current_thread() is not threading.main_thread():
            raise CDSEAuthenticationError(
                _fmt_issue(
                    "AUTH",
                    "GUI credential prompt requested from a worker thread. "
                    "Provide a main-thread prompt callback.",
                )
            )
        return _prompt_cdse_userpass_gui()

    return _prompt_cdse_userpass_cli()


def _make_interactive_totp_provider(
    username: str,
    allow_gui_prompt: bool,
    prompt_userpass_fn: Optional[PromptUserpassFn],
) -> TotpProviderFn:
    """Return a provider that asks the user for a *fresh* TOTP code.

    Used only by sessions created from an interactive prompt. Failures (no
    terminal, cancelled, different user) yield ``None`` so the token factory
    raises its explicit "fresh TOTP required" error.
    """

    def _provider() -> Optional[str]:
        try:
            logger.warning(
                _fmt_issue("AUTH", "CDSE session expired; a fresh TOTP code is required to continue.")
            )
            user, _pwd, totp = _request_cdse_userpass(
                allow_gui_prompt=allow_gui_prompt,
                prompt_userpass_fn=prompt_userpass_fn,
            )
        except Exception:
            return None
        if str(user or "").strip() != str(username or "").strip():
            return None
        return str(totp or "").strip() or None

    return _provider


def _create_public_session_with_retry(
    allow_gui_prompt: bool = False,
    max_prompt_attempts: int = 2,
    prompt_userpass_fn: Optional[PromptUserpassFn] = None,
) -> requests.Session:
    last_error: Optional[Exception] = None
    # TOTP codes are single-use: if this process already logged in with one,
    # any configured/cached code is stale and must not be replayed.
    totp_required = _cached_cdse_account_requires_totp()
    cached_userpass = _get_cached_cdse_public_credentials()
    if cached_userpass is not None and totp_required:
        logger.info(
            "Cached CDSE account uses 2FA; a fresh TOTP code is required for session refresh."
        )
    elif cached_userpass is not None:
        user, pwd, _ = cached_userpass
        try:
            logger.info("Reusing cached CDSE username/password for session refresh.")
            return _create_cdse_public_session(user, pwd)
        except Exception as exc:
            logger.warning(
                _fmt_issue(
                    "AUTH",
                    f"Cached CDSE credentials failed ({exc}); requesting credentials again.",
                )
            )
            _clear_cached_cdse_public_credentials()
            last_error = exc

    configured_userpass = _resolve_configured_public_credentials(
        include_env=True,
        include_file=True,
    )
    if configured_userpass is not None and totp_required:
        logger.info(
            "Skipping configured CDSE credentials: the account uses 2FA and configured "
            "TOTP codes are never replayed."
        )
    elif configured_userpass is not None:
        user, pwd, totp, source = configured_userpass
        try:
            logger.info(
                "Using configured CDSE username/password from %s for session refresh.",
                source,
            )
            sess = _create_cdse_public_session(user, pwd, totp=totp)
            _cache_cdse_public_credentials(
                username=user,
                password=pwd,
                totp=totp,
            )
            return sess
        except Exception as exc:
            logger.warning(
                _fmt_issue(
                    "AUTH",
                    f"Configured CDSE username/password from {source} failed ({exc}); "
                    "requesting credentials again.",
                )
            )
            _clear_cached_cdse_public_credentials()
            last_error = exc

    attempts = max(1, int(max_prompt_attempts))
    for attempt in range(1, attempts + 1):
        try:
            user, pwd, totp = _request_cdse_userpass(
                allow_gui_prompt=allow_gui_prompt,
                prompt_userpass_fn=prompt_userpass_fn,
            )
            sess = _create_cdse_public_session(
                user,
                pwd,
                totp=totp,
                totp_provider=(
                    _make_interactive_totp_provider(user, allow_gui_prompt, prompt_userpass_fn)
                    if totp
                    else None
                ),
            )
            _cache_cdse_public_credentials(username=user, password=pwd, totp=totp)
            return sess
        except CDSEAuthenticationError as exc:
            lower_msg = str(exc).lower()
            if "cancelled" in lower_msg or "no interactive terminal" in lower_msg:
                if totp_required:
                    raise CDSEAuthenticationError(
                        _fmt_issue(
                            "AUTH",
                            "CDSE session expired and the account requires a fresh TOTP "
                            f"(2FA) code, which could not be obtained: {exc}",
                        )
                    ) from exc
                raise
            last_error = exc
        except RuntimeError as exc:
            if "cancelled" in str(exc).lower():
                raise CDSEAuthenticationError(str(exc)) from exc
            last_error = exc
        except Exception as exc:
            last_error = exc

        if attempt < attempts:
            logger.warning(
                _fmt_issue(
                    "AUTH",
                    f"CDSE login failed (attempt {attempt}/{attempts}): {last_error}",
                )
            )

    raise CDSEAuthenticationError(
        _fmt_issue("AUTH", f"CDSE username/password authentication failed: {last_error}")
    )


def _create_cdse_session_with_retry(
    allow_gui_prompt: bool = False,
    prompt_userpass_fn: Optional[PromptUserpassFn] = None,
) -> requests.Session:
    env_session = _create_cdse_session_from_environment()
    if env_session is not None:
        return env_session

    return _create_public_session_with_retry(
        allow_gui_prompt=allow_gui_prompt,
        max_prompt_attempts=2,
        prompt_userpass_fn=prompt_userpass_fn,
    )


__all__ = [
    "CDSEAuthenticationError",
    "_BearerAuth",
    "_RefreshableBearerAuth",
    "_cache_cdse_public_credentials",
    "_clear_cached_cdse_public_credentials",
    "_create_cdse_bearer_session",
    "_create_cdse_client_session",
    "_create_cdse_public_session",
    "_create_cdse_refreshable_bearer_session",
    "_create_cdse_session_from_environment",
    "_create_cdse_session_with_retry",
    "_create_public_session_with_retry",
    "_decode_jwt_exp",
    "_extract_cdse_credential_value",
    "_force_refresh_cdse_session",
    "_generate_cdse_client_access_token",
    "_generate_cdse_public_access_token",
    "_get_cached_cdse_public_credentials",
    "_prompt_cdse_userpass_cli",
    "_prompt_cdse_userpass_gui",
    "_read_cdse_credentials_file",
    "_request_cdse_access_token",
    "_request_cdse_token_response",
    "_request_cdse_userpass",
    "_resolve_cdse_credentials_file_path",
]
