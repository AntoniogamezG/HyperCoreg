"""CDSE authentication and credential helpers.

This module is the migration target for auth/session logic previously hosted
in ``_legacy_coreg.py``.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import sys
import threading
from time import sleep
from typing import Any, Callable, Dict, Optional, Tuple

import requests
from requests.auth import AuthBase

logger = logging.getLogger("COREG_PROCESSING")

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
HTTP_TIMEOUT_S = 30
HTTP_AUTH_RETRY_ATTEMPTS = 2
HTTP_RETRY_BACKOFF_BASE_S = 1.0
HTTP_RETRY_BACKOFF_FACTOR = 2.0
HTTP_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

PromptUserpassFn = Callable[[], Tuple[str, str, Optional[str]]]
_CDSE_CREDENTIAL_CACHE_LOCK = threading.Lock()
_CDSE_CREDENTIAL_CACHE: Dict[str, Optional[str]] = {
    "username": None,
    "password": None,
    "totp": None,
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
    user = str(username or "").strip()
    pwd = str(password or "")
    otp = str(totp).strip() if totp is not None else ""
    with _CDSE_CREDENTIAL_CACHE_LOCK:
        _CDSE_CREDENTIAL_CACHE["username"] = user or None
        _CDSE_CREDENTIAL_CACHE["password"] = pwd if pwd else None
        _CDSE_CREDENTIAL_CACHE["totp"] = otp or None


def _get_cached_cdse_public_credentials() -> Optional[Tuple[str, str, Optional[str]]]:
    with _CDSE_CREDENTIAL_CACHE_LOCK:
        user = str(_CDSE_CREDENTIAL_CACHE.get("username") or "").strip()
        pwd = _CDSE_CREDENTIAL_CACHE.get("password") or ""
        otp = str(_CDSE_CREDENTIAL_CACHE.get("totp") or "").strip() or None
    if user and pwd:
        return user, str(pwd), otp
    return None


def _clear_cached_cdse_public_credentials() -> None:
    with _CDSE_CREDENTIAL_CACHE_LOCK:
        _CDSE_CREDENTIAL_CACHE["username"] = None
        _CDSE_CREDENTIAL_CACHE["password"] = None
        _CDSE_CREDENTIAL_CACHE["totp"] = None


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


class _BearerAuth(AuthBase):
    def __init__(self, token: str):
        self.token = token

    def __call__(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        return request


def _create_cdse_bearer_session(access_token: str) -> requests.Session:
    token = (access_token or "").strip()
    if not token:
        raise ValueError(_fmt_issue("AUTH", "Empty CDSE bearer token provided."))
    session = requests.Session()
    session.auth = _BearerAuth(token)
    session.headers.update({"Accept": "application/json"})
    return session


def _request_cdse_access_token(payload: Dict[str, str], flow_name: str) -> str:
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
                raise RuntimeError(
                    _fmt_issue("AUTH", f"{flow_name} token generation failed ({response.status_code}): {msg}")
                )

            token = response.json().get("access_token")
            if not token:
                raise RuntimeError(_fmt_issue("AUTH", f"{flow_name} token response missing access_token."))
            return token
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
            raise RuntimeError(_fmt_issue("AUTH", f"{flow_name} token request failed: {exc}")) from exc
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
            raise

    if last_error is not None:
        raise RuntimeError(_fmt_issue("AUTH", f"{flow_name} token generation failed: {last_error}"))
    raise RuntimeError(_fmt_issue("AUTH", f"{flow_name} token generation failed unexpectedly."))


def _generate_cdse_public_access_token(username: str, password: str, totp: Optional[str] = None) -> str:
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


def _create_cdse_public_session(username: str, password: str, totp: Optional[str] = None) -> requests.Session:
    token = _generate_cdse_public_access_token(username=username, password=password, totp=totp)
    return _create_cdse_bearer_session(token)


def _create_cdse_session_from_environment() -> Optional[requests.Session]:
    access_token = (os.environ.get("CDSE_ACCESS_TOKEN") or "").strip()
    if access_token:
        logger.info("Using CDSE bearer token from environment (CDSE_ACCESS_TOKEN).")
        return _create_cdse_bearer_session(access_token)

    client_id = (os.environ.get("CDSE_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("CDSE_CLIENT_SECRET") or "").strip()
    if client_id and client_secret:
        logger.info("Using CDSE client credentials from environment.")
        token = _generate_cdse_client_access_token(client_id=client_id, client_secret=client_secret)
        return _create_cdse_bearer_session(token)
    if client_id or client_secret:
        logger.warning(
            _fmt_issue(
                "AUTH",
                "Incomplete CDSE client credentials in environment; set both "
                "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET.",
            )
        )

    username = (os.environ.get("CDSE_USERNAME") or "").strip()
    password = os.environ.get("CDSE_PASSWORD") or ""
    if username and password:
        logger.info("Using CDSE username/password from environment.")
        totp = (os.environ.get("CDSE_TOTP") or "").strip() or None
        sess = _create_cdse_public_session(username=username, password=password, totp=totp)
        _cache_cdse_public_credentials(username=username, password=password, totp=totp)
        return sess
    if username or password:
        logger.warning(
            _fmt_issue(
                "AUTH",
                "Incomplete CDSE username/password in environment; set both "
                "CDSE_USERNAME and CDSE_PASSWORD.",
            )
        )

    file_creds = _read_cdse_credentials_file()
    if isinstance(file_creds, dict):
        creds_path = _resolve_cdse_credentials_file_path()

        file_access_token = _extract_cdse_credential_value(
            file_creds,
            "access_token",
            "CDSE_ACCESS_TOKEN",
            "cdse_access_token",
            "token",
        )
        if file_access_token:
            logger.info("Using CDSE bearer token from credentials file: %s", creds_path)
            return _create_cdse_bearer_session(file_access_token)

        file_client_id = _extract_cdse_credential_value(file_creds, "client_id", "CDSE_CLIENT_ID")
        file_client_secret = _extract_cdse_credential_value(file_creds, "client_secret", "CDSE_CLIENT_SECRET")
        if file_client_id and file_client_secret:
            logger.info("Using CDSE client credentials from credentials file: %s", creds_path)
            token = _generate_cdse_client_access_token(
                client_id=file_client_id,
                client_secret=file_client_secret,
            )
            return _create_cdse_bearer_session(token)
        if file_client_id or file_client_secret:
            logger.warning(
                _fmt_issue(
                    "AUTH",
                    "Incomplete CDSE client credentials in credentials file; expected "
                    "both client_id and client_secret.",
                )
            )

        file_username = _extract_cdse_credential_value(file_creds, "username", "CDSE_USERNAME")
        file_password = _extract_cdse_credential_value(file_creds, "password", "CDSE_PASSWORD")
        if file_username and file_password:
            file_totp = _extract_cdse_credential_value(file_creds, "totp", "CDSE_TOTP") or None
            logger.info("Using CDSE username/password from credentials file: %s", creds_path)
            sess = _create_cdse_public_session(username=file_username, password=file_password, totp=file_totp)
            _cache_cdse_public_credentials(username=file_username, password=file_password, totp=file_totp)
            return sess
        if file_username or file_password:
            logger.warning(
                _fmt_issue(
                    "AUTH",
                    "Incomplete CDSE username/password in credentials file; expected "
                    "both username and password.",
                )
            )

    return None


def _prompt_cdse_userpass_cli(max_prompt_attempts: int = 2) -> Tuple[str, str, Optional[str]]:
    stdin = getattr(sys, "stdin", None)
    if stdin is None or not callable(getattr(stdin, "isatty", None)) or not bool(stdin.isatty()):
        raise RuntimeError(
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
            raise RuntimeError(
                _fmt_issue(
                    "AUTH",
                    "CDSE credential prompt reached EOF (non-interactive input stream). "
                    "Set CDSE_USERNAME/CDSE_PASSWORD (or CDSE_CLIENT_ID/CDSE_CLIENT_SECRET), "
                    "or define them in ~/.hypercoreg/credentials.json.",
                )
            ) from exc
        except KeyboardInterrupt as exc:
            raise RuntimeError(_fmt_issue("AUTH", "CDSE login cancelled by user.")) from exc

        if user and pwd:
            return user, pwd, totp

        logger.warning(_fmt_issue("AUTH", f"Username and password are required (attempt {attempt}/{attempts})."))

    raise RuntimeError(_fmt_issue("AUTH", "CDSE username/password were not provided."))


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
        raise RuntimeError(_fmt_issue("AUTH", "CDSE login cancelled by user."))

    return result["username"], result["password"], (result["totp"] or None)


def _request_cdse_userpass(
    allow_gui_prompt: bool,
    prompt_userpass_fn: Optional[PromptUserpassFn] = None,
) -> Tuple[str, str, Optional[str]]:
    if allow_gui_prompt:
        if prompt_userpass_fn is not None:
            return prompt_userpass_fn()
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                _fmt_issue(
                    "AUTH",
                    "GUI credential prompt requested from a worker thread. "
                    "Provide a main-thread prompt callback.",
                )
            )
        return _prompt_cdse_userpass_gui()

    return _prompt_cdse_userpass_cli()


def _create_public_session_with_retry(
    allow_gui_prompt: bool = False,
    max_prompt_attempts: int = 2,
    prompt_userpass_fn: Optional[PromptUserpassFn] = None,
) -> requests.Session:
    cached_userpass = _get_cached_cdse_public_credentials()
    if cached_userpass is not None:
        user, pwd, totp = cached_userpass
        try:
            logger.info("Reusing cached CDSE username/password for session refresh.")
            return _create_cdse_public_session(user, pwd, totp=totp)
        except Exception as exc:
            logger.warning(
                _fmt_issue(
                    "AUTH",
                    f"Cached CDSE credentials failed ({exc}); requesting credentials again.",
                )
            )
            _clear_cached_cdse_public_credentials()

    last_error: Optional[Exception] = None
    attempts = max(1, int(max_prompt_attempts))
    for attempt in range(1, attempts + 1):
        try:
            user, pwd, totp = _request_cdse_userpass(
                allow_gui_prompt=allow_gui_prompt,
                prompt_userpass_fn=prompt_userpass_fn,
            )
            sess = _create_cdse_public_session(user, pwd, totp=totp)
            _cache_cdse_public_credentials(username=user, password=pwd, totp=totp)
            return sess
        except RuntimeError as exc:
            if "cancelled" in str(exc).lower():
                raise
            last_error = exc
        except Exception as exc:
            last_error = exc

        if attempt < attempts:
            logger.warning(
                _fmt_issue("AUTH", f"CDSE login failed (attempt {attempt}/{attempts}): {last_error}")
            )

    raise RuntimeError(_fmt_issue("AUTH", f"CDSE username/password authentication failed: {last_error}"))


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
    "_BearerAuth",
    "_cache_cdse_public_credentials",
    "_clear_cached_cdse_public_credentials",
    "_create_cdse_bearer_session",
    "_create_cdse_public_session",
    "_create_cdse_session_from_environment",
    "_create_cdse_session_with_retry",
    "_create_public_session_with_retry",
    "_extract_cdse_credential_value",
    "_generate_cdse_client_access_token",
    "_generate_cdse_public_access_token",
    "_get_cached_cdse_public_credentials",
    "_prompt_cdse_userpass_cli",
    "_prompt_cdse_userpass_gui",
    "_read_cdse_credentials_file",
    "_request_cdse_access_token",
    "_request_cdse_userpass",
    "_resolve_cdse_credentials_file_path",
]
