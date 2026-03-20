"""Thin FastAPI proxy for steered vLLM servers.

Supports two modes depending on how the upstream vLLM server is started:

**Server-level steering** (preferred — enables CUDA graphs for ~2.6x speedup):
    The upstream vLLM is started with ``--steer-vector-path``.  The proxy
    simply passes requests through without modification.  Steering config
    can be updated via ``POST /v1/steering`` on the upstream.

**Per-request injection** (legacy — no CUDA graphs):
    The upstream vLLM is started with ``--enable-steer-vector`` only.
    The proxy injects ``steer_vector_request`` into every POST body.

Usage:
    # Server-level mode (proxy just forwards):
    uv run python -m src.steering_proxy \
        --upstream http://localhost:8017 \
        --server-level \
        --port 8018

    # Legacy per-request injection mode:
    uv run python -m src.steering_proxy \
        --upstream http://localhost:8017 \
        --vector-path outputs/vector.gguf \
        --scale 2.0 \
        --target-layers 10 11 12 13 14 15 \
        --port 8018
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

# Module-level state set by configure() before server starts.
_upstream: str = ""
_steer_dict: dict = {}
_server_level: bool = False
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create a persistent httpx.AsyncClient for the app's lifetime."""
    global _http_client
    _http_client = httpx.AsyncClient(timeout=300.0)
    try:
        yield
    finally:
        await _http_client.aclose()
        _http_client = None


app = FastAPI(title="Steering Proxy", lifespan=_lifespan)


def configure(
    upstream: str,
    *,
    server_level: bool = False,
    vector_path: str | None = None,
    scale: float = 1.0,
    target_layers: list[int] | None = None,
    algorithm: str = "direct",
    normalize: bool = True,
) -> None:
    """Set the proxy's upstream URL and steering parameters.

    Must be called before the ASGI app handles any requests.

    Parameters
    ----------
    upstream:
        Base URL of the vLLM server.
    server_level:
        If True, upstream uses ``--steer-vector-path`` and no per-request
        injection is needed.
    vector_path:
        Path to steering vector file (required for per-request mode).
    scale:
        Steering scale factor (per-request mode only).
    target_layers:
        Target layer indices (per-request mode only).
    algorithm:
        Steering algorithm (per-request mode only).
    normalize:
        Whether to normalize (per-request mode only).

    Raises
    ------
    FileNotFoundError:
        If per-request mode and the vector file does not exist.
    ValueError:
        If per-request mode and required args are missing.
    """
    global _upstream, _steer_dict, _server_level

    _upstream = upstream.rstrip("/")
    _server_level = server_level

    if server_level:
        _steer_dict = {}
        logger.info(
            "Proxy configured in SERVER-LEVEL mode: upstream=%s "
            "(requests forwarded without steer_vector_request injection)",
            _upstream,
        )
        return

    # Per-request injection mode
    if vector_path is None:
        raise ValueError(
            "--vector-path is required in per-request mode "
            "(omit --server-level to use per-request injection)"
        )
    if target_layers is None:
        raise ValueError("--target-layers is required in per-request mode")

    vp = Path(vector_path)
    if not vp.exists():
        raise FileNotFoundError(f"Steering vector not found: {vector_path}")

    _steer_dict = {
        "steer_vector_local_path": str(vp.resolve()),  # absolute path for vLLM
        "scale": scale,
        "target_layers": target_layers,
        "algorithm": algorithm,
        "normalize": normalize,
        # Trigger tokens [-1] = apply steering to ALL tokens.
        # Without these, the vector loads but never activates.
        "prefill_trigger_tokens": [-1],
        "generate_trigger_tokens": [-1],
    }
    logger.info(
        "Proxy configured in PER-REQUEST mode: upstream=%s, steer=%s",
        _upstream,
        _steer_dict,
    )


def verify_upstream_supports_steering(
    upstream: str, vector_path: str, target_layers: list[int]
) -> None:
    """Verify the upstream vLLM supports steering by sending a scale=0 test request.

    Uses the real vector at scale=0 (no-op) to confirm the server accepts
    steer_vector_request without silently ignoring it. This is safe — it won't
    crash the server like a nonexistent vector path would.

    Raises RuntimeError if the upstream is unreachable or appears to be vanilla vLLM.
    """
    # Confirm upstream is alive
    try:
        resp = httpx.get(f"{upstream}/v1/models", timeout=5.0)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Upstream at {upstream} returned {resp.status_code} for /v1/models"
            )
    except httpx.ConnectError as e:
        raise RuntimeError(f"Cannot reach upstream at {upstream}: {e}") from e

    model_name = resp.json()["data"][0]["id"]

    # Send a request WITH steer_vector_request at scale=0 (no-op steering)
    steered_body = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 3,
        "temperature": 0.0,
        "steer_vector_request": {
            "steer_vector_local_path": vector_path,
            "scale": 0.0,
            "target_layers": target_layers,
            "prefill_trigger_tokens": [-1],
            "generate_trigger_tokens": [-1],
        },
    }
    try:
        steered_resp = httpx.post(
            f"{upstream}/v1/chat/completions", json=steered_body, timeout=30.0
        )
    except httpx.ConnectError as e:
        raise RuntimeError(f"Cannot reach upstream at {upstream}: {e}") from e

    if steered_resp.status_code != 200:
        error_msg = steered_resp.text[:500]
        raise RuntimeError(
            f"Upstream rejected steering test request (status={steered_resp.status_code}): "
            f"{error_msg}"
        )

    # Now send the SAME request WITHOUT steer_vector_request
    plain_body = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 3,
        "temperature": 0.0,
    }
    try:
        plain_resp = httpx.post(
            f"{upstream}/v1/chat/completions", json=plain_body, timeout=30.0
        )
    except httpx.ConnectError as e:
        raise RuntimeError(f"Cannot reach upstream at {upstream}: {e}") from e

    if plain_resp.status_code != 200:
        raise RuntimeError(
            f"Upstream failed on plain request (status={plain_resp.status_code})"
        )

    # Both succeeded — the server accepted steer_vector_request.
    # Vanilla vLLM would also accept it (silently ignoring), so this isn't
    # a perfect test. But combined with the vector file check and the
    # --enable-steer-vector requirement, it catches most failure modes.
    logger.info(
        "Steering verification passed: upstream accepted steered request "
        "with vector=%s",
        vector_path,
    )


def get_steer_dict() -> dict:
    """Return a copy of the current steering injection dict."""
    return dict(_steer_dict)


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request) -> Response:
    """Forward requests to upstream vLLM, optionally injecting steering on POST."""
    url = f"{_upstream}/v1/{path}"

    assert _http_client is not None, "App not started — lifespan not entered"

    if request.method == "GET":
        resp = await _http_client.get(url, headers=dict(request.headers))
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    # POST — inject steering into the body only in per-request mode
    body = await request.json()

    if not _server_level and _steer_dict:
        # Per-request injection mode: add steer_vector_request to body
        body["steer_vector_request"] = dict(_steer_dict)

    # Check if streaming is requested
    stream = body.get("stream", False)

    if stream:
        async def stream_response():
            async with _http_client.stream("POST", url, json=body) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )

    resp = await _http_client.post(url, json=body)

    # Fail loudly if upstream returned an error
    if resp.status_code != 200:
        logger.error(
            "Upstream returned %d for %s: %s",
            resp.status_code,
            url,
            resp.text[:500],
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "upstream": _upstream, "mode": "server-level" if _server_level else "per-request"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steering injection proxy")
    parser.add_argument("--upstream", required=True, help="Upstream vLLM base URL")
    parser.add_argument(
        "--server-level",
        action="store_true",
        help="Upstream uses --steer-vector-path; proxy just forwards requests.",
    )
    parser.add_argument("--vector-path", default=None, help="Path to steering vector (per-request mode)")
    parser.add_argument("--scale", type=float, default=1.0, help="Steering scale (per-request mode)")
    parser.add_argument(
        "--target-layers", type=int, nargs="+", default=None, help="Target layer indices (per-request mode)"
    )
    parser.add_argument("--algorithm", default="direct", help="Steering algorithm (per-request mode)")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--port", type=int, default=8018, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    configure(
        upstream=args.upstream,
        server_level=args.server_level,
        vector_path=args.vector_path,
        scale=args.scale,
        target_layers=args.target_layers,
        algorithm=args.algorithm,
        normalize=args.normalize,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
