"""Tests for the steering proxy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.steering_proxy import (
    app,
    configure,
    get_steer_dict,
    verify_upstream_supports_steering,
)


@pytest.fixture()
def dummy_vector(tmp_path: Path) -> Path:
    """Create a dummy vector file for testing."""
    p = tmp_path / "test_vector.gguf"
    p.write_bytes(b"fake")
    return p


@pytest.fixture(autouse=True)
def _setup_proxy(dummy_vector: Path) -> None:
    """Configure the proxy with test parameters before each test."""
    configure(
        upstream="http://fake-vllm:8017",
        vector_path=str(dummy_vector),
        scale=2.0,
        target_layers=[10, 11, 12],
        algorithm="direct",
        normalize=True,
    )


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestConfigure:
    def test_steer_dict_set(self, dummy_vector: Path) -> None:
        d = get_steer_dict()
        # configure resolves to absolute path
        assert d["steer_vector_local_path"] == str(dummy_vector.resolve())
        assert d["scale"] == 2.0
        assert d["target_layers"] == [10, 11, 12]
        assert d["algorithm"] == "direct"
        assert d["normalize"] is True

    def test_reconfigure(self, tmp_path: Path) -> None:
        other_vec = tmp_path / "other.gguf"
        other_vec.write_bytes(b"fake")
        configure(
            upstream="http://other:9000",
            vector_path=str(other_vec),
            scale=0.5,
            target_layers=[5, 6],
        )
        d = get_steer_dict()
        assert d["scale"] == 0.5
        assert d["target_layers"] == [5, 6]

    def test_missing_vector_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Steering vector not found"):
            configure(
                upstream="http://fake:8017",
                vector_path="/nonexistent/vector.gguf",
                scale=1.0,
                target_layers=[10],
            )


class TestHealth:
    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "fake-vllm" in data["upstream"]


class TestProxyInjection:
    """Test that POST requests get steering dict injected."""

    def test_injects_steer_vector_request(
        self, client: TestClient, dummy_vector: Path
    ) -> None:
        captured_body: dict = {}

        async def mock_post(self, url, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello"}}]},
            )

        with patch.object(httpx.AsyncClient, "post", mock_post):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10,
                },
            )

        assert resp.status_code == 200
        assert "steer_vector_request" in captured_body
        svr = captured_body["steer_vector_request"]
        assert svr["steer_vector_local_path"] == str(dummy_vector.resolve())
        assert svr["scale"] == 2.0
        assert svr["target_layers"] == [10, 11, 12]

    def test_preserves_original_body(self, client: TestClient) -> None:
        captured_body: dict = {}

        async def mock_post(self, url, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            return httpx.Response(200, json={"choices": []})

        with patch.object(httpx.AsyncClient, "post", mock_post):
            client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "def hello():",
                    "max_tokens": 100,
                    "temperature": 0.8,
                },
            )

        assert captured_body["model"] == "test-model"
        assert captured_body["prompt"] == "def hello():"
        assert captured_body["max_tokens"] == 100
        assert captured_body["temperature"] == 0.8

    def test_forwards_to_correct_url(self, client: TestClient) -> None:
        captured_url: str = ""

        async def mock_post(self, url, **kwargs):
            nonlocal captured_url
            captured_url = url
            return httpx.Response(200, json={"choices": []})

        with patch.object(httpx.AsyncClient, "post", mock_post):
            client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": []},
            )

        assert captured_url == "http://fake-vllm:8017/v1/chat/completions"


class TestVerifyUpstream:
    @staticmethod
    def _mock_request(method: str = "GET", url: str = "http://fake:8017") -> httpx.Request:
        return httpx.Request(method, url)

    def test_passes_when_both_requests_succeed(self, dummy_vector: Path) -> None:
        """Verification passes when upstream accepts both steered and plain requests."""
        req = self._mock_request()

        def mock_get(url, **kwargs):
            return httpx.Response(
                200, json={"data": [{"id": "test-model"}]}, request=req
            )

        def mock_post(url, **kwargs):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "OK"}}]},
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx, "get", mock_get), patch.object(
            httpx, "post", mock_post
        ):
            verify_upstream_supports_steering(
                "http://fake:8017", str(dummy_vector), [10]
            )

    def test_raises_when_steered_request_fails(self, dummy_vector: Path) -> None:
        """If upstream rejects the steered request, raise."""
        req = self._mock_request()

        def mock_get(url, **kwargs):
            return httpx.Response(
                200, json={"data": [{"id": "test-model"}]}, request=req
            )

        def mock_post(url, **kwargs):
            return httpx.Response(
                500,
                json={"error": "Engine crashed"},
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx, "get", mock_get), patch.object(
            httpx, "post", mock_post
        ):
            with pytest.raises(RuntimeError, match="rejected steering test request"):
                verify_upstream_supports_steering(
                    "http://fake:8017", str(dummy_vector), [10]
                )

    def test_raises_on_unreachable(self, dummy_vector: Path) -> None:
        def mock_get(url, **kwargs):
            raise httpx.ConnectError("refused")

        with patch.object(httpx, "get", mock_get):
            with pytest.raises(RuntimeError, match="Cannot reach upstream"):
                verify_upstream_supports_steering(
                    "http://fake:8017", str(dummy_vector), [10]
                )


class TestParseArgs:
    def test_parse_args(self) -> None:
        from src.steering_proxy import parse_args

        args = parse_args([
            "--upstream", "http://localhost:8017",
            "--vector-path", "/tmp/v.gguf",
            "--scale", "1.5",
            "--target-layers", "10", "11", "12",
            "--port", "9000",
        ])
        assert args.upstream == "http://localhost:8017"
        assert args.vector_path == "/tmp/v.gguf"
        assert args.scale == 1.5
        assert args.target_layers == [10, 11, 12]
        assert args.port == 9000
        assert args.normalize is True

    def test_no_normalize(self) -> None:
        from src.steering_proxy import parse_args

        args = parse_args([
            "--upstream", "http://localhost:8017",
            "--vector-path", "/tmp/v.gguf",
            "--scale", "1.0",
            "--target-layers", "10",
            "--no-normalize",
        ])
        assert args.normalize is False
