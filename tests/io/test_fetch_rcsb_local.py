"""fetch_rcsb against a local HTTP server (no network in CI).

The Rust backend reads PROXIDE_RCSB_BASE_URL on every call, so these tests
point it at a 127.0.0.1 fixture server.
"""

import http.server
import os
import pathlib
import threading

import pytest

from proxide.io import fetching


class _Handler(http.server.BaseHTTPRequestHandler):
  """Serves the routes stored on the server; unknown paths get 404."""

  def do_GET(self) -> None:  # noqa: N802 - http.server API
    status, body = self.server.routes.get(self.path, (404, b""))  # type: ignore[attr-defined]
    self.send_response(status)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def log_message(self, format: str, *args: object) -> None:  # noqa: A002
    return


@pytest.fixture
def rcsb_server(monkeypatch: pytest.MonkeyPatch):
  server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
  server.routes = {}  # type: ignore[attr-defined]
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  monkeypatch.setenv("PROXIDE_RCSB_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/")
  try:
    yield server
  finally:
    server.shutdown()
    server.server_close()


def test_success_writes_full_body(rcsb_server, tmp_path: pathlib.Path) -> None:
  body = b"HEADER    TEST\n" + b"ATOM  line\n" * 500
  rcsb_server.routes["/1ABC.pdb"] = (200, body)  # type: ignore[attr-defined]

  path = fetching.fetch_rcsb("1abc", output_dir=str(tmp_path), format_type="pdb")

  assert path.endswith("1ABC.pdb")
  assert pathlib.Path(path).read_bytes() == body
  assert sorted(p.name for p in tmp_path.iterdir()) == ["1ABC.pdb"]


def test_404_in_both_formats_raises_and_leaves_no_file(rcsb_server, tmp_path: pathlib.Path) -> None:
  with pytest.raises(RuntimeError, match="404"):
    fetching.fetch_rcsb("9zzz", output_dir=str(tmp_path), format_type="pdb")

  assert list(tmp_path.iterdir()) == []


def test_empty_cached_file_is_redownloaded(rcsb_server, tmp_path: pathlib.Path) -> None:
  rcsb_server.routes["/1ABC.pdb"] = (200, b"HEADER    FRESH\n")  # type: ignore[attr-defined]
  (tmp_path / "1ABC.pdb").write_bytes(b"")

  path = fetching.fetch_rcsb("1abc", output_dir=str(tmp_path), format_type="pdb")

  assert pathlib.Path(path).read_bytes() == b"HEADER    FRESH\n"


def test_nonempty_cached_file_is_hit_without_request(rcsb_server, tmp_path: pathlib.Path) -> None:
  # No route registered: any request would 404 and fail the call.
  (tmp_path / "1ABC.pdb").write_bytes(b"HEADER    CACHED\n")

  path = fetching.fetch_rcsb("1abc", output_dir=str(tmp_path), format_type="pdb")

  assert pathlib.Path(path).read_bytes() == b"HEADER    CACHED\n"


@pytest.mark.network
def test_real_rcsb_smoke(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Real download of 1CRN from files.rcsb.org (skipped in CI unless opted in)."""
  if os.environ.get("CI") and not os.environ.get("PROXIDE_NETWORK_TESTS"):
    pytest.skip("network smoke test: set PROXIDE_NETWORK_TESTS=1 to run under CI")
  monkeypatch.delenv("PROXIDE_RCSB_BASE_URL", raising=False)

  path = fetching.fetch_rcsb("1CRN", output_dir=str(tmp_path), format_type="pdb")

  content = pathlib.Path(path).read_bytes()
  assert len(content) > 10_000
  assert content.startswith(b"HEADER")
