"""CLI tests for `proxide fetch`. Fetch functions are mocked, so no network is used."""

from unittest import mock

from typer.testing import CliRunner

from proxide.cli.main import app
from proxide.io.fetching import AFDB_DEFAULT_VERSION

runner = CliRunner()


def test_afdb_default_version_is_6() -> None:
  assert AFDB_DEFAULT_VERSION == 6


@mock.patch("proxide.cli.main.fetch_afdb", return_value="/tmp/AF-P69905-F1-model_v6.pdb")
def test_fetch_afdb_without_version_flag_uses_default(mock_fetch: mock.MagicMock, tmp_path) -> None:
  result = runner.invoke(app, ["fetch", "afdb", "P69905", "-o", str(tmp_path)])

  assert result.exit_code == 0, result.output
  mock_fetch.assert_called_once_with("P69905", str(tmp_path), version=AFDB_DEFAULT_VERSION)


@mock.patch("proxide.cli.main.fetch_afdb", return_value="/tmp/AF-P69905-F1-model_v4.pdb")
def test_fetch_afdb_version_flag_overrides_default(mock_fetch: mock.MagicMock, tmp_path) -> None:
  result = runner.invoke(app, ["fetch", "afdb", "P69905", "-o", str(tmp_path), "-v", "4"])

  assert result.exit_code == 0, result.output
  mock_fetch.assert_called_once_with("P69905", str(tmp_path), version=4)
