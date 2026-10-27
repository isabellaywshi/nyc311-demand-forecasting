import yaml
from typer.testing import CliRunner

from nyc311.cli import app


def test_panel_command_writes_panel_and_reports_unmapped(raw_fixture, tmp_path, monkeypatch):
    settings = yaml.safe_load(open("config/settings.yaml"))
    settings["raw_dir"] = str(raw_fixture)
    settings["processed_dir"] = str(tmp_path / "processed")
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.setenv("NYC311_SETTINGS", str(path))

    result = CliRunner().invoke(app, ["panel"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "processed" / "weekly_panel.parquet").exists()
    assert "unmapped" in result.output.lower()
