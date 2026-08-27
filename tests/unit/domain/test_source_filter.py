"""Source-file admission policy tests."""

from oce.domain.services.source_filter import is_binary_source, is_ignored_source_path


def test_ignores_svg_and_binary_artifacts():
    assert is_ignored_source_path("docs/assets/install-script.svg")
    assert is_ignored_source_path("assets/logo.PNG")
    assert is_ignored_source_path("dist/client.min.js")


def test_ignores_dependency_directories_by_path_segment():
    assert is_ignored_source_path("frontend/node_modules/pkg/index.js")
    assert is_ignored_source_path("src/pkg.egg-info/PKG-INFO")
    assert is_ignored_source_path("openclaw-retrieval-eval/queries.jsonl")
    assert is_ignored_source_path("generated/schema.xml")
    assert is_ignored_source_path("coverage/report.json")
    assert not is_ignored_source_path("src/build_tools/compiler.py")


def test_keeps_text_configuration_formats():
    assert not is_ignored_source_path("package.json")
    assert not is_ignored_source_path("config/settings.json5")
    assert not is_ignored_source_path("config/settings.yaml")
    assert not is_ignored_source_path("config/settings.toml")
    assert not is_ignored_source_path("apps/android/AndroidManifest.xml")
    assert not is_ignored_source_path("test/fixtures/prompt-snapshots/result.json")


def test_ignores_data_and_visual_asset_formats():
    assert is_ignored_source_path("data/events.jsonl")
    assert is_ignored_source_path("data/events.csv")
    assert is_ignored_source_path("test/fixtures/assets/icon.svg")


def test_binary_detection_uses_nul_signal():
    assert is_binary_source("PNG\x00data")
    assert not is_binary_source("plain text")
