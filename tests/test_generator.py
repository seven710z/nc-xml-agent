"""XML 生成器单元测试。"""

from app.core.schemas import ControlKind, RequirementItem, RequirementParse
from app.xml_gen.generator import _sanitize_id, build_xml


def _make_parse() -> RequirementParse:
    return RequirementParse(
        intent="测试界面",
        items=[
            RequirementItem(
                widget="主轴转速显示",
                control_kind=ControlKind.DISPLAY,
                binding="#5001",
                layout_hint="顶部",
            ),
            RequirementItem(
                widget="启动按钮",
                control_kind=ControlKind.BUTTON,
                binding="#G7.2",
                layout_hint="左侧",
            ),
        ],
    )


def test_build_xml_has_screen_root():
    xml = build_xml(_make_parse(), retrieved=[])
    assert xml.lstrip().startswith("<?xml")
    assert "<screen>" in xml
    assert "</screen>" in xml


def test_build_xml_contains_all_widgets():
    xml = build_xml(_make_parse(), retrieved=[])
    assert "spindleSpeed" in xml
    assert "startButton" in xml
    assert 'type="display"' in xml
    assert 'type="button"' in xml


def test_sanitize_id_maps_known_chinese():
    assert _sanitize_id("主轴转速显示") == "spindleSpeed"
    assert _sanitize_id("启动按钮") == "startButton"


def test_sanitize_id_falls_back_safely():
    # 未知中文名应降级为不破坏 XML 的合法 id
    result = _sanitize_id("未知控件!!")
    assert result == "widget" or result.isidentifier()
