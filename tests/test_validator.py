"""XML 校验器单元测试 —— 覆盖三层校验逻辑。"""

from app.xml_gen.validator import validate_xml


def _valid_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<screen>
<widget id="spindleSpeed" type="display">
  <label>主轴转速显示</label>
  <binding var="#5001">#5001</binding>
  <layout>top</layout>
</widget>
</screen>"""


def test_valid_xml_passes():
    result = validate_xml(_valid_xml(), retrieved=[])
    assert result.valid is True
    assert result.issues == []


def test_malformed_xml_rejected():
    # 标签不闭合 —— 良构性错误（第 1 层）
    bad = "<screen><widget id='a'></screen>"
    result = validate_xml(bad, retrieved=[])
    assert result.valid is False
    assert any(i.rule == "wellformed" for i in result.issues)


def test_wrong_root_node_rejected():
    # 根节点不是 screen —— schema 错误（第 2 层）
    bad = "<page><widget id='a' type='display'></widget></page>"
    result = validate_xml(bad, retrieved=[])
    assert result.valid is False
    assert any(i.rule == "schema" and "根节点" in i.message for i in result.issues)


def test_duplicate_id_rejected():
    bad = """<screen>
<widget id="dup" type="display"><binding var="#5001">#5001</binding></widget>
<widget id="dup" type="button"><binding var="#G7.2">#G7.2</binding></widget>
</screen>"""
    result = validate_xml(bad, retrieved=[])
    assert result.valid is False
    assert any("id 重复" in i.message for i in result.issues)


def test_input_bound_to_readonly_var_rejected():
    # 只读变量 #5xxx 不能绑给 input —— 业务规则错误（第 3 层）
    bad = """<screen>
<widget id="feed" type="input">
  <binding var="#5001">#5001</binding>
</widget>
</screen>"""
    result = validate_xml(bad, retrieved=[])
    assert result.valid is False
    assert any(i.rule == "binding" and "只读" in i.message for i in result.issues)


def test_input_missing_minmax_warns():
    # input 缺 min/max 只警告、不判 error
    xml = """<screen>
<widget id="feed" type="input">
  <binding var="#1420">#1420</binding>
</widget>
</screen>"""
    result = validate_xml(xml, retrieved=[])
    assert result.valid is True  # warning 不算 error
    assert any(i.severity == "warning" and "min/max" in i.message for i in result.issues)


def test_too_many_nodes_rejected():
    # 节点数超上限，防递归爆炸
    widgets = "".join(
        f'<widget id="w{i}" type="display"><binding var="#5001">#5001</binding></widget>'
        for i in range(250)
    )
    xml = f"<screen>{widgets}</screen>"
    result = validate_xml(xml, retrieved=[])
    assert result.valid is False
    assert any("节点数" in i.message for i in result.issues)
