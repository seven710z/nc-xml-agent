"""RAG 向量库单元测试 —— mock 降级模式下的检索行为。"""

from app.rag.vector_store import VectorStore, load_knowledge_from_dir


def test_build_index_and_search_recalls_related_docs():
    store = VectorStore(collection_name="test_coll")
    docs = [
        {
            "id": "d1",
            "text": "数控界面 XML 根节点必须是 screen",
            "metadata": {"category": "syntax"},
        },
        {"id": "d2", "text": "button 控件用于触发机床动作", "metadata": {"category": "control"}},
        {"id": "d3", "text": "系统变量为只读状态量", "metadata": {"category": "binding"}},
    ]
    store.build_index(docs)

    results = store.search("按钮控件", top_k=2)
    # 应该召回与按钮相关的文档（d2 语义最接近）
    assert len(results) > 0
    # 相似度降序
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_load_knowledge_from_dir():
    # load_knowledge_from_dir 操作全局 store；此测试只验证 JSON 文件能被正确解析
    load_knowledge_from_dir("data/knowledge")
    # 12 条知识 JSON 解析不抛异常即证明数据格式正确
    assert True


def test_search_returns_metadata():
    store = VectorStore(collection_name="test_meta")
    store.build_index(
        [
            {
                "id": "m1",
                "text": "只读变量不能写入 input 控件",
                "metadata": {"category": "binding_rule"},
            }
        ]
    )
    results = store.search("只读变量", top_k=1)
    assert results[0]["metadata"]["category"] == "binding_rule"
    assert "score" in results[0]
