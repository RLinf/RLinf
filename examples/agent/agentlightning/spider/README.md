# Spider（文本转 SQL）— RLinf 版

## 本 Agent 做什么

与 [Agent-Lightning Spider 示例](https://github.com/microsoft/agent-lightning/tree/main/examples/spider) 一致：在 **Spider** 风格数据上训练 **text-to-SQL** 智能体；基于 **LangGraph / LangChain**（`sql_agent.py` 中 `LitSQLAgent`），按 **`db_id`** 访问 SQLite、生成 SQL，与标准 **`query`** 比对得奖励。

## 依赖

1. **RLinf**：安装与环境见官方文档 <https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html>。
2. **AgentLightning + LangGraph / LangChain**。

```bash
pip install "agentlightning" "langgraph<1.0" "langchain[openai]<1.0" "langchain-community" "langchain-text-splitters<1.0" "sqlparse" "nltk"
```

4. **数据**：[How to Train a SQL Agent](https://github.com/microsoft/agent-lightning/blob/main/docs/how-to/train-sql-agent.md)。

**SQLite 根目录**：默认取 **`train_data_paths[0]` 所在目录**（与 parquet 同级的 `database/`、`test_database/`）。可在 yaml 里用 **`data.spider_data_dir`** 显式覆盖。仅在不传 `LitSQLAgent(spider_data_dir=...)` 时（例如 `debug_sql_agent`）才回退到环境变量 **`RLINF_SPIDER_DATA_DIR`** 或相对路径 **`data/`**。


## 运行

在 **`examples/agent/agentlightning/spider`** 下（Linux 推荐）：

```bash
bash run_spider.sh
bash run_spider.sh qwen2.5-1.5b-coder-routerserver
```


**配置说明**：

- **`enginehttp`**：`rollout.sglang.serving_mode: worker_http`（进程内 HTTP）。
- **`routerserver`**：`router_server`（子进程 SGLang + Router，端口见 yaml）。

**评估模式**：

```bash
bash run_spider.sh eval=true eval_checkpoint_dir=/path/to/hf_or_converted_ckpt
```

## 主要文件

| 文件 | 说明 |
|------|------|
| `main.py` | 训练、测试入口 |
| `run_spider.sh` | 启动封装脚本：配好环境并执行 `main.py` |
| `sql_agent.py` | agent 定义 |
| `config/*.yaml` | 训练超参、SGLang、数据路径 |
| `spider_eval/` | 判分工具 |
