# RULER_Trainer
Scripts for train agents for ruler

## 数据准备（Data）

从本仓库的 **GitHub Releases** 页面下载 `data.zip`，解压到 `datamaker/` 目录：

```bash
unzip data.zip -d datamaker/
```

> 说明：`data.zip` 内部有一个 `data/` 外层目录，解压后文件会落在
> `datamaker/data/`（包含 `squad.json`、`hotpot.json`、`essay.json`，约 65MB）。
> `qa_squad` / `qa_hotpot` 依赖 `squad.json` / `hotpot.json`；若缺失，
> pipeline 打印 warning 并自动跳过该任务（数据生成与训练都不会失败）。

## 快速开始

```bash
bash start.sh
```
