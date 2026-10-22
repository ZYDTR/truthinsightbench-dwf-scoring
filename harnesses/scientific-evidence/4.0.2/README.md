# Scientific Evidence 4.0.2：做题 Agent 的共享上下文

这是 2026-10-21 实验使用的 **V4.0.2 冻结原版**，包含做题 Agent 读取的系统提示词、项目薄入口、索引、核心 Skill 和全部八个参考模块。13 个原始文件逐字保留，版本与 SHA-256 见 [发布清单](publication_manifest.json)。后续实验已有 V5；本目录按指定版本归档，不把 V5 混入 V4。

本包是科研 Harness 中的参与者上下文层。模型服务、SAP/OpenCode 运行环境、外层操作者、实验控制脚本和评分器由宿主环境提供；本包不包含这些组件，也不包含历史科研答案或运行日志。

## 入口与完整结构

- [系统提示词](system_prompt.md)：主侧职责、提前委派、真实验证、执行证据与任务边界。
- [项目 CLAUDE.md](project/CLAUDE.md)：做题工作区的薄入口。
- [上下文索引](project/context/INDEX.md)：连接公共任务材料与核心 Skill。
- [核心 Skill](project/.claude/skills/scientific-evidence/SKILL.md)：完成标准和按需读取路由。

```text
4.0.2/
├── system_prompt.md
├── publication_manifest.json
└── project/
    ├── CLAUDE.md
    ├── context/
    │   ├── INDEX.md
    │   └── delegation.md
    └── .claude/skills/scientific-evidence/
        ├── SKILL.md
        └── references/
            ├── question-coverage.md
            ├── delegation-contract.md
            ├── analysis-state.md
            ├── data-design.md
            ├── inference.md
            ├── separated-validation.md
            ├── verification.md
            └── delivery.md
```

读取关系为 `CLAUDE.md → context/INDEX.md → SKILL.md → 按职责选择参考模块`。`context/delegation.md` 另提供委派合同入口。Markdown 链接用于定位，不代表模块正文已自动加载。

## 放入一个干净的任务环境

1. 建立独立于本仓库的任务目录，只放该题公共材料。Life_03 的公共材料可从仓库内 [clean_tasks/Life_03_nutrient_response_kinetics](../../../clean_tasks/Life_03_nutrient_response_kinetics/README.md) 获取，其 `workspace/` 内容应放在任务根目录。
2. 将本目录的 `project/` **全部内容，包括隐藏的 `.claude/`** 复制到任务根目录。保留目录层级；已有同名上下文时先人工合并或使用全新目录。
3. 将 `system_prompt.md` 正文加入宿主提供的系统提示或固定上下文入口。原实验通过用户级 `CLAUDE.md` 注入这段正文；这不等于替换宿主原生 system preset。单独把该文件放在磁盘上不会自动生效。
4. 以原题 `LaunchPrompt.md` 启动任务，要求执行者实际读取项目入口与索引，再按职责加载模块。研究过程写入 `working/`，最终交付按公共契约写入 `output/`。

任务根目录应提供 `ResearchObjective.md`、`DATA_GUIDE.md`、`PublicTaskContract.md`、`data_manifest.json`、`LaunchPrompt.md`、`validate_output.py` 和原始数据。它们属于各题输入，不由共享 Skill 生成。

宿主需要文件读写、实际分析执行能力，以及原生 `general-purpose` 子 Agent 工具。原实验主、子模型均为 Qwen3.7 Flash；本包不包含模型或密钥。原生委派最多四个并发，只允许一层；工具不可用时按原文记录未覆盖条件。具体数值分析依赖由题目和选用方法决定，本包本身只有 Markdown，没有需要额外安装的软件依赖。

**做题环境只接收该题公共输入与本上下文包。** 本仓库其他目录包含历史提交和评测资产，不应把整个仓库作为参与者工作区。

## 版本与验收范围

来源提交为 `ef5f1d78ffde83662df42da813a250360a5f8ef1`，来源标签为 `tib-scientific-context-v4.0.2-20261021`。本次仅发布已有版本，未修改 Skill 或系统提示词，也未重新运行模型、科学分析或评分。

历史运行确认了索引读取和部分要求的自主落实，同时存在提前委派遗漏、验证类型误标及执行记录不可靠等问题。此版本保留为可复查、可复用的实验上下文，不表示所有科学验收标准已稳定满足。

发布检查覆盖原版字节一致性、索引链接、干净目录与公共任务材料的组合，以及凭据和历史资产保护；这些检查不替代真实 Agent 的行为验收。
