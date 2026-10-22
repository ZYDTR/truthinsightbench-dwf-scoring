# 能力与文件索引

| 需要 | 入口与用途 |
|---|---|
| 当前题目的科学问题、数据与输出要求 | 项目根 ResearchObjective.md、DATA_GUIDE.md、PublicTaskContract.md、data_manifest.json；按 LaunchPrompt.md 的要求读取 |
| 共享科研能力 | [.claude/skills/scientific-evidence/SKILL.md](../.claude/skills/scientific-evidence/SKILL.md)：主 Agent 分析前及子 Agent 接受科研委派时读取；核心内索引到问题覆盖、研究设计、竞争解释、分离验证、复核、交付、委派和状态模块 |
| 恢复主侧工作 | working/analysis-state.md：由主 Agent 建立的当前事实与权威产物导航，按记录定位原件 |
| 公共交付格式 | 项目根 validate_output.py：按 PublicTaskContract.md 运行，检查范围是格式 |

路径相对项目根。模块路径只有核心 Skill 维护；无需一次加载全部模块。路径不存在时保留准确错误并在当前项目定位。
