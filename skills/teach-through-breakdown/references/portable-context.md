# 独立环境的证据与协作约束

本文件承接原工作区的机制拆解、时间敏感信息验证和子代理护栏，使公开包不依赖原工作区路径。解释方法、输出 schema 和教学验收仍以本 Skill 的四份原有 reference 合同为准。

## 事实与机制证据

从一个具体问题或真实场景组织正文。核心概念需要说明对象形态、状态由谁持有、输入输出、变化发生在哪里以及设计解决的缺口；比喻必须说明适用边界，不能用另一个陌生词代替解释。

核对实现时读取相关源码、接口合同和实际运行证据。关键断言在出现处标明来源，区分官方说明、观察到的行为、案例和推断。代码定义不能自动证明实际运行配置；一次成功也不能自动证明一般保证。示意图围绕当前需要解释的状态变化，并遵守本包的拓扑及浏览器验收合同。

涉及可能变化的版本、API 参数、产品能力或发布日期时，使用当前环境允许的检索工具核对官方文档或原始材料，记录来源与版本/日期。不能因模型记忆中没有某个版本就否定其存在。无法核实时标记未知或未验证；涉及物理量时检查单位、量级和基本约束。

## 子代理边界

只有用户明确授权时，顶层主 Agent 才能委派独立冷读等任务。本 Skill 不自行授予调度权限。

直接委派消息必须提供目标、成功标准、允许读取/修改的文件、输出要求和裁剪后的必要上下文。默认不继承完整历史；收到的历史或参考文件只提供背景。

委派消息必须明确接收者是 leaf，只能完成直接分配的工作。禁止继续派生子代理、创建或切换 thread/session、fork worktree、启动 CLI 模型进程或向 sibling 发调度消息。需要追加人手时只能报告主 Agent。

主 Agent 在启动后做一次轻量任务树检查。发现嵌套或横向调度时中断或隔离相应任务，不依赖其结果；主 Agent 亲自读取最终证据。没有独立读者或浏览器验证条件时，保留未完成 Gate，不能用自评冒充通过。

## External 写作依赖

完整写作流程和确定性文风校验已随包提供，固定到 `grapeot/writing-skill` 的提交 `c68f46620711e10f2c16f1f39bfab5badcbae59f`：

- [写作 Router](../vendor/writing-skill/skills/writing_workflows.md)
- [Internal workflow](../vendor/writing-skill/skills/workflow_internal_writing.md)
- [External workflow](../vendor/writing-skill/skills/workflow_external_writing.md)
- [External prose lint 合同](../vendor/writing-skill/skills/external_prose_lint.md)

从本 Skill 目录运行，使用已打包的 Python 源码，无需安装 Python 包：

```bash
python3 -B scripts/lint_external_prose.py path/to/article.md --fail-on hard
```

保留完整 stdout，修复 hard findings 后再执行独立冷读。校验器以中文为主，不把通过机械检查当成读者已理解。完整 Internal / External 写作任务再按需加载对应 workflow；当前 Skill 已定义教学解释需要的正文与卡片合同。
