# 输出合同

## 目录

- 标准产物
- Internal 与 External 的差别
- 术语卡 JSON
- 验收 manifest
- HTML 构建命令
- 文件布局示例

## 标准产物

根据用户要求选择单文档或双文档。完整双版本交付包含：

1. Internal canonical Markdown：技术对象、状态、命令、证据和失败传播完整。
2. External canonical Markdown：面向无共享上下文读者，以发现、证据和认识修正推进。
3. term decision ledger：逐文 KEEP、DELETE、RELATE、关系组与解释负担。
4. 每篇独立的 term cards JSON。
5. 每篇独立、自包含的 HTML 阅读器。
6. deterministic acceptance 输出。
7. Internal 内容回归、External prose/冷读、卡片冷读和浏览器检查报告。

用户只要求一个版本时仍保留对应的术语账本、卡片、HTML 与验收证据。

## Internal 与 External 的差别

Internal 假设读者共享项目背景，但不假设读者掌握本轮新概念。它强调对象身份、数据形状、执行空间、状态机、参数范围、命令输入输出、产物过期条件、决定权和证据强度。

External 假设读者没有项目背景。它先建立具体问题，让案例证据逐步修正直觉；术语跟在动作之后；主线可以用日常语言复述；作者姿态保持为分享发现的实践者。

两篇可以共享概念身份，但正文入口、别名范围、卡片深度和最终卡片数允许不同。

## 术语卡 JSON

使用对象映射，便于稳定引用和检测别名冲突：

```json
{
  "meta": {
    "title": "阅读器标题",
    "audience": "面向只有基础计算机知识的读者"
  },
  "cards": {
    "runtime": {
      "label": "Runtime",
      "aliases": ["Runtime"],
      "mentions": ["运行时"],
      "structure": "composite",
      "topology": {
        "caption": "一次执行怎样在 Runtime 内持续推进",
        "nodes": [
          {"id": "input", "label": "外部输入", "row": 0, "column": 0},
          {"id": "state", "label": "当前执行状态", "row": 1, "column": 0},
          {"id": "rule", "label": "决定下一步的代码", "row": 1, "column": 1},
          {"id": "model_call", "label": "发送模型请求的代码", "row": 2, "column": 0},
          {"id": "tool_call", "label": "执行工具请求的代码", "row": 2, "column": 1},
          {"id": "result", "label": "最终结果", "row": 3, "column": 1}
        ],
        "edges": [
          {"from": "input", "to": "state", "label": "写入"},
          {"from": "state", "to": "rule", "label": "读取"},
          {"from": "rule", "to": "model_call", "label": "需要模型时触发"},
          {"from": "model_call", "to": "state", "label": "写回模型结果"},
          {"from": "rule", "to": "tool_call", "label": "需要工具时触发"},
          {"from": "tool_call", "to": "state", "label": "写回工具结果"},
          {"from": "rule", "to": "result", "label": "满足结束条件"}
        ],
        "groups": [
          {"label": "Runtime", "members": ["state", "rule", "model_call", "tool_call"]}
        ]
      },
      "body": "先沿图解释状态怎样被读取、改变和写回，最后给出 Runtime 的精确边界。"
    }
  }
}
```

约束：

- key 使用稳定的 snake_case 概念身份。
- label 明确指出当前唯一解释对象，不能把多个独立术语并列成一个标题。
- aliases 恰好包含一个元素：该篇正文实际需要的唯一触发词。
- mentions 可选，只放与主词完全同指的大小写、空格、翻译或代码写法变体；它们可检索，但不生成正文下划线。
- structure 必须是 `atomic` 或 `composite`。composite 表示该术语由多个重要对象或反复状态传递共同成立。
- composite 卡必须有 topology；atomic 卡不得用装饰性 topology。
- topology.caption 是图的可访问说明；nodes 使用唯一 id、普通语言 label、非负整数 row/column；edges 引用已有 node 并使用动词 label；groups 用 members 表达包围边界。
- body 是自由长度 Markdown，无固定栏目。
- 大小写归一后的 alias 只能归属一张卡。
- 不同卡的 body 归一化后不能相同；完全相同意味着应保留一个主词并把另一写法移入 mentions。

## 验收 manifest

`scripts/validate_explainer_bundle.py` 接受以下 JSON：

```json
{
  "documents": [
    {
      "name": "internal",
      "markdown": "topic_internal.md",
      "cards": "internal_term_cards.json",
      "html": "topic_internal.html",
      "expected_cards": 28,
      "forbidden_terms": ["Agent", "Prompt"],
      "required_terms": ["Pillow", "PyTorch", "NumPy"],
      "topology_terms": ["Runtime", "Agent Runtime"],
      "distinct_term_groups": [
        ["eval()", "no_grad", "gradient", "backward()", "optimizer"]
      ]
    }
  ]
}
```

相对路径以 manifest 所在目录为基准。`forbidden_terms`、`required_terms`、`topology_terms` 和 `distinct_term_groups` 应来自当前术语账本，不要机械复制示例。`topology_terms` 要求对应卡标记为 composite 且拥有拓扑；`distinct_term_groups` 要求组内每个词存在，并且分别属于不同卡。

从本 Skill 目录运行：

```bash
python3 scripts/validate_explainer_bundle.py \
  path/to/bundle_manifest.json
```

使用 `--json` 输出机器可读报告。

## HTML 构建命令

先对每篇文档分别从本 Skill 目录运行：

```bash
node scripts/build_term_card_reader.mjs \
  --markdown path/to/article.md \
  --cards path/to/term_cards.json \
  --output path/to/article.html \
  --title "标题" \
  --audience "读者说明"
```

`marked` 已包含在 `vendor/marked/`，无需安装。生成器优先读取 `MARKED_MODULE`，然后使用随包版本，最后尝试正常模块解析和当前目录的 `node_modules`。使用环境变量时提供 ESM 文件的绝对路径或 file URL。

生成器提供：

- 左侧目录
- 全部术语搜索
- hover、focus、click、Enter、Space 与 Escape
- 点击固定卡片
- 手机底部 sheet
- 深浅主题与阅读进度
- composite 卡内的自包含拓扑图
- 宽表格、代码块和长卡片内部滚动
- 内嵌 CSS、JavaScript 与卡片数据的独立 HTML

桌面布局使用靠左目录与宽正文：页面 gutter 为 24–52px，目录宽 230–300px，目录与正文的响应式间距为 56–112px，正文通常上限 1180px，超宽屏上限 1260px。避免把目录与 800px 正文整体塞进居中的窄容器。

视觉 token、参考页面提取方法、浅深主题关系和术语入口样式遵循 [阅读器视觉设计合同](reader-visual-design.md)。静止术语保持正文原色和透明背景，仅用低对比度 1px 细实线提供入口；交互时再增强反馈。

## 文件布局示例

下列目录名只是示例，可放在当前项目授权的输出目录；无需原工作区。manifest 中路径始终相对 manifest 文件解析。

```text
contexts/survey_sessions/
  topic_internal_YYYYMMDD.md
  topic_internal_YYYYMMDD.html
  topic_external_YYYYMMDD.md
  topic_external_YYYYMMDD.html

tmp/topic_YYYYMMDD/
  source_contract.md
  term_decision_ledger.md
  internal_term_cards.json
  external_term_cards.json
  bundle_manifest.json
  deterministic_acceptance.json
  internal_content_qa.md
  external_prose_qa.md
  external_terminal_cold_read.md
  term_card_qa.md
  browser_qa.md
```
