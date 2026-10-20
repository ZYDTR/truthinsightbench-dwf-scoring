# Teach Through Breakdown

把复杂主题讲成读者能复述的因果链，并交付专属术语卡、必要的卡内拓扑和独立 HTML 阅读器。Agent 入口是 [SKILL.md](SKILL.md)。这是 2026-10-20 的可移植发布副本。

## 使用

完整复制本目录到目标环境的 Skill 目录，或直接让 Agent 读取本目录的 `SKILL.md`。调用时明确说：

> 使用 Teach Through Breakdown，把这个主题写成能让我理解并复述的解释，提供术语卡和独立 HTML 阅读器。

需要 **Node.js 20+** 和 **Python 3.10+**。HTML 生成器所需的 `marked`、写作工作流和 Python 文风校验器均已放入 `vendor/`，不需要 npm/pip 安装，也不需要访问原工作区。浏览器交互检查和独立读者验收由实际执行环境提供。

从本目录运行：

```bash
node scripts/build_term_card_reader.mjs --self-test
node scripts/build_term_card_reader.mjs --markdown path/to/article.md --cards path/to/term_cards.json --output path/to/article.html
python3 -B scripts/validate_explainer_bundle.py path/to/bundle_manifest.json --json
python3 -B scripts/lint_external_prose.py path/to/article.md --fail-on hard
```

输入 schema、manifest 和产物要求见 [输出合同](references/output-contract.md)。参数中的相对路径以当前命令目录为基准；manifest 内的相对路径以 manifest 所在目录为基准。

## 内容与依赖

- `SKILL.md`：目标、触发边界、解释方法、产物和验收标准。
- `references/`：术语方法、输出合同、验收合同、视觉设计和独立环境约束。
- `scripts/`：HTML 生成器、产物校验器和文风校验入口。
- `agents/openai.yaml`：原有 Agent 展示元数据。
- `vendor/marked/`：`marked` 17.0.5 ESM 运行文件、source map、包元数据和 MIT 许可证；无运行时传递依赖。
- `vendor/writing-skill/`：固定提交的写作流程、校验源码、原有测试、说明和 MIT 许可证。
- `publication_manifest.json`：文件哈希、原 Skill 文件哈希与必要的移植调整。

写作依赖来源：[grapeot/writing-skill](https://github.com/grapeot/writing-skill/tree/c68f46620711e10f2c16f1f39bfab5badcbae59f)。Markdown 解析器来源：[marked 17.0.5](https://www.npmjs.com/package/marked/v/17.0.5)。两个依赖的原始许可证都保留在各自目录。

## 与工作区版本的差异

教学解释方法、卡片 schema、渲染实现和验收标准保留。公开副本调整了脚本运行路径，移除了本机专用的依赖回退路径，优先使用随包 `marked`；原工作区条件引用由 `references/portable-context.md` 承接，External 写作流程和 prose lint 已完整随包提供。`MARKED_MODULE` 仍可显式覆盖解析器，使用绝对路径或 file URL。

另外修复生成器经符号链接路径启动时静默跳过 CLI 的入口判断；`python3 -B -m unittest discover -s tests -v` 可运行这项回归检查。本地工作区的原 Skill 保持原样，此修复属于公开副本。

发布检查验证脚本可运行和数据合同，不能替代具体文章的浏览器验收或独立冷读。此包不自动启动研究、科学解题或评分。

这个仓库另有历史科学答案和评分资产。需要在干净解题环境使用本 Skill 时，只复制本目录和已授权的原题输入，保持历史答案与 evaluator 材料在环境外。
