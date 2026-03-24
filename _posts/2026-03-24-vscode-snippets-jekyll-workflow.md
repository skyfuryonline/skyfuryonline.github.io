---
layout: post
title: "博客开发指南 (二)：极速创作流与 VS Code 代码片段自动化"
subtitle: "再也不用手敲 Front Matter：用 `!post` 开启你的创作灵感"
date: 2026-03-24 10:00:00 +0800
author: "skyfury"
header-img: "img/post-bg-universe.jpg"
catalog: true
group: tech
tags:
  - VS Code
  - Workflow
  - Productivity
  - Jekyll
---

在构建了现代化的 Jekyll 博客后，写博客最大的阻力往往变成了繁琐的“文件头”（Front Matter）。

本博客使用了多分组（Tech, Life, Vision 等）、多标签以及侧边栏目录（`catalog: true`），每次新建 Markdown 文件都要写一长串 YAML 配置。为了解决这个问题，本站深入定制了 VS Code 的工作流。

## VS Code Markdown 限制与突破

VS Code 默认在 Markdown 文件中是**关闭** Quick Suggestions（快捷提示）的。这是为了避免在正常打字时频繁弹出代码提示框。

为了在博客目录开启快捷指令，我们在项目的 `.vscode/settings.json` 中进行了特例配置：
```json
{
  "[markdown]": {
    "editor.quickSuggestions": {
      "other": true,
      "comments": false,
      "strings": false
    }
  }
}
```

## 自定义 Snippets：一键生成标准模板

接下来，在 `.vscode/markdown.code-snippets` 中，我们定义了专属的代码片段：
- **前缀 (prefix)**：输入 `!post` 或 `!gwy` 即可触发。
- **动态变量**：使用 `$CURRENT_YEAR-$CURRENT_MONTH-$CURRENT_DATE` 自动生成今天的日期。
- **下拉选择**：针对文章分组，使用 `${1|tech,life,vision,gwy|}` 语法，可以在插入代码片段后，直接按上下方向键选择属于哪个专栏！

```json
"Jekyll Post Front Matter": {
    "prefix": ["!post", "jekyll-post"],
    "body": [
      "---",
      "layout: post",
      "title: \"${2:Your Title Here}\"",
      "subtitle: \"${3:Optional Subtitle}\"",
      "date: $CURRENT_YEAR-$CURRENT_MONTH-$CURRENT_DATE $CURRENT_HOUR:$CURRENT_MINUTE:$CURRENT_SECOND +0800",
      "author: \"skyfury\"",
      "header-img: \"img/post-bg-os-metro.jpg\"",
      "catalog: true",
      "group: ${1|tech,life,vision|}",
      "tags:",
      "  - ${4:Tag1}",
      "---",
      "",
      "$0"
    ]
}
```

**如何使用？**
只需新建一个后缀名为 `.md` 的文件，输入 `!post` 并回车，整个标准的博客头部就自动写好了，光标会停留在 Title 处等待输入，按 Tab 键自动跳转到下一个待填字段，极大提升了创作体验！
        