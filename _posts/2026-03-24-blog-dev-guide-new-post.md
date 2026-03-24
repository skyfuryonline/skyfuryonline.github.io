---
layout: post
title: "博客开发指南 (四)：VS Code 极速创作与发文指南"
subtitle: "用自定义 Snippets 告别繁琐的 Front Matter"
date: 2026-03-24 10:00:00 +0800
author: "skyfury"
header-img: "img/cover-zaji.png"
catalog: true
tags:
  - Workflow
  - VS Code
---

对于写博客来说，最大的阻力往往是每次都要手敲那一段复杂的头部信息（Front Matter）。为了让你更专注于内容创作，本仓库深度定制了 **VS Code** 的工作流。

## 打破 Markdown 限制
VS Code 默认在 Markdown 文件中禁用了代码自动提示功能。我们在本项目的 `.vscode/settings.json` 中配置了白名单，开启了提示功能。

## 见证奇迹的 `!post`

我们在 `.vscode/markdown.code-snippets` 中注册了一段专属魔法指令。
现在，你想写一篇新文章，只需两个步骤：

1. **新建文件**：在 `_posts/` 目录下创建一个文件，例如 `2026-03-24-hello-world.md`。
2. **触发魔法**：在文件开头输入 `!post`，按回车或 Tab 键。

你会发现，下面这一大段信息瞬间就生成好了：
```yaml
---
layout: post
title: "Your Title Here"
subtitle: "Optional Subtitle"
date: 2026-03-24 12:00:00 +0800
author: "skyfury"
header-img: "img/post-bg-os-metro.jpg"
catalog: true
group: tech
tags:
  - Tag1
---
```

**而且它有极其智能的交互：**
- **自动日期**：`date` 字段会自动填入当前的真实时间。
- **快速补全跳转**：光标会自动停在 `Your Title Here` 处等你输入；写完按一下 `Tab`，光标就会自动跳到 `group`，而且会弹出一个下拉菜单，让你直接在键盘上用上下箭头选择（如 `tech`, `vision`, `llm` 等）！

有了这套工作流，每次创作的启动成本被降到了绝对的零。
        