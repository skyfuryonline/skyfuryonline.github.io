---
layout: post
title:  "博客部署迁移：从Branch到Actions"
subtitle: "一次充满“惊喜”的CI/CD实践"
date:   2025-11-03 10:00:00 +0800
author: "LH"
tags:
  - CI/CD
  - Jekyll
  - GitHub Actions
---

为了给博客集成更强大的自动化功能（例如定时爬虫），我决定将博客的部署方式从传统的“从分支部署 (Deploy from a branch)”切换到更现代、更灵活的“GitHub Actions”。

整个过程比预想的要曲折，但也因此积累了宝贵的排错经验。本文旨在复盘整个迁移过程，记录遇到的每一个问题和最终的解决方案。

### 第一阶段：创建基础工作流

最初的目标很简单：创建一个 GitHub Actions 工作流，让它能自动构建和部署 Jekyll 网站。然而，这个看似简单的目标，却引发了一连串的“构建失败”。

### 问题一：`Could not locate Gemfile`

**原因**：GitHub Actions 的运行环境是一个“无菌”的虚拟机，它不像我的本地电脑，不知道需要安装 Jekyll。`Gemfile` 文件就像一个“购物清单”，必须提供给它，它才知道需要安装什么。

**解决方案**：在项目根目录创建 `Gemfile`，并列出所有必需的依赖，如 `jekyll`, `jekyll-paginate` 等。

### 问题二：`Invalid date` in Jekyll's own template

**原因**：这是 Jekyll 3.9.x 版本的一个已知 bug。在干净的环境中安装时，它会尝试处理自己内部的一个模板文件，但该文件自身的日期格式却不符合 Jekyll 自己的校验规则。

**解决方案**：在 `_config.yml` 中，将 `bundler` 安装 gems 的 `vendor/` 目录排除掉，不让 Jekyll 去扫描它。

### 问题三：`cannot load such file -- kramdown-parser-gfm`

**原因**：我的 `_config.yml` 中配置了使用 `GFM` (GitHub Flavored Markdown)，这需要一个名为 `kramdown-parser-gfm` 的额外插件。而我的 `Gemfile` “购物清单”上又漏掉了它。

**解决方案**：在 `Gemfile` 中补上 `gem "kramdown-parser-gfm"` 这个缺失的依赖。

### 问题四：`Permission to ... denied` (Error 403)

**原因**：GitHub Actions 默认生成的 `GITHUB_TOKEN` 只有**读取**仓库的权限。而我最初使用的 `peaceiris/actions-gh-pages` 这个 Action 需要**写入**权限，因为它要将构建好的网站推送到 `gh-pages` 分支。

**解决方案**：在 `deploy.yml` 工作流文件中，为部署任务明确授予 `contents: write` 的权限。

---

### 第二阶段：最棘手的问题——构建成功，但网站不更新

在解决了所有“构建失败”的红叉后，我遇到了一个更诡异的问题：**Actions 列表里显示构建和部署都成功了（绿色对勾），但我的线上博客页面却没有任何变化。**

**原因分析**：
经过检查，我发现 `Deployments` 列表中并没有新的部署记录。这说明，虽然我的工作流“成功跑完”了，但它和 GitHub Pages 服务之间没有建立起真正的连接。我使用的 `peaceiris/actions-gh-pages` 只是将文件推送到了 `gh-pages` 分支，但我的 GitHub Pages 服务并没有被配置为从这个分支读取内容。

**最终解决方案：改用官方推荐的部署方式**

我决定彻底放弃 `peaceiris/actions-gh-pages`，转而使用 GitHub 官方推荐的、更现代的部署 Action 组合：

1.  **`actions/upload-pages-artifact`**: 这个 Action 负责将构建好的 `_site` 目录打包成一个标准的“部署构件 (Artifact)”。
2.  **`actions/deploy-pages`**: 这个 Action 负责通知 GitHub Pages 服务：“请直接拉取我刚刚上传的那个构件，并用它来部署网站。”

同时，还需要在工作流的顶层 `permissions` 中，为 `pages` 和 `id-token` 授予 `write` 权限，以允许工作流能向部署服务证明自己的身份。

```yaml
# deploy.yml
permissions:
  contents: read
  pages: write      # 允许写入部署页面
  id-token: write   # 允许进行身份验证

jobs:
  build:
    # ... 构建步骤
    - name: Upload artifact
      uses: actions/upload-pages-artifact@v3 # v3是必须的
      with:
        path: './_site'

  deploy:
    needs: build
    # ... 部署步骤
    - name: Deploy to GitHub Pages
      uses: actions/deploy-pages@v4 # v4是必须的
```

在切换到这个方案后，我又遇到了几个因为 Action 版本过时而导致的报错，将它们全部升级到最新版（`v3` 和 `v4`）后，部署终于成功了！

### 总结

这次迁移过程虽然坎坷，但收获巨大。它让我深刻理解了 GitHub Actions 的权限模型、环境依赖和部署机制。如果你也想从传统的分支部署切换到 Actions，希望这篇“踩坑笔记”能对你有所帮助。