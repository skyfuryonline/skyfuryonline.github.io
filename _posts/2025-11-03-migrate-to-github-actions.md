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

最初的目标很简单：创建一个 GitHub Actions 工作流，让它能自动构建和部署 Jekyll 网站。

1.  **创建工作流文件**: 在项目根目录下创建 `.github/workflows/deploy.yml`。
2.  **编写基础流程**: 工作流的核心步骤包括：
    *   `actions/checkout@v3`: 检出代码。
    *   `ruby/setup-ruby@v1`: 设置 Ruby 环境。
    *   `bundle exec jekyll build`: 构建网站。
    *   `peaceiris/actions-gh-pages@v3`: 将构建产物（`_site` 目录）部署到 `gh-pages` 分支。

然而，第一次运行就失败了。

### 问题一：`Could not locate Gemfile`

**错误日志**:
```
Run bundle exec jekyll build
Could not locate Gemfile or .bundle/ directory
```

**原因分析**:
这是一个典型的环境差异问题。我的本地环境已经全局安装了 Jekyll，但 GitHub Actions 的运行环境是一个“无菌”的虚拟机，它不知道需要安装任何工具。

**解决方案**:
在项目根目录下创建一个 `Gemfile` 文件。这个文件像一个“购物清单”，告诉 Actions 的 Ruby 环境需要安装哪些依赖。

```ruby
# Gemfile
source "https://rubygems.org"

gem "jekyll", "~> 3.9.0"
gem "jekyll-paginate"
gem "tzinfo-data"
```

### 问题二：`Invalid date` in Jekyll's own template

**错误日志**:
```
Invalid date ... in ... welcome-to-jekyll.markdown.erb
```

**原因分析**:
这是 Jekyll 3.9.x 版本的一个已知 bug。在干净的环境中安装时，它会尝试处理自己内部的一个模板文件，但该文件自身的日期格式却不符合 Jekyll 自己的校验规则。

**解决方案**:
在 `_config.yml` 中，将 `bundler` 安装 gems 的 `vendor/` 目录排除掉，不让 Jekyll 去扫描它。同时，根据新版 Jekyll 的建议，将 `gems:` 配置项重命名为 `plugins:`。

```yaml
# _config.yml
exclude: ["less", "node_modules", "Gruntfile.js", "package.json", "README.md", "vendor/"]
plugins: [jekyll-paginate]
```

### 问题三：`cannot load such file -- kramdown-parser-gfm`

**错误日志**:
```
Dependency Error: Yikes! It looks like you don't have kramdown-parser-gfm or one of its dependencies installed.
```

**原因分析**:
我的 `_config.yml` 中配置了使用 `GFM` (GitHub Flavored Markdown)，这需要一个名为 `kramdown-parser-gfm` 的额外插件。而我的 `Gemfile` “购物清单”上漏掉了它。

**解决方案**:
在 `Gemfile` 中补上这个缺失的依赖。

```ruby
# Gemfile
# ... (other gems)
gem "kramdown-parser-gfm"
```

### 问题四：`Permission to ... denied` (Error 403)

**错误日志**:
```
remote: Permission to skyfuryonline/skyfuryonline.github.io.git denied to github-actions[bot].
fatal: unable to access '...': The requested URL returned error: 403
```

**原因分析**:
这是最后一个，也是最关键的一个问题。GitHub Actions 默认生成的 `GITHUB_TOKEN` 只有**读取**仓库的权限。而部署操作（推送到 `gh-pages` 分支）需要**写入**权限。

**解决方案**:
在 `deploy.yml` 工作流文件中，为部署任务明确授予写入权限。

```yaml
# .github/workflows/deploy.yml
jobs:
  build_and_deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write # <--- 授予写入权限
    steps:
      # ...
```

### 最终步骤：切换部署源

在所有构建问题都解决后，最后一步是在 GitHub 仓库的 `Settings > Pages` 中，将部署源从 `Deploy from a branch` 修改为 `GitHub Actions`。

至此，整个迁移过程宣告完成。虽然过程一波三折，但每一个错误都加深了我对 CI/CD 环境和 Jekyll 构建机制的理解。
