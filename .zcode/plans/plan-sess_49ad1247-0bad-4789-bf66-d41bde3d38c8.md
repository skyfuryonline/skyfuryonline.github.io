# 博客改进方案：死代码清理 + 工程升级 + 基建补全

## 一、死代码大扫除（公开仓库为主）

每个文件删除前 grep 全库验证零引用：
1. **css（6 个，~310KB）**：bootstrap.css、hux-blog.css、daily-report.css、gwy-report.css、jzxm.css、tuitui.css（页面实际用的都是 .min 版或内联样式）
2. **js（6 个，~400KB）**：jquery.js、bootstrap.js、hux-blog.js、animatescroll.min.js、cat-follow.js、time.js
3. **`less/` 目录**（36KB，hux-blog.css 的过时源码）
4. **孤儿 layout**：`_layouts/keynote.html`（零页面使用）
5. **图片孤儿**：cover_algorithm.webp（129KB，分组已不存在）、bg-circuit.svg、icon-glitch.svg、neon-divider.svg；**img/gwy/yeyou_1.jpg + yeyou_2.jpg（7.4MB，上次漏删，CI 会用 blog-source 的压缩版覆盖）**
6. **Gitalk 死代码**：post.html 中被 `enable: false` 永久关闭的评论块 + js/md5.min.js（配置里保留 gitalk 段与说明，想开评论换 giscus）
7. **禁用功能残留**：公开仓库的 weekly_report.yml、monthly_report.yml（if:false）、`_layouts/gwy_log_entry.html`；blog-source 的 `_gwy_reports/`（25 个文件）
8. **文档修正**：README 里不存在的 `{% include group_posts.html %}` 教程改为实际机制；.vscode snippet 里已删除分组的枚举项

## 二、工程升级

1. **Jekyll 3.9 → 4.4**：Gemfile + _config.yml 调整（jekyll-paginate 兼容保留），Ruby 3.1 已满足。失败回滚预案：CI 红则 revert（旧站点保持在线，Pages 保留最后一次成功部署）
2. **Gemfile.lock**：用 docker ruby:3.1 生成（本机 ruby 2.6 太老）；docker 不可用则跳过并在 SKILL.md 记录原因
3. **daily.html 拆分（零视觉改动）**：
   - 4 个内联 `<style>` 块 → `css/daily.css`、`css/daily-modals.css`（带 `?v=构建时间` 缓存戳）
   - 主 `<script>` → `js/daily.js`，放在原位置以普通同步 script 加载（行为一致）
   - Liquid 标记与数据注入保留在页面里；预计 2595 行 → ~700 行标记 + 2 css + 1 js
   - 部署后逐板块核对（时钟/影视/Steam/情报/新闻 tab/图推/成语/像素场景/jzxm 彩蛋）

## 三、基建补全

1. **Search Console**：给你三步手动指引（验证所有权 → 提交 sitemap.xml → 请求移除已收录的 jzxm 旧 URL）
2. **CDN 整理（低风险收敛）**：
   - 删除 oss.maxcdn 的 html5shiv/respond.js（IE8 垫片，2026 年纯死重）
   - 核查 head.html 的 Google Fonts http:// 引用（在注释里则清理注释，在活动代码则改 https 或移除）
   - fastclick 1.0.6（现代浏览器不需要）评估后移除
   - 在用的 KaTeX、Chart.js、marked、font-awesome 保持不动
3. 备份不额外做（用户决定以 blog-source 仓库为备份）

## 四、执行顺序与验证

1. 公开仓库死代码清理 → push → 部署绿 + 首页/文章/tags/daily/gwy 各抽查 200 与内容
2. blog-source `_gwy_reports` 删除 → push（触发部署验证）
3. daily.html 拆分 → push → 部署绿 + 线上板块核对（请你也过目一眼）
4. Jekyll 4 升级（+lock）→ push → 部署绿 + 全页面抽查 + sitemap 条目数对比（应仍为 386）
5. CDN 清理 → push → 验证
6. 更新 SKILL.md（记录 Jekyll 4、文件结构变化、giscus 备注）
7. 每个 push 走仓库级代理；分逻辑 commit

## 不做的事（明确出界）
- 数据链路体检（daily 断供排查与告警）——用户选择跳过
- 任何视觉/布局改动
- Keynote layout 删除后不补替代（零使用）