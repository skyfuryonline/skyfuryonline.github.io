# LH的博客

这是一个基于 Jekyll 和 GitHub Pages 搭建的个人博客，用于记录博主在NLP领域的学习、研究以及生活点滴。

## 网站结构

本博客的结构如下：

- `_config.yml`: 网站的全局配置文件，用于设置网站的标题、作者、主题、社交链接等。
- `_posts`: 存放所有博客文章的文件夹，文章格式为 Markdown。
- `_layouts`: 存放网站的布局模板，决定了不同类型页面的外观。
- `about.html`: “关于我”页面。
- `index.html`: 网站首页，展示最新的博客文章列表。
- `his.md`: 记录了本博客的修改历史。
- `img`: 存放网站中使用的图片，例如头像、文章配图等。
- `css`, `js`, `fonts`, `less`: 存放网站的样式、脚本和字体等静态资源。

## 参考

本博客的主题修改自以下开源项目，感谢原作者的分享：

- **[qiubaiying/qiubaiying.github.io](https://github.com/qiubaiying/qiubaiying.github.io)**

## 如何在本地运行

1.  **安装 Ruby 和 Jekyll:**
    请参考 Jekyll 官方文档：[https://jekyllrb.com/docs/installation/](https://jekyllrb.com/docs/installation/)

2.  **安装依赖:**
    在项目根目录下运行 `bundle install`。

3.  **启动本地服务:**
    运行 `bundle exec jekyll serve`，然后通过浏览器访问 `http://localhost:4000`。
