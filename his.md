2025年10月30日

- **项目初始化与主题更换:**
  - 删除了旧的博客主题文件，并从 `qiubaiying.github.io-master` 仓库中引入了新的主题。

- **个性化配置与修改:**
  - **博客标题:** 更新为 “LH的博客”。
  - **背景图片:** 分别更新了主页、About页面和Tags页面的背景图片。
  - **个人信息:** 更新了网站描述、个人简介和头像等。
  - **社交链接:** 删除了无需的社交平台链接。
  - **内容清理:** 删除了模板自带的旧文章和 `friends` 列表。

- **新增内容:**
  - `_posts/2025-10-30-helloworld.md`: 创建了第一篇博客，内容为“如何发布新博客及管理图片”的教程。
  - `README.md`: 创建了项目介绍文件，并添加了网站截图。
  - `his.md`: 创建了本文件用于记录修改历史。

- **系统功能调整:**
  - **分析服务:**
    - `_config.yml`: 注释了百度统计和谷歌分析的配置，为后续替换为自定义统计功能做准备。
  - **评论系统:**
    - **问题记录:** 发现并记录了 `Gitalk` 评论系统因未在 GitHub 上创建对应的 `OAuth App` 并正确配置 `Client ID` 和 `Client Secret`，导致出现 `redirect_uri` 相关的报错。

2025年11月2日

- **主页UI重构 (第一阶段):**
  - **目标:** 将主页的线性文章列表重构为“大卡片套小卡片”的模块化布局，以“主题”对文章进行分组展示。
  - **实现方式:**
    - 创建了 `_data/homepage_groups.yml` 数据文件，用于独立定义主页的卡片分组、标题、封面和包含的文章，实现了布局与内容的解耦。
    - 重写了 `index.html`，使其通过读取上述数据文件来动态生成卡片布局。
    - 使用 `CSS Flexbox` 进行两列布局，并编写 `jQuery` 脚本实现卡片的点击展开/折叠交互效果。
  - **遇到的问题及解决:**
    1.  **文章不显示:** 最初通过 `post.name` 匹配文章失败。后通过植入诊断代码，发现 `post.name` 变量在当前环境无效。最终改为通过 `post.path` 截取文件名进行匹配，成功解决。
    2.  **卡片联动展开:** 最初的 `jQuery` 选择器不够精确，导致点击一个卡片会错误地影响其他卡片。最终改为通过 `data-target` 属性和唯一的 `ID` 进行点对点操作，彻底解决问题。
    3.  **展开时布局错位:** 展开一个卡片时，同行未展开的卡片被拉伸出现空白。通过为 `Flexbox` 容器添加 `align-items: flex-start;` 样式，成功解决。

- **主页UI重构 (第二阶段):**
  - **目标:** 将卡片点击后的“原地展开”交互模式，升级为“跳转到独立主题页面”的模式，使网站结构更清晰。
  - **实现方式:**
    - 在 `_config.yml` 中注册了 `groups` 集合，并创建了 `_groups` 文件夹。
    - 为每个主题创建了对应的 Markdown 文件（如 `llm.md`, `life.md`）和布局文件 `_layouts/group_page.html`。
    - 通过修改布局文件，实现了在不依赖插件的情况下，主题页面也能自动查找并显示其包含的文章列表。
    - 修改了 `index.html`，将大卡片链接到对应的主题页面。
  - **遇到的问题及解决:**
    1.  **文章背景图丢失:** 在为文章页统一主题背景图时，`Liquid` 逻辑未能正确生成URL。最终通过在 `_layouts/post.html` 中重构逻辑，优先计算出最终的图片路径再赋值给 `style` 标签，成功解决。
    2.  **链接逻辑错误:** 在重命名主题卡片后，忘记同步更新 `index.html` 中的 `if/elsif` 判断条件，导致链接失效。通过修正 `index.html` 中的判断逻辑解决。

- **其他:**
  - 更新了 `404.html` 页面的背景图片。

---

### **附录：如何新建一个“大卡片”主题及第一篇笔记**

这是一个完整的操作流程，严格按照步骤操作即可。

**场景假设:** 我们要新建一个名为 `“PyTorch学习”` 的主题，它的英文ID是 `pytorch`，并为其添加第一篇名为 `“Tensor基础”` 的笔记。

**第一步：准备图片和文章**

1.  **准备主题封面图**: 准备一张用作“PyTorch学习”这个大卡片封面的背景图片。将它命名为 `cover-pytorch.png` 并放入项目根目录的 `img/` 文件夹下。
2.  **准备文章**: 将您写好的 `“Tensor基础”` 这篇笔记准备好，文件名为 `2025-11-03-tensor-basics.md`。

**第二步：修改 `_data/homepage_groups.yml` 文件**

这是最关键的一步，用于在主页上“注册”新的大卡片，并把文章放进去。

打开该文件，在末尾添加一个新的 `group` 定义：

```yaml
- group_name: "PyTorch学习"
  group_image: "/img/cover-pytorch.png"
  group_description: "PyTorch框架的学习与实践笔记。"
  posts:
    - "2025-11-03-tensor-basics.md"
```

**第三步：创建主题页面**

1.  **创建Markdown文件**: 在项目根目录的 `_groups` 文件夹下，创建一个新的文件，**文件名必须是这个主题的英文ID**，即 `pytorch.md`。
2.  **填入内容**: 打开新建的 `pytorch.md`，填入以下两行即可：
    ```yaml
    ---
    layout: group_page
    title: "PyTorch学习"
    ---
    ```
    这里的 `title` 必须和您在第二步中设置的 `group_name` 完全一致。

**第四步：修改 `index.html` 的链接逻辑**

打开 `index.html`，找到 `Liquid` 的 `if/elsif` 判断链，在 `{% endfor %}` 前添加一个新的 `elsif` 分支：

```liquid
{% raw %}
        {% elsif group.group_name == "杂记" %}
            {% assign group_id = "life" %}
        {% elsif group.group_name == "PyTorch学习" %}
            {% assign group_id = "pytorch" %}
        {% endif %}
{% endraw %}

        <a href="{{ site.baseurl }}/groups/{{ group_id }}" ...>
```

**第五步：上传文章**

将您准备好的 `2025-11-03-tensor-basics.md` 文件放入项目根目录的 `_posts` 文件夹下。

**第六步：提交所有更改**

完成以上所有步骤后，将所有新建和修改的文件提交到您的 GitHub 仓库即可。

---

- **参考链接:**
  - [博客搭建详细教程](https://github.com/qiubaiying/qiubaiying.github.io/wiki/博客搭建详细教程)
