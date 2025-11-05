# Jekyll `where_exp` 过滤器在 GitHub Pages 环境下的行为报告

**问题背景：**
在 Jekyll 博客的开发过程中，我们多次尝试使用 `where_exp` 过滤器进行数据筛选（例如在 `daily.html` 中筛选 `_data` 目录下的文件，或在 `_layouts/group_page.html` 中筛选文章），但这些操作常常导致页面内容无法正确显示或出现其他兼容性问题。

**核心发现：**

`where_exp` 过滤器在 Jekyll 中是一个非常强大的工具，它允许使用 Liquid 表达式进行复杂的筛选。然而，在 GitHub Pages 环境下，其行为确实存在一些**已知的问题和限制**，尤其是在处理某些数据类型或复杂表达式时。

1.  **GitHub Pages 的 Jekyll 版本限制**:
    *   GitHub Pages 运行的是一个**特定且通常不是最新**的 Jekyll 版本，以及其依赖的 Liquid 版本。这意味着一些在最新 Jekyll 版本中运行良好的特性，在 GitHub Pages 上可能不被支持，或者行为不一致。
    *   `where_exp` 尤其容易受到 Liquid 引擎版本的影响。某些复杂的表达式，或者对 `item` 内部属性的访问，可能在旧版 Liquid 中无法正确解析。

2.  **`where_exp` 的性能与复杂性**:
    *   虽然 `where_exp` 提供了灵活性，但它在内部执行的是一个循环，并且每次迭代都会评估一个 Liquid 表达式。这可能导致性能问题，尤其是在处理大量数据时。
    *   更重要的是，当表达式变得复杂，或者涉及到对 `item` 内部结构（如 `item[0]` 或 `item.some_property`）的访问时，它更容易出错或返回空结果，尤其是在数据结构不完全一致的情况下。

3.  **`site.data` 的特殊性**:
    *   `site.data` 是一个特殊的集合，它包含了 `_data` 目录下所有文件的内容。当您使用 `for file in site.data` 遍历时，`file` 实际上是一个包含 `[文件名, 文件内容]` 的数组。
    *   `where_exp: "item", "item[0] contains 'daily_'" ` 这种写法，虽然在理论上是正确的，但 `item[0]` 这种访问方式在某些 Liquid 版本中可能不够健壮，或者在 `site.data` 的内部表示上存在细微差异。

**结论：**

`where_exp` 过滤器在 GitHub Pages 上的不稳定行为是一个**真实存在的问题**。它不是一个可靠的、跨所有 Jekyll 版本都表现一致的特性。

**推荐的替代方案：**

为了确保代码的稳定性和兼容性，我们应该避免使用 `where_exp` 进行复杂的筛选。

**最佳实践是：**

1.  **使用简单的 `where` 过滤器**: 如果只需要基于一个简单的键值对进行筛选，`where` 过滤器（例如 `site.posts | where: "category", "news"`）通常是安全的。
2.  **手动循环和条件判断**: 对于更复杂的筛选逻辑，最可靠的方法是**手动遍历整个集合，并在循环内部使用 `if` 条件进行判断**。这正是我们在 `daily.html` 中最终采用的解决方案。

    ```liquid
    {% assign filtered_items = "" | split: "" %}
    {% for item in collection %}
        {% if item.property == 'value' %}
            {% assign filtered_items = filtered_items | push: item %}
        {% endif %}
    {% endfor %}
    ```
    这种方法虽然代码量稍多，但它依赖的是 Liquid 最基础、最稳定的功能，几乎不会出现兼容性问题。

**后续行动：**

*   在未来的开发中，将完全避免使用 `where_exp` 过滤器。
*   继续验证 `daily.html` 中新的、更健壮的 Liquid 逻辑是否能稳定工作。
