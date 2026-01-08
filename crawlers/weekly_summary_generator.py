import os
import frontmatter
import yaml
from datetime import datetime, timedelta

# --- 配置 ---
LOG_DIR = "_gwy_logs"
REPORT_DIR = "_gwy_reports"
# 注意：在 GitHub Actions 中，我们需要配置 OPENAI_API_KEY 等环境变量
# from openai import OpenAI
# client = OpenAI()

# --- 1. 数据收集 ---
def get_this_week_data():
    """收集本周的学习数据和日记内容"""
    today = datetime.now()
    # 周一为 0，周日为 6
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    print(f"正在收集本周数据: {start_of_week.date()} -> {end_of_week.date()}")

    weekly_total_hours = 0
    subject_distribution = {}
    active_days = set()
    diary_content = ""

    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                post = frontmatter.load(f)
            except yaml.YAMLError:
                print(f"  - 跳过文件（YAML 解析失败）: {filename}")
                continue
                
            log_date = post.get('date').date()

            if start_of_week.date() <= log_date <= end_of_week.date():
                log_hours = 0
                if post.get('subjects'):
                    for subject in post.get('subjects'):
                        hours = subject.get('hours', 0)
                        if hours > 0:
                            log_hours += hours
                            subject_name = subject.get('name', '未知')
                            subject_distribution[subject_name] = (
                                subject_distribution.get(subject_name, 0) + hours
                            )
                
                if log_hours > 0:
                    weekly_total_hours += log_hours
                    active_days.add(log_date)
                
                if post.content.strip():
                    diary_content += f"\n\n### {log_date.strftime('%Y-%m-%d')} ({post.get('title', '无标题')})\n\n{post.content.strip()}"

    return {
        "total_hours": weekly_total_hours,
        "active_days": len(active_days),
        "subjects": subject_distribution,
        "diaries": diary_content.strip()
    }

# --- 2. 生成 Prompt ---
def create_prompt(data):
    """根据数据创建发送给 LLM 的 Prompt"""
    subject_lines = "\n".join([f"- {name}: {hours:.1f} 小时" for name, hours in data["subjects"].items()])
    
    prompt = f"""
你是一位专业的学习分析与激励教练。请根据我过去一周的学习数据和日记，为我生成一份周报。
周报应遵循严格的 Markdown 格式，包含以下部分：

1.  **本周概览**: 基于数据，一句话总结总时长和打卡天数，并与上周对比（如果我提供了上周数据）。给出简短的总体评价。
2.  **学习重点**: 分析科目分布，指出本周投入最多的科目，并分析可能的原因。
3.  **状态与反思**: 结合我的日记内容，敏锐地洞察我本周的学习状态、情绪波动、遇到的问题和感悟。
4.  **改进建议**: 根据以上分析，为我提出 2-3 条具体的、可执行的下周学习建议。

请让你的语言风格既专业又充满鼓励性，像一位真正的教练。

---
【我的本周数据】
- 总学习时长: {data['total_hours']:.1f} 小时
- 有效打卡天数: {data['active_days']} 天
- 科目分布:
{subject_lines}

---
【我的日记内容】
{data['diaries']}
---
"""
    return prompt.strip()

# --- 3. 调用 LLM (示例) ---
def get_llm_summary(prompt):
    """调用 LLM API 获取周报。这是一个示例，需要替换为真实调用。"""
    print("\n--- 发送给 LLM 的 Prompt ---")
    print(prompt)
    
    # response = client.chat.completions.create(
    #     model="qwen2.5-max",
    #     messages=[
    #         {"role": "system", "content": "你是一位专业的学习分析与激励教练。"},
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # summary = response.choices[0].message.content
    
    # --- For testing purposes ---
    summary = """
# 备考周报 (示例)

你好！这是你过去一周的备考分析报告，你做得非常棒！

### 1. 本周概览

你本周有效学习了 **15.5 小时**，共打卡 **3** 天。与上周相比，时长有所增加，这是一个积极的信号！整体来看，你正逐步进入稳定的学习节奏。

### 2. 学习重点

本周的绝对重心是 **判断推理**，投入了超过一半的时间。这表明你可能正在攻克一个难点，或者对这个模块特别感兴趣。同时，**申论** 的学习时间较少，需要关注。

### 3. 状态与反思

从你的日记中可以看出，你对“一笔画”问题感到困惑，这是一个非常具体的学习难点。同时，你也意识到了素材积累的重要性。这些都是宝贵的自我洞察！

### 4. 改进建议

1.  **专项突破**: 下周可以继续巩固“一笔画”问题，找一些专题视频或课程，彻底解决它。
2.  **平衡科目**: 建议下周至少安排 2-3 次申论学习，特别是素材的阅读和积累，避免偏科。
3.  **保持复盘**: 你已经有了复盘的习惯，这非常好，请务必坚持下去！

继续加油，你正在正确的道路上稳步前进！
"""
    # --- End of testing block ---
    
    print("\n--- 从 LLM 收到的周报 ---")
    print(summary)
    return summary.strip()

# --- 4. 写入文件 ---
def save_report(summary):
    """将周报保存为新的 Markdown 文件"""
    today = datetime.now()
    # 使用周一的日期来命名文件
    start_of_week = today - timedelta(days=today.weekday())
    filename = f"week-of-{start_of_week.strftime('%Y-%m-%d')}.md"
    filepath = os.path.join(REPORT_DIR, filename)

    # 创建一个新的 frontmatter post 对象
    post = frontmatter.Post(summary)
    post['layout'] = 'gwy_log_entry' # 假设我们为周报和日记用同一个布局
    post['title'] = f"{start_of_week.strftime('%Y年%m月%d日')} 学习周报"
    post['date'] = today
    post['is_report'] = True # 添加一个特殊标记

    with open(filepath, 'wb') as f:
        frontmatter.dump(post, f, encoding='utf-8')
    
    print(f"\n✅ 周报已保存到: {filepath}")

# --- 主函数 ---
if __name__ == "__main__":
    weekly_data = get_this_week_data()
    if weekly_data["total_hours"] > 0:
        llm_prompt = create_prompt(weekly_data)
        report_content = get_llm_summary(llm_prompt)
        save_report(report_content)
    else:
        print("本周没有学习记录，跳过周报生成。")

