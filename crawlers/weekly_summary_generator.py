import os
import frontmatter
import yaml
import re
from datetime import datetime, timedelta

# --- 配置 ---
LOG_DIR = "_gwy_logs"
REPORT_DIR = "_gwy_reports"
# 注意：在 GitHub Actions 中，我们需要配置 OPENAI_API_KEY 等环境变量
# 从环境变量读取

api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_API_BASE")
from openai import OpenAI
client = OpenAI(api_key=api_key, base_url=base_url)

# --- 1. 数据收集 ---
def get_week_data(start_of_week, end_of_week):
    """收集指定周的学习数据和日记内容（通用函数）"""
    weekly_total_hours = 0
    subject_distribution = {}
    active_days = set()
    diary_content = ""

    if not os.path.exists(LOG_DIR):
        return {
            "total_hours": 0, "active_days": 0, "subjects": {}, "diaries": ""
        }

    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                post = frontmatter.load(f)
            except yaml.YAMLError:
                continue
                
            log_date_obj = post.get('date')
            if isinstance(log_date_obj, datetime):
                log_date = log_date_obj.date()
            else:
                log_date = log_date_obj

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

def get_last_week_data():
    """收集上周的学习数据和日记内容"""
    beijing_tz = timedelta(hours=8)
    today = datetime.now() + beijing_tz
    start_of_this_week = today - timedelta(days=today.weekday())
    start_of_last_week = start_of_this_week - timedelta(days=7)
    end_of_last_week = start_of_last_week + timedelta(days=6)
    return get_week_data(start_of_last_week, end_of_last_week)

def get_week_before_last_data():
    """收集上上周的学习数据和日记内容"""
    beijing_tz = timedelta(hours=8)
    today = datetime.now() + beijing_tz
    start_of_this_week = today - timedelta(days=today.weekday())
    start_of_last_week = start_of_this_week - timedelta(days=7)
    start_of_week_before_last = start_of_last_week - timedelta(days=7)
    end_of_week_before_last = start_of_week_before_last + timedelta(days=6)
    return get_week_data(start_of_week_before_last, end_of_week_before_last)

# --- 新增：月报与历史数据读取 ---

def get_reports_by_month(month_str):
    """读取某个月份（如'2026-02'）的所有周报，按日期排序返回"""
    reports = []
    if not os.path.exists(REPORT_DIR):
        return reports
    
    for filename in sorted(os.listdir(REPORT_DIR)):
        # 匹配 `week-of-2026-02-xx.md`
        if filename.startswith(f"week-of-{month_str}"):
            filepath = os.path.join(REPORT_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    post = frontmatter.load(f)
                    date_str = filename.replace('week-of-', '').replace('.md', '')
                    reports.append({
                        'date_str': date_str,
                        'content': post.content
                    })
                except:
                    continue
    return reports

def extract_core_sections(content):
    """从周报中提取核心的 '状态与反思' 和 '改进建议' 章节"""
    # 匹配 "状态与反思" 以及其后的所有内容
    match = re.search(r'(#+)?\s*(?:\d+\.\s*)?\**状态与反思\**.*', content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    return content[:800] + "..." # Fallback，如果没有找到匹配项则截取前半部分

def generate_and_save_monthly_report(month_str):
    """生成并保存月度报告"""
    reports = get_reports_by_month(month_str)
    if not reports:
        print(f"  - 没有找到 {month_str} 的周报，跳过月报生成。")
        return ""

    content_concat = ""
    for r in reports:
        content_concat += f"\n\n### {r['date_str']} 周报\n{r['content']}\n"
    
    prompt = f"""你是一位专业的学习分析与战略教练。请根据我 {month_str} 月份的所有周报内容，为我生成一份【月度学习画像与长期战略建议】。

请直接输出核心分析，避免繁琐的数据罗列。你的报告应当遵循 Markdown 格式，包含以下核心部分：
1. **长期情绪与状态变化轨迹**: 总结这个月情绪周期的起伏。
2. **核心瓶颈与突破**: 指出我在这个月克服了什么困难，还有哪些瓶颈未能突破。
3. **下个月宏观战略建议**: 为下个月的学习定下基调和核心策略。

【我的 {month_str} 月份所有周报记录】：
{content_concat}
"""
    print(f"\n--- 正在为 {month_str} 生成月度报告 ---")
    summary = get_llm_summary(prompt)

    if summary:
        filename = f"month-of-{month_str}.md"
        filepath = os.path.join(REPORT_DIR, filename)
        
        post = frontmatter.Post(summary)
        post['layout'] = 'gwy_log_entry'
        post['title'] = f"{month_str} 学习月报"
        post['date'] = datetime.now()
        post['is_report'] = True

        if not os.path.exists(REPORT_DIR):
            os.makedirs(REPORT_DIR)
        
        with open(filepath, 'wb') as f:
            frontmatter.dump(post, f, encoding='utf-8')
        print(f"✅ 月报已保存到: {filepath}")
        
    return summary

# --- 2. 生成 Prompt ---
def create_prompt(this_week_data, last_week_data=None, last_month_report="", current_month_reports=""):
    """根据数据创建发送给 LLM 的 Prompt"""
    subject_lines = "\n".join([f"- {name}: {hours:.1f} 小时" for name, hours in this_week_data["subjects"].items()])
    
    last_week_section = ""
    if last_week_data and last_week_data["total_hours"] > 0:
        last_subject_lines = "\n".join([f"- {name}: {hours:.1f} 小时" for name, hours in last_week_data["subjects"].items()])
        last_week_section = f"""
【我的上周核心数据 (作为数值对比)】
- 总学习时长: {last_week_data['total_hours']:.1f} 小时
- 有效打卡天数: {last_week_data['active_days']} 天
- 科目分布:
{last_subject_lines}
"""

    context_section = ""
    if last_month_report:
        context_section += f"\n【我的上个月学习画像 (长期背景)】\n这是我上个月的总体状态总结，请将其作为大背景参考：\n{last_month_report}\n"
    if current_month_reports:
        context_section += f"\n【我本月前几周的周报状态 (短期连贯性)】\n这是本月内已经生成的周报核心摘要，请注意我的状态连贯性与问题是否改善：\n{current_month_reports}\n"
    
    prompt = f"""
你是一位专业的学习分析与激励教练。请根据我的历史状态（月报）与本月进展（前置周报），分析我上周的数据，为我生成一份连贯的周报。
周报应遵循严格的 Markdown 格式，包含以下部分：

1.  **本周概览**: 基于数据，一句话总结总时长和打卡天数，并与上周对比。给出简短的总体评价。
2.  **学习重点**: 分析科目分布，指出本周投入最多的科目，并分析可能的原因。
3.  **状态与反思**: 结合我的日记内容和【历史状态】，敏锐地洞察我本周的学习状态、情绪波动。如果我曾经在历史中遇到过某个问题且本周有改善，请给予肯定；如果某个问题反复出现，请提出更深度的干预建议。
4.  **改进建议**: 根据以上分析，为我提出 2-3 条具体的、可执行的下周学习建议。

请让你的语言风格既专业又充满鼓励性，像一位真正的教练。

{context_section}

---
【我的本周最新数据与日记 (核心分析对象)】
- 总学习时长: {this_week_data['total_hours']:.1f} 小时
- 有效打卡天数: {this_week_data['active_days']} 天
- 科目分布:
{subject_lines}

日记内容:
{this_week_data['diaries']}
---
{last_week_section}
"""
    return prompt.strip()

# --- 3. 调用 LLM ---
def get_llm_summary(prompt):
    """调用 LLM API 获取汇总。"""
    print("\n--- 发送给 LLM 的 Prompt (截断显示) ---")
    print(prompt[:500] + "\n...[内容截断]...\n")
    
    response = client.chat.completions.create(
        model="qwen3-max-2026-01-23",
        messages=[
            {"role": "system", "content": "你是一位专业的学习分析与激励教练。"},
            {"role": "user", "content": prompt}
        ]
    )
    summary = response.choices[0].message.content
    
    return summary.strip()

# --- 4. 写入文件 ---
def save_report(summary, report_start_date):
    """将周报保存为新的 Markdown 文件"""
    filename = f"week-of-{report_start_date.strftime('%Y-%m-%d')}.md"
    filepath = os.path.join(REPORT_DIR, filename)

    post = frontmatter.Post(summary)
    post['layout'] = 'gwy_log_entry'
    post['title'] = f"{report_start_date.strftime('%Y年%m月%d日')} 学习周报"
    post['date'] = datetime.now()
    post['is_report'] = True

    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    with open(filepath, 'wb') as f:
        frontmatter.dump(post, f, encoding='utf-8')
    
    print(f"\n✅ 周报已保存到: {filepath}")

# --- 主函数 ---
if __name__ == "__main__":
    # 确保目录存在
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # 1. 确定时间基准
    beijing_tz = timedelta(hours=8)
    today = datetime.now() + beijing_tz
    start_of_this_week = today - timedelta(days=today.weekday())
    start_of_last_week = start_of_this_week - timedelta(days=7) # 目标周报的起始日（上周一）
    
    report_week_data = get_last_week_data()
    
    if report_week_data["total_hours"] > 0:
        # 获取用于数值对比的上上周数据
        comparison_week_data = get_week_before_last_data()
        
        # 2. 计算所处的月份和上个月
        report_month_str = start_of_last_week.strftime('%Y-%m')
        prev_month_date = start_of_last_week.replace(day=1) - timedelta(days=1)
        prev_month_str = prev_month_date.strftime('%Y-%m')
        
        # 3. 处理“长期记忆” (上个月的月报)
        last_month_report_content = ""
        monthly_report_path = os.path.join(REPORT_DIR, f"month-of-{prev_month_str}.md")
        
        if os.path.exists(monthly_report_path):
            with open(monthly_report_path, 'r', encoding='utf-8') as f:
                last_month_report_content = frontmatter.load(f).content
        else:
            # 如果上个月的月报不存在，尝试自动生成
            print(f"检测到上个月 ({prev_month_str}) 尚未生成月度报告，尝试自动生成...")
            last_month_report_content = generate_and_save_monthly_report(prev_month_str)
            
        # 4. 处理“中期连贯记忆” (本月内，处于当前目标周之前的周报)
        current_month_reports_concat = ""
        current_month_reports = get_reports_by_month(report_month_str)
        target_date_str = start_of_last_week.strftime('%Y-%m-%d')
        
        for r in current_month_reports:
            # 只提取在本周报之前生成的周报
            if r['date_str'] < target_date_str:
                core_content = extract_core_sections(r['content'])
                current_month_reports_concat += f"\n- {r['date_str']} 周报状态摘要:\n{core_content}\n"
        
        # 5. 组装终极 Prompt 并生成周报
        llm_prompt = create_prompt(
            this_week_data=report_week_data, 
            last_week_data=comparison_week_data, 
            last_month_report=last_month_report_content, 
            current_month_reports=current_month_reports_concat
        )
        
        report_content = get_llm_summary(llm_prompt)
        save_report(report_content, start_of_last_week)
    else:
        print("上周没有学习记录，跳过周报生成。")