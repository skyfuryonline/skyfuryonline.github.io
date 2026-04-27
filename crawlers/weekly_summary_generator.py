import os
import frontmatter
import yaml
import re
import json
from datetime import datetime, timedelta

# --- 配置 ---
LOG_DIR = "_gwy_logs"
REPORT_DIR = "_gwy_reports"
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

# 从配置中读取 LLM 模型
COACH_MODEL = "qwen3.6-plus-2026-04-02"
COACH_SYSTEM_PROMPT = "你是一位专业的学习分析与激励教练。"
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
        coach_profile = config_data.get("llm_profiles", {}).get("learning_coach", {})
        if coach_profile.get("model"):
            COACH_MODEL = coach_profile.get("model")
        if coach_profile.get("prompt"):
            COACH_SYSTEM_PROMPT = coach_profile.get("prompt")
except Exception as e:
    print(f"Warning: Failed to load LLM config from {CONFIG_FILE}, using defaults. Error: {e}")

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
    """读取某个月份（如'2026-02'）的所有周报，按日期排序返回（跨月周报也会被正确归属）"""
    reports = []
    if not os.path.exists(REPORT_DIR):
        return reports
    
    try:
        target_year, target_month = map(int, month_str.split('-'))
    except ValueError:
        return reports
        
    for filename in sorted(os.listdir(REPORT_DIR)):
        if filename.startswith("week-of-") and filename.endswith(".md"):
            date_str = filename.replace('week-of-', '').replace('.md', '')
            try:
                start_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                end_date = start_date + timedelta(days=6)
                
                # 如果这周的周一或周日在这个月内，就认为属于这个月的周报
                start_match = (start_date.year == target_year and start_date.month == target_month)
                end_match = (end_date.year == target_year and end_date.month == target_month)
                
                if start_match or end_match:
                    filepath = os.path.join(REPORT_DIR, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        post = frontmatter.load(f)
                        reports.append({
                            'date_str': date_str,
                            'content': post.content
                        })
            except:
                continue
    return reports

def extract_core_sections(content):
    """从周报中提取核心的 '行动纠偏' 或 '状态与不足' 及之后的章节"""
    # 兼容新旧版本的周报格式
    match = re.search(r'(#+)?\s*(?:\d+\.\s*)?\**(?:行动纠偏与具体任务|状态与不足|状态与反思|状态与不足)\**.*', content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    return content[:800] + "..." # Fallback，如果没有找到匹配项则截取前半部分

def extract_soul_questions(content):
    """专门从上一篇周报中提取 '灵魂拷问' 环节"""
    match = re.search(r'(#+)?\s*(?:\d+\.\s*)?\**教练的(?:灵魂拷问|深度启发)\**.*', content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    return ""

def extract_monthly_strategy(content):
    """专门从上个月的月报中提取 '强制战略部署' 和 '终极灵魂拷问' 环节"""
    match = re.search(r'(#+)?\s*(?:\d+\.\s*)?\**(?:下月强制战略部署|下月弹性战略部署|强制战略部署|弹性战略部署|宏观战略建议)\**.*', content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    return ""

def generate_and_save_monthly_report(month_str):
    """生成并保存月度报告"""
    reports = get_reports_by_month(month_str)
    if not reports:
        print(f"  - 没有找到 {month_str} 的周报，跳过月报生成。")
        return ""

    content_concat = ""
    for r in reports:
        content_concat += f"\n\n### {r['date_str']} 周报\n{r['content']}\n"
    
    # 尝试获取上个月的月报，提取战略部署和灵魂拷问
    prev_month_context = ""
    try:
        current_month_date = datetime.strptime(month_str, '%Y-%m')
        # 获取上个月的日期（当前月1号减去1天）
        prev_month_date = current_month_date.replace(day=1) - timedelta(days=1)
        prev_month_str = prev_month_date.strftime('%Y-%m')
        
        prev_month_report_path = os.path.join(REPORT_DIR, f"month-of-{prev_month_str}.md")
        if os.path.exists(prev_month_report_path):
            with open(prev_month_report_path, 'r', encoding='utf-8') as f:
                prev_content = frontmatter.load(f).content
                strategy_content = extract_monthly_strategy(prev_content)
                if strategy_content:
                    prev_month_context = f"\n【🎯 上个月 ({prev_month_str}) 的战略指南与深度启发】\n这是你在上个月末给我的建议和方向。请在本次月报的开头，**温和地复盘我这整个月的实践情况，如果我调整了方向请分析并支持**：\n{strategy_content}\n"
    except Exception as e:
        print(f"尝试读取上月月报时出错: {e}")

    if prev_month_context:
        prompt = f"""你是一位专业、务实且充满同理心的公考备考主教练。请重点根据我 {month_str} 月份各周的数据和报告内容，为我生成一份【月度复盘与下月指南】。

请客观分析我的状态，基于事实给予肯定和鼓励。如果发现我的备考重心或实际情况发生了变化，请理解这些变化，灵活调整你的建议，而不是死板地坚持之前的目标。你的报告应当遵循 Markdown 格式，包含以下核心部分：

1. **月度回顾与目标对齐**: 
   - 温和地复盘我本月是否落实了上个月的【战略部署】。如果我因实际困难调整了方向，请分析新方向的合理性并给予鼓励；如果我付出了努力，请给予真诚的认可。
2. **月度时间与精力盘点**: 评估整个月的精力投放。模块的时间侧重点演变是否符合我当前的真实状态？指出本月做得好的地方以及潜在的时间隐患。
3. **核心短板与突破口**: 结合这几周的数据和日记，指出我当前需要关注的能力短板或心态问题，并提供建设性的克服建议。
4. **下月弹性战略部署**: 基于我当前的真实状态给出下个月的宏观战略方向。提供 2-3 条**具有弹性的行动指南**，切忌生搬硬套或强制执行。
5. **月度深度启发**: 提出 1 个启发性的问题，帮助我缓解焦虑、理清思路，让我在下个月的备考中保持思考。

{prev_month_context}

【我的 {month_str} 月份所有周报记录】：
{content_concat}
"""
    else:
        prompt = f"""你是一位专业、务实且充满同理心的公考备考主教练。请重点根据我 {month_str} 月份各周的数据和报告内容，为我生成一份【月度复盘与下月指南】。

请客观分析我的状态，基于事实给予肯定和鼓励。如果发现我的备考重心或实际情况发生了变化，请理解这些变化，灵活调整你的建议，而不是死板地坚持之前的目标。你的报告应当遵循 Markdown 格式，包含以下核心部分：

1. **月度时间与精力盘点**: 评估整个月的精力投放。模块的时间侧重点演变是否符合我当前的真实状态？指出本月做得好的地方以及潜在的时间隐患。
2. **核心短板与突破口**: 结合这几周的数据和日记，指出我当前需要关注的能力短板或心态问题，并提供建设性的克服建议。
3. **下月弹性战略部署**: 基于我当前的真实状态给出下个月的宏观战略方向。提供 2-3 条**具有弹性的行动指南**，切忌生搬硬套或强制执行。
4. **月度深度启发**: 提出 1 个启发性的问题，帮助我缓解焦虑、理清思路，让我在下个月的备考中保持思考。

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
def create_prompt(this_week_data, last_week_data=None, last_month_report="", current_month_reports="", last_week_soul_questions=""):
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
    if last_week_soul_questions:
        context_section += f"\n【🎯 上周你给我的深度启发 (重点回顾区)】\n这是上周周报末尾你给我留下的启发问题。请你在本次周报中，关注我的日记中对这些问题的思考情况：\n{last_week_soul_questions}\n"
    if last_month_report:
        context_section += f"\n【我的上个月学习画像 (长期背景)】\n这是我上个月的总体状态总结，请将其作为大背景参考：\n{last_month_report}\n"
    if current_month_reports:
        context_section += f"\n【我本月前几周的周报状态 (短期连贯性)】\n这是本月内已经生成的周报核心摘要，请注意我的状态连贯性与问题是否改善：\n{current_month_reports}\n"
    
    if last_week_soul_questions:
        prompt = f"""
你是一位专业、务实且充满同理心的公考备考主教练。请客观分析我的状态，基于事实给予肯定和鼓励。如果发现我的备考重心或实际情况发生了变化，请理解这些变化，灵活调整你的建议，而不是死板地坚持之前的目标。

请根据我的历史状态（月报）、本月前置周报、以及本周的数据和日记，生成一份高质量的 Markdown 周报，包含以下核心部分：

1. **状态跟进 (重点环节)**:
   - 关注我本周的日记中对【上周启发问题】的思考。倾听并分析我遇到困难的原因。如果我因实际情况改变了计划，请给予理解并帮忙调整；如果我付出了努力，请给予真诚的鼓励。
2. **时间分布与效率诊断**:
   - 客观对比本周与上周的总时长、打卡天数，看到我的进步并给予肯定。
   - 剖析本周各个科目的时间分布情况。客观指出数据反映的备考倾向，提供温和的优化建议。
3. **行动建议与具体任务**:
   - 从日记中提取出本周遇到的具体困难。
   - 针对这些困难，给出**具体且灵活**的建议（如“如果觉得累，可以尝试调整目标，每天专注完成并订正20道题”），重在具有实操性和可适应性。
4. **教练的深度启发 (关键环节)**:
   - 基于本周的真实状态，向我提出 1 到 2 个启发性的新问题。
   - 明确指出这些问题是为了帮助我理清思路、缓解焦虑或明确下周的重心，供我在下周日记中自由探讨。

请保持你的语言专业、温和、具有启发性。

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
    else:
        prompt = f"""
你是一位专业、务实且充满同理心的公考备考主教练。请客观分析我的状态，基于事实给予肯定和鼓励。如果发现我的备考重心或实际情况发生了变化，请理解这些变化，灵活调整你的建议，而不是死板地坚持之前的目标。

请根据我的历史状态（月报）、本月前置周报、以及本周的数据和日记，生成一份高质量的 Markdown 周报，包含以下核心部分：

1. **时间分布与效率诊断**:
   - 客观对比本周与上周的总时长、打卡天数，看到我的进步并给予肯定。
   - 剖析本周各个科目的时间分布情况。客观指出数据反映的备考倾向，提供温和的优化建议。
2. **行动建议与具体任务**:
   - 从日记中提取出本周遇到的具体困难。
   - 针对这些困难，给出**具体且灵活**的建议（如“如果觉得累，可以尝试调整目标，每天专注完成并订正20道题”），重在具有实操性和可适应性。
3. **教练的深度启发 (关键环节)**:
   - 基于本周的真实状态，向我提出 1 到 2 个启发性的新问题。
   - 明确指出这些问题是为了帮助我理清思路、缓解焦虑或明确下周的重心，供我在下周日记中自由探讨。

请保持你的语言专业、温和、具有启发性。

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
        model=COACH_MODEL,
        messages=[
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
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
            # 修复：只对 2026-03 及以后的月份触发月报兜底，避免复活早期没有正确跑通的废弃月报
            if prev_month_str >= "2026-03":
                print(f"检测到上个月 ({prev_month_str}) 未按时生成月度报告，作为兜底机制现在尝试自动补发...")
                last_month_report_content = generate_and_save_monthly_report(prev_month_str)
            else:
                print(f"检测到上个月 ({prev_month_str}) 报告不存在，但属于早期数据，跳过自动补发以避免复活旧错误。")
            
        # 4. 处理“中期连贯记忆” (本月内，处于当前目标周之前的周报)
        current_month_reports_concat = ""
        current_month_reports = get_reports_by_month(report_month_str)
        target_date_str = start_of_last_week.strftime('%Y-%m-%d')
        
        last_week_soul_questions = "" # 提取上周灵魂拷问
        
        # 为了找上周灵魂拷问，我们需要所有已生成的周报（包括可能跨月的“上周”）
        # 简单处理：我们看看上周对应日期的报告文件是否存在
        start_of_week_before_last = start_of_last_week - timedelta(days=7)
        last_week_report_filename = f"week-of-{start_of_week_before_last.strftime('%Y-%m-%d')}.md"
        last_week_report_filepath = os.path.join(REPORT_DIR, last_week_report_filename)
        if os.path.exists(last_week_report_filepath):
            with open(last_week_report_filepath, 'r', encoding='utf-8') as f:
                last_week_content = frontmatter.load(f).content
                last_week_soul_questions = extract_soul_questions(last_week_content)

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
            current_month_reports=current_month_reports_concat,
            last_week_soul_questions=last_week_soul_questions
        )
        
        report_content = get_llm_summary(llm_prompt)
        save_report(report_content, start_of_last_week)
    else:
        print("上周没有学习记录，跳过周报生成。")