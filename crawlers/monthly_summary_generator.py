import os
import sys
from datetime import datetime, timedelta

# 复用已有的月报生成逻辑
from weekly_summary_generator import generate_and_save_monthly_report

def main():
    # 确定当前北京时间
    beijing_tz = timedelta(hours=8)
    today = datetime.now() + beijing_tz
    
    # 检查明天是不是1号
    tomorrow = today + timedelta(days=1)
    
    if tomorrow.day != 1:
        print(f"今天 ({today.strftime('%Y-%m-%d')}) 不是本月的最后一天。跳过生成月报。")
        sys.exit(0)
        
    # 如果明天是1号，说明今天是本月最后一天
    current_month_str = today.strftime('%Y-%m')
    print(f"今天 ({today.strftime('%Y-%m-%d')}) 是本月最后一天。开始生成 {current_month_str} 月度报告...")
    
    # 调用现有的生成月报逻辑，传入当月字符串
    generate_and_save_monthly_report(current_month_str)
    print("月度报告生成流程结束。")

if __name__ == "__main__":
    main()