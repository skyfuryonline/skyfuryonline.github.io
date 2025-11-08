
import requests
import json
import os
import datetime

# LeetCode API URL
LEETCODE_API_URL = "https://leetcode.cn/graphql/"

# GraphQL query for daily question
DAILY_QUESTION_QUERY = """
query questionOfToday {
  todayRecord {
    question {
      questionFrontendId
      questionTitle
      questionTitleSlug
      difficulty
      topicTags {
        name
        slug
        translatedName
      }
    }
  }
}
"""

def get_daily_question():
    """
    Fetches the daily question from LeetCode.
    """
    try:
        response = requests.post(LEETCODE_API_URL, json={'query': DAILY_QUESTION_QUERY})
        response.raise_for_status()
        data = response.json()
        return data['data']['todayRecord'][0]['question']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching LeetCode data: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing LeetCode data: {e}")
        return None

def format_question_data(question):
    """
    Formats the question data into a dictionary.
    """
    if not question:
        return None

    title = f"LeetCode 每日一题: {question['questionTitle']}"
    difficulty = question['difficulty']
    link = f"https://leetcode.cn/problems/{question['questionTitleSlug']}/"
    tags = [tag['translatedName'] for tag in question.get('topicTags', []) if tag.get('translatedName')]
    
    summary = f"难度: {difficulty}\\n知识点: {', '.join(tags)}"

    return {
        'title': title,
        'summary': summary,
        'link': link,
        'source': 'LeetCode',
        'cache_path': '',  # Not used for LeetCode entries
        'image_files': []  # Not used for LeetCode entries
    }

def save_to_data_file(data):
    """
    Saves the data to a Jekyll data file.
    """
    if not data:
        return

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join('_data', f'daily_{today}.yml')

    # Ensure _data directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Read existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = yaml.safe_load(f) or []
    else:
        existing_data = []

    # Check if LeetCode entry already exists
    for entry in existing_data:
        if entry.get('source') == 'LeetCode':
            print("LeetCode daily question already exists for today.")
            return

    # Add new data and write back to file
    existing_data.append(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(existing_data, f, allow_unicode=True, default_flow_style=False)
    print(f"Successfully saved LeetCode daily question to {file_path}")

def run():
    """
    Main function to run the crawler.
    """
    # This needs PyYAML, so we'll handle the import here
    # to make it clear what is needed.
    try:
        global yaml
        import yaml
    except ImportError:
        print("PyYAML is not installed. Please install it using: pip install PyYAML")
        return

    question = get_daily_question()
    if question:
        formatted_data = format_question_data(question)
        save_to_data_file(formatted_data)

if __name__ == '__main__':
    run()
