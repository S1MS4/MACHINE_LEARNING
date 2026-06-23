import pandas as pd


csv_path = r'../data/student_dropout_dataset.csv'
df = pd.read_csv(csv_path)

cols = ['region', 'courses_enrolled', 'completed_assignments', 'completion_rate', 'login_frequency',
        'last_activity_days_ago', 'forum_posts_count', 'exam_season', 'label_name']
df = df[cols]
df.to_csv(csv_path, index=False)

unique_categories = df['category'].unique()

print(f"\nTotal number of unique categories: {len(unique_categories)}")

