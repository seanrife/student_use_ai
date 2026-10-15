import csv
import pandas as pd
from ftfy import fix_text
from pangram import Pangram

pangram_client = Pangram()

def call_pangram(text):
    result = pangram_client.predict(
        text,
        model="pangram-4",
        public_dashboard_link=False,
    )
    return result

# Read in Excel file
df = pd.read_excel('Fall2025Spring2026_AIStudy_Deidentified_07052026.xlsx')

# Open an output CSV and write headers
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['PartID', 'Prompt1', 'Prompt1_score', 'Prompt2',
                     'Prompt2_score', 'Prompt3', 'Prompt3_score',
                     'Prompt4', 'Prompt4_score', 'Prompt5', 'Prompt5_score'])

for index, row in df.iterrows():
    print(call_pangram(fix_text(row['Prompt1'])))