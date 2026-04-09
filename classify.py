from binoculars import Binoculars
import csv
import pandas as pd

bino = Binoculars()


# Remove problematic characters and clean text
def clean_string(text: str) -> str:
    clean_text = text.encode('utf-8', errors='ignore').decode('utf-8')
    clean_text = clean_text.replace('Â', '')
    return clean_text

# Grab the AI probability from the classifier and return rounded float
def ai_probability(text: str) -> float:
    """
    Returns the model's estimated probability that the text
    belongs to the AI-generated class.
    """
    text = clean_string(text)
    return float(bino.compute_score(text))

# Score text (but only if it is actually text)
def score(text: str) -> str:
    if not isinstance(text, str):
        return ""
    else:
        return round(ai_probability(text), 5)

# Read in Excel file
df = pd.read_excel('Fall2025_AIStudy_Deidentified.xlsx')

# Open an output CSV and write headers
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['PartID', 'Prompt1', 'Prompt1_score', 'Prompt2',
                     'Prompt2_score', 'Prompt3', 'Prompt3_score',
                     'Prompt4', 'Prompt4_score', 'Prompt5', 'Prompt5_score'])

    # Iterate over df and write scores and text to output CSV
    for index, row in df.iterrows():
        Prompt1_score = score(row['Prompt1'])
        Prompt2_score = score(row['Prompt2'])
        Prompt3_score = score(row['Prompt3'])
        Prompt4_score = score(row['Prompt4'])
        Prompt5_score = score(row['Prompt5'])
        writer.writerow([row['PartID'], row['Prompt1'], Prompt1_score,
                         row['Prompt2'], Prompt2_score, row['Prompt3'],
                         Prompt3_score, row['Prompt4'], Prompt4_score,
                         row['Prompt5'], Prompt5_score])

