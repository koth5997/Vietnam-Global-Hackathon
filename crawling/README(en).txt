Key Features
PDF Extraction: Extracts text content from PDF resume files.
Contextual Analysis (spaCy): Uses Named Entity Recognition (NER) to identify relevant sections such as education, experience, and awards.
Keyword Extraction (NLTK): Extracts phrases related to filmmaking roles (e.g., director, film, production).
Duplicate Filtering: Prevents keywords from being counted multiple times in a single sentence.
Weighted Scoring: Assigns different weights to keywords based on their importance.
Pass/Fail Classification: Classifies resumes as Pass or Fail based on a user-defined score threshold.
Technology Used
Python: The main programming language used for the script.
NLTK (Natural Language Toolkit): A library for text processing tasks such as tokenization, stop word removal, and part-of-speech tagging.
spaCy: An advanced NLP library used for contextual analysis, particularly Named Entity Recognition (NER).
PyPDF2: A library for extracting text content from PDF files.
Workflow
PDF Extraction: Extracts text content from the provided PDF resume using PyPDF2.

Contextual Analysis: Identifies relevant sections like Education, Experience, and Awards using spaCy's Named Entity Recognition (NER). This focuses the analysis on areas most likely to contain relevant keywords.

Keyword Extraction: NLTK performs tokenization, stop word removal, and part-of-speech tagging to extract phrases containing keywords related to filmmaking roles, such as director, film, and production.

Duplicate Filtering: Tracks identified keywords using a set. Only the first occurrence of a keyword in a sentence contributes to the score, preventing overestimation from repeated keywords.

Weighted Scoring: Each keyword is assigned a weight based on its importance in filmmaking roles. For example, "screenwriter" might have a higher weight than "film." The script iterates through sentences and adds the weight of each identified keyword to the total score.

Pass/Fail Classification: Compares the final score to a user-defined threshold. Resumes scoring above the threshold are classified as Pass, while those below are classified as Fail.

How to Use the Script
Update Keywords and Threshold:

Modify the relevant_keywords dictionary and keyword_weights dictionary to add or adjust keywords and their weights.
Change the pass/fail threshold by editing the if pdf_score >= 50: line in the script.
Set PDF Path:

Update the pdf_path variable in the if __name__ == "__main__": block with the path of the PDF resume to analyze.
Run the Script:

Use Python to execute the script.
The script will display:
Extracted text content.
Identified relevant sections.
Extracted keyword phrases.
Final score and pass/fail classification.