import os
import glob

from sentence_transformers import SentenceTransformer

#Loading a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#METRIC 1: TEXT EMBEDDINGS
def text_embeddings(course_quality):
    
    lines = course_quality.splitlines()
    
    #List to store the transition scores between chunks
    scores = []
    
    for i in range(0, len(lines)-7, 4):
        sentence1 = lines[i:i+4]
        sentence2 = lines[i+4:i+8]
        
        # Merge the lists of 4 sentences into 2 single paragraphs
        chunk1 = " ".join(sentence1)
        chunk2 = " ".join(sentence2)
        
        #Calculating embeddings
        embedding1 = model.encode([chunk1])
        embedding2 = model.encode([chunk2])
        
        #Calculating the embedding similarities
        similarities = model.similarity(embedding1, embedding2)
        
        #Extract the value from similarities
        score = similarities.item()
        
        #Value appended in a list
        scores.append(score)
        
    
    if (len(scores) > 0):
        return 1 + ((sum(scores)/len(scores)) * 4)
    else:
        return 1
    

#METRIC 2: INTERACTION OPPORTUNITY DENSITY (IOD)
def IOD(course_quality):
    
    lowercase = course_quality.lower()    
    total_words = len(lowercase.split())
    
    #Specific terms to focus on 
    interaction_keywords = [
        "quiz", 
        "exercise", 
        "coding task", 
        "assignment",
        "homework",
        "challenge", 
        "reflection prompt", 
        "knowledge check",
        "try it yourself",
        "hands-on"
    ]
    
    #Counting the number of occurances of interaction_keywords
    interactions = sum(lowercase.count(kw) for kw in interaction_keywords)
    
    
    IOD = interactions / total_words

    #Returning score based on 1 interaction per n words
    if IOD >= 0.02:    # 1 per 50 words
         return min(5.0, 4.8 + (IOD * 2))
    elif IOD >= 0.01:  # 1 per 100 words
        return 4.0 + (IOD * 10)  
    elif IOD >= 0.005: # 1 per 200 words
        return 3.0 + (IOD * 10)
    elif IOD >= 0.002: # 1 per 500 words
        return 2.0 + (IOD * 10)
    else:
        return 1.0 + (IOD * 10)
        
        


#TESTING

#Set this to where the folder you unzipped is located at
root_generated_path = "/Users/ms3ood/Desktop/learnpack-generator-6fd80dd"

#This targets ALL files inside ANY subfolder under 'lessons' across ALL 4 run directories
search_pattern = os.path.join(root_generated_path, "*", "lessons", "**", "*")
all_files = glob.glob(search_pattern, recursive=True)

print(f"Found {len(all_files)} total items across all runs.\n")

for file_path in all_files:
    #Skip directories, only process actual files
    if os.path.isdir(file_path):
        continue
        
    file_name = os.path.basename(file_path)
    path_parts = file_path.split(os.sep)
    
    try:
        #Extract folder names for clean console logging
        lesson_name = path_parts[-2]  #e.g., 00_introduction_to_typescript
        course_run = path_parts[-4]   #e.g., typescript-the-basics-20260526024515
    except IndexError:
        lesson_name, course_run = "Unknown", "Unknown"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            course_text = f.read()
            
        print(f"--- Run: {course_run} | Lesson: {lesson_name} | File: {file_name} ---")
        
        # --- RUN YOUR METRICS HERE ---
        TE_score = text_embeddings(course_text)
        IOD_score = IOD(course_text)
        
        print(f"Text Embeddings Score: {TE_score}")
        print(f"IOD Score: {IOD_score}")
       
        
    except Exception as e:
        # Silently skip non-text files or system garbage
        continue
