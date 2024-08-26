import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

data = pd.read_csv('jobs_data.csv')
data = data.drop_duplicates()
data = data.dropna()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

data['job_title_bert'] = data['Job_title'].apply(lambda x: get_bert_embeddings(x))

pools = []

similarity_threshold = 0.87



def add_job_to_pool(job_title, embedding):
    if not pools:
        pools.append({'job_titles': [job_title], 'embeddings': [embedding]})
    else:
        added_to_pool = False
        for pool in pools:
            similarities = [cosine_similarity([embedding], [emb])[0][0] for emb in pool['embeddings']]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity >= similarity_threshold:
                if job_title not in pool['job_titles']:
                    pool['job_titles'].append(job_title)
                    pool['embeddings'].append(embedding)
                added_to_pool = True
                break
        
        if not added_to_pool:
            pools.append({'job_titles': [job_title], 'embeddings': [embedding]})

for index, row in data.iterrows():
    job_title = row['Job_title']
    embedding = row['job_title_bert']
    add_job_to_pool(job_title, embedding)

largest_pool = max(pools, key=lambda p: len(p['job_titles']))

def generate_pool_name(job_titles):
    all_keywords = ' '.join(job_titles)
    all_keywords = re.sub(r'[^\w\s]', '', all_keywords)
    words = all_keywords.split()
    common_keywords = [word for word, count in Counter(words).most_common(2)]
    return ' & '.join(common_keywords) or 'General'

largest_pool['name'] = generate_pool_name(largest_pool['job_titles'])
print(f"Demanding jobs: {largest_pool['name']}")
# print(f"Job Titles: {largest_pool['job_titles']}")
# print(f"Number of jobs in the largest pool: {len(largest_pool['job_titles'])}")
