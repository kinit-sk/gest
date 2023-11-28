import json
import os

from tqdm import tqdm
from trankit import Pipeline

class Parser:

    cache_path = os.path.join('/','labs', 'cache', 'parser')

    language_map = {
        'be': 'belarusian',
        'bg': 'bulgarian',
        'hr': 'croatian',
        'cs': 'czech',
        'pl': 'polish',
        'ru': 'russian',
        'sl': 'slovenian',
        'sk': 'slovak',
        'sr': 'serbian',
        'uk': 'ukrainian',
    }

    def __init__(self, language, embedding='xlm-roberta-base'):
        self.file_path = os.path.join(self.cache_path, language, 'results.json')
        self.language = language
        self.embedding = embedding
        self.loaded = False
        
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as json_file:
                self.dict = json.load(json_file)
        else:
            self.dict = {}

    def load_model(self):
        self.pipeline = Pipeline(self.language_map[self.language], embedding=self.embedding)
        return self

    
    def parse(self, texts):
        save = False
        for text in tqdm(texts):
            if text not in self.dict:
                parse = self.pipeline.posdep(text)
                self.dict[text] = parse
                save = True

        if save:
            self.save()

        return {
            text: self.dict[text]
            for text in texts
        }

    
    def save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w') as json_file:
            json.dump(self.dict, json_file)
