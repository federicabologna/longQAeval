import os
import json
import datetime
from pymongo import MongoClient

def serialize_datetime(obj): 
    if isinstance(obj, datetime.datetime): 
        return obj.isoformat() 
    raise TypeError("Type not serializable") 

def export_annotations_to_jsonl(mongo_uri, db_name, output_folder):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    
    for collection_name in db.list_collection_names():
        output_file = f"{collection_name}.jsonl"
        with open(os.path.join(output_folder, output_file), 'w', encoding='utf-8') as f:
            for doc in db[collection_name].find({"rated": "Yes"}):
                doc['_id'] = str(doc['_id'])
                f.write(json.dumps(doc, ensure_ascii=False, default=serialize_datetime) + '\n')
        print(f"Exported {db[collection_name].count_documents({'rated': 'Yes'})} documents to {output_file}")
    
    client.close()


uri = f"mongodb+srv://{open(os.path.join('..', '..', 'PhD', 'apikeys', 'mongodb_clinicalqa_uri.txt')).read().strip()}/?retryWrites=true&w=majority&appName=clinicalqa"
client = MongoClient(uri)
db_name = "pilot1_fine"  # Database containing multiple collections
output_folder = os.path.join('output', 'fine', 'pilot1_fine')
os.makedirs(output_folder, exist_ok=True)
export_annotations_to_jsonl(uri, db_name, output_folder)