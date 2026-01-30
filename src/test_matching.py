from embeddings import extract_embedding, build_embedding_database
from matching import match_face

DB_PATH = "data/processed"
QUERY_IMAGE = "data/processed/person1/001.jpg"

db = build_embedding_database(DB_PATH)
query_emb = extract_embedding(QUERY_IMAGE)

name, distance, decision = match_face(query_emb, db)

print("Predicted identity:", name)
print("Distance:", distance)
print("Accepted:", decision)
