import chromadb
from chromadb.utils import embedding_functions
chroma_client = chromadb.Client()

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="KBLab/sentence-bert-swedish-cased")

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=["Han förtärde en närande och nyttig måltid.", 
"Varje exempel blir konverterad",
"Det var ett sunkigt hak med ganska gott käk.",
"Han inmundigade middagen tillsammans med ett glas rödvin.",
"Potatischips är jättegoda.",
"Tryck på knappen för att få tala med kundsupporten."],
    ids=["id1", "id2", "id3", "id4", "id5", "id6"]
)

results = collection.query(
    query_texts=["Mannen åt mat."] # Chroma will embed this for you
)

print(results)
print(results["documents"])
