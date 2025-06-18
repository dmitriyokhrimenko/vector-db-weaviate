import sys
import weaviate
import weaviate.classes as wvc
import torch
from transformers import CLIPProcessor, CLIPModel

client = weaviate.connect_to_local(
    host="127.0.0.1",
    port=8080,
    grpc_port=50051,
)

try:
    client.is_ready()
except Exception as e:
    print('Error: %s', repr(e))
    sys.exit(1)

print(f"Collections: {client.collections.list_all()}")

docs = client.collections.get("GoogleDocs")
response = docs.query.hybrid(
    query="during commissioning at the Closer Pets factory",
    alpha=0.1,
    limit=2,
    return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
)
for result in response.objects:
    print(result.properties['text'], result.metadata.score)
print(response.objects)

responseTotal = docs.query.fetch_objects(
    limit=1000,
)

for o in responseTotal.objects:
    print(o.properties)

for collection in client.collections.list_all():
    print(collection)


imageCollection = client.collections.get("ImagesCollection")
# resImages = imageCollection.query.fetch_objects(
#     include_vector=True,
#     limit=1000,
# )
# for img in resImages.objects:
#     print(img.vector)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Folder path
folder_path = "/Users/dmytrookhrymenko/vector_dbs/weaviate-import/images"

# Manually embed the text query
query_text = "dessert"
inputs = processor(text=[query_text], return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model.get_text_features(**inputs)
query_embedding = outputs[0].detach().numpy().tolist()

resAppleAndKiwi = imageCollection.query.near_vector(
    near_vector=query_embedding,
    limit=1000,
    target_vector="text_vector",
    return_metadata=wvc.query.MetadataQuery(distance=True),
)
for image in resAppleAndKiwi.objects:
    print(image.properties['metadata'], image.metadata)
    print("---------")

# print(json.dumps(response.objects, indent=4))
client.close()