import os
import sys
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
import weaviate
from weaviate.classes.config import Configure, DataType, Property

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


# 1. Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Folder path
folder_path = "/Users/dmytrookhrymenko/vector_dbs/weaviate-import/images"

# 3. Optional metadata per file (or read from CSV/JSON)
image_metadata = {
    "apple.jpg": {"title": "Apple", "description": "A red apple"},
    "banana.jpg": {"title": "Banana", "description": "A ripe banana"},
    "cherry.jpg": {"title": "Cherry", "description": "Red cherries with green stems"},
    "chickoo.jpg": {"title": "Chickoo", "description": "Yellow chickoos"},
    "grapes.jpg": {"title": "Grapes", "description": "Two bunches of grapes with green leafs"},
    "kiwi.jpg": {"title": "Kiwi", "description": "A green kiwi with black seeds"},
    "mango.jpg": {"title": "Mango", "description": "A juicy mango"},
    "orange.jpg": {"title": "Orange", "description": "An orange"},
    "strawberry.jpg": {"title": "Strawberry", "description": "A red strawberry with green tail"},
}

# 4. Collect images and embeddings
embeddings = []

for file in os.listdir(folder_path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, file)
        image = Image.open(image_path).convert("RGB")

        text = image_metadata.get(file, {}).get("description", file)
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

        # Get the multimodal embedding
        outputs = model(**inputs)
        # embedding = outputs.pooler_output[0].detach().numpy().tolist()

        image_embedding = outputs.image_embeds[0].detach().numpy().tolist()  # ✅ this is your vector
        text_embedding = outputs.text_embeds[0].detach().numpy().tolist()

        embeddings.append({
            "image_path": image_path,
            "metadata": image_metadata.get(file, {}),
            "image_embedding": image_embedding,
            "text_embedding": text_embedding
        })

# 5. You now have structured embeddings for DB insert
print(json.dumps(embeddings[0], indent=4))



client.collections.delete("ImagesCollection")

client.collections.create(
    "ImagesCollection",
    properties=[
        Property(name="image_path", data_type=DataType.TEXT),
        Property(name="metadata", data_type=DataType.TEXT),
    ],
    vectorizer_config=[
        Configure.NamedVectors.none(
            name="text_vector",
        ),
        # Configure.NamedVectors.multi2vec_clip(
        #     name="text_vector",
        #     text_fields=["image_path", "metadata"],
        #     vectorize_collection_name=False
        # ),
        Configure.NamedVectors.none(
            name="image_vector",
        ),
        # Configure.NamedVectors.multi2vec_clip(
        #     name="image_vector",
        #     image_fields=["image_path", "metadata"],
        #     vectorize_collection_name=False
        # )
    ],
)

collection = client.collections.get("ImagesCollection")

# Batch import into weaviate
with collection.batch.fixed_size(batch_size=50) as batch:
    batch.batch_size = 20  # tweak as needed

    for img_data in embeddings:  # your list of images and metadata
        batch.add_object(
            properties={
                "image_path": img_data["image_path"],
                "metadata": json.dumps(img_data["metadata"]),
            },
            vector={
                "text_vector": img_data['text_embedding'],
                "image_vector": img_data['image_embedding'],
            }
        )
client.close()


# from transformers import BlipProcessor, BlipModel
# from PIL import Image
# import torch
#
# # Load model
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-text-model")
# model = BlipModel.from_pretrained("Salesforce/blip-image-text-model")
#
# def get_blip_embeddings(image_path: str, text: str):
#     image = Image.open(image_path).convert('RGB')
#     inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#         image_embeds = outputs.image_embeds.squeeze().cpu().numpy()
#         text_embeds = outputs.text_embeds.squeeze().cpu().numpy()
#
#     return {
#         "image_vector": image_embeds.tolist(),
#         "text_vector": text_embeds.tolist()
#     }
#
#
#
# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel
#
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#
# def get_image_embedding(image_path):
#     image = Image.open(image_path)
#     inputs = processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         embeddings = model.get_image_features(**inputs)
#     # Normalize the embeddings vector
#     embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
#     return embeddings[0].cpu().numpy().tolist()
#
#
#
#
#