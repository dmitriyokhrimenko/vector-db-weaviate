import sys
from weaviate.classes.config import Configure, DataType, Multi2VecField, Property
import weaviate

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

client.collections.delete("DemoCollection")

client.collections.create(
    "DemoCollection",
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="poster", data_type=DataType.BLOB),
    ],
    vectorizer_config=[
        Configure.NamedVectors.multi2vec_clip(
            name="title_vector",
            # Define the fields to be used for the vectorization - using image_fields, text_fields, video_fields
            image_fields=[
                Multi2VecField(name="poster", weight=0.9)
            ],
            text_fields=[
                Multi2VecField(name="title", weight=0.1)
            ]
        )
    ],
    # Additional parameters not shown
)

collections = client.collections.list_all()
print(collections)
for name, config in collections.items():
    print(f"Collection: {name}")
    print(f"  Properties: {[prop.name for prop in config.properties]}")
    if config.vectorizer_config:
        print(f"  Vectorizer: {config.vectorizer_config.vectorizer}")
    print()





collection = client.collections.get("DemoCollection")

with collection.batch.fixed_size(batch_size=200) as batch:
    for src_obj in source_objects:
        poster_b64 = url_to_base64(src_obj["poster_path"])
        weaviate_obj = {
            "title": src_obj["title"],
            "poster": poster_b64  # Add the image in base64 encoding
        }

        # The model provider integration will automatically vectorize the object
        batch.add_object(
            properties=weaviate_obj,
            # vector=vector  # Optionally provide a pre-obtained vector
        )



client.close()