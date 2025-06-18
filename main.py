import sys
import weaviate
from weaviate.collections import Collection
from weaviate.classes.config import Property, DataType
from weaviate.util import generate_uuid5
import weaviate.classes.config as wvcc
from google.drive import get_google_drive_files
from google.doc import extract_google_doc
from textsplitter import split_ssplitter, split_langchain
from embedder import embed_with_ollama


client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051
)

try:
    client.is_ready()
except Exception as e:
    print('Error: %s', repr(e))
    sys.exit(1)

print(f"Collections: {client.collections.list_all()}")

collection_name = "GoogleDocs"


if collection_name in client.collections.list_all():
    client.collections.delete(collection_name)

if collection_name not in client.collections.list_all():
    client.collections.create(
        name=collection_name,
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_ollama(api_endpoint="http://host.docker.internal:11434"),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="type", data_type=DataType.TEXT),
        ],
    )

collection: Collection = client.collections.get(collection_name)

g_drivr_files = get_google_drive_files()
if g_drivr_files is not None:
    for file in g_drivr_files:
        data = extract_google_doc(file['id'])
        split_data = split_langchain(data)

        with collection.batch.fixed_size(batch_size=50) as batch:
            for chunk in split_data:
                obj_uuid = generate_uuid5(chunk)
                vector = embed_with_ollama(chunk)
                batch.add_object(
                    properties={
                        "text": chunk,
                        "source": file["name"],
                        "type": "google doc native"
                    },
                    vector=vector
                )
                if batch.number_errors > 10:
                    print("Batch import stopped due to excessive errors.")
                    break


failed_objects = collection.batch.failed_objects
if failed_objects:
    print(f"Number of failed imports: {len(failed_objects)}")
    print(f"First failed object: {failed_objects[0]}")

print(f"Collections: {client.collections.list_all()}")


client.close()