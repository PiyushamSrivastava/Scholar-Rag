
import os
from qdrant_client.models import PayloadSchemaType
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Collection configuration
COLLECTION_NAME = "research_papers"
VECTOR_SIZE = 768

def create_payload_indexes():
    """Create indexes for fields used in Qdrant filters."""

    client = get_qdrant_client()

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True
    )

    print("Payload index for 'source' created successfully!")
def get_qdrant_client():
    """Connect to Qdrant Cloud."""

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Qdrant credentials are missing!")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    return client


def create_collection():
    """Create the research papers collection if it doesn't exist."""

    client = get_qdrant_client()

    # Check existing collections
    existing_collections = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists!")
        return

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    print(f"Collection '{COLLECTION_NAME}' created successfully!")


def get_collection_info():
    """Display collection information."""

    client = get_qdrant_client()

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print("\n========== COLLECTION INFORMATION ==========")
    print("Collection name:", COLLECTION_NAME)
    print("Vector size:", collection_info.config.params.vectors.size)
    print("Distance metric:", collection_info.config.params.vectors.distance)
    print("Total points:", collection_info.points_count)


if __name__ == "__main__":

    # 1. Connect to Qdrant
    client = get_qdrant_client()
    create_payload_indexes()

    print("Qdrant connection successful!")

    # 2. Create collection
    create_collection()

    # 3. Display collections
    collections = client.get_collections()

    print("\nAvailable collections:")

    for collection in collections.collections:
        print("-", collection.name)

    # 4. Display collection information
    get_collection_info()
