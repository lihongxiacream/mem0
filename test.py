
from mem0 import Memory
import os
os.environ["AZURE_OPENAI_API_KEY"] = "2fc02fc43f394566ad206bea5a3dad34"

# Needed to use custom models
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://btree.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"

config = {
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "multi-qa-MiniLM-L6-cos-v1"
        }
    },
    "llm": {
            "provider": "azure_openai",
            "config": {
                "model": "btree",
                "temperature": 0.7,
                "max_tokens": 2000,
            }
        }
}

m = Memory.from_config(config)


result = m.add("Likes to play cricket on weekends", user_id="alice", metadata={"category": "hobbies"})
print(result)


# Get all memories
all_memories = m.get_all()
print(all_memories)

related_memories = m.search(query="What are Alice's hobbies?", user_id="alice")
print(related_memories)

m.delete(memory_id="8b3f6e79-9ee2-4d00-ba34-a9af6c46ffba") # Delete a memory
m.delete_all(user_id="alice") # Delete all memories
