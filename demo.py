import os
from openai import AzureOpenAI
from mem0 import Memory

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set the OpenAI API key
os.environ["AZURE_OPENAI_API_KEY"] = "2fc02fc43f394566ad206bea5a3dad34"
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
class PersonalTravelAssistant:
    def __init__(self):
        self.client = AzureOpenAI()
        self.memory = Memory.from_config(config)
        self.messages = [{"role": "system", "content": "You are a personal AI Assistant."}]

    def ask_question(self, question, user_id):
        # Fetch previous related memories
        previous_memories = self.search_memories(question, user_id=user_id)
        prompt = question
        if previous_memories:
            prompt = f"User input: {question}\n Previous memories: {previous_memories}"
        self.messages.append({"role": "user", "content": prompt})

        # Generate response using GPT-4o
        response = self.client.chat.completions.create(
            model="btree",
            messages=self.messages
        )
        answer = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": answer})

        # Store the question in memory
        self.memory.add(question, user_id=user_id)
        return answer

    def get_memories(self, user_id):
        memories = self.memory.get_all(user_id=user_id)
        return [m['text'] for m in memories]

    def search_memories(self, query, user_id):
        memories = self.memory.search(query, user_id=user_id)
        return [m['text'] for m in memories]

# Usage example
user_id = "traveler_123"
ai_assistant = PersonalTravelAssistant()

def main():
    file_path = './output.txt'
    while True:
        question = input("Question: ")
        if question.lower() in ['q', 'exit']:
            print("Exiting...")
            break

        result=[]
        answer = ai_assistant.ask_question(question, user_id=user_id)
        print(f"Answer: {answer}")
        memories = ai_assistant.get_memories(user_id=user_id)
        result.append((question, answer, memories))
        print("Memories:")
        for memory in memories:
            print(f"- {memory}")
        print("-----")

        # 打开文件并写入列表
        with open(file_path, 'w') as file:
            for item in result:
                file.write(f"{item}\n")
if __name__ == "__main__":
    main()
