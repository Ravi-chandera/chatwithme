from langchain import Pipeline, Prompts, Task, ConversationBufferMemory, ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from streamlit import st

# Configure Hugging Face model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

# Define memory and retrieval chain
memory = ConversationBufferMemory(max_length=10)  # Store 10 recent conversations
retrieval_chain = ConversationalRetrievalChain(memory=memory)

# Define Langchain pipeline
pipeline = Pipeline(
    tasks=[
        Task(
            model=model,
            tokenizer=tokenizer,
            prompts=Prompts(
                input_prompt="Query: {query} {context}",
                output_prompt="Answer: {model_output}",
            ),
            input_key="query",
            output_key="model_output",
            chains=[retrieval_chain],  # Add retrieval chain to the task
        )
    ]
)

# Streamlit interface
st.title("Chatbot")

# User input field
user_query = st.text_input("Ask me anything:")

# Run Langchain pipeline
if user_query:
    with st.spinner("Thinking..."):
        response = pipeline.run(query=user_query, context=memory.get())  # Pass context from memory

    # Update memory with current conversation
    memory.append({"query": user_query, "response": response["model_output"]})

    # Display chatbot response
    st.markdown(f"**Chatbot:** {response['model_output']}")

