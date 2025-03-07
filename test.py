import torch
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index import load_index_from_storage
from llama_index.response.notebook_utils import display_response
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt

MODEL = "HuggingFaceH4/zephyr-7b-beta"
#MODEL = "zephyr-support-chatbot/checkpoint-250"

llm = HuggingFaceLLM(
    model_name=MODEL,
    tokenizer_name=MODEL,
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)



# Create storage context from persisted data
storage_context = StorageContext.from_defaults(persist_dir="./storage/index")


model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=model, chunk_size=512)


# Load index from storage context
index = load_index_from_storage(storage_context, service_context=service_context)



new_summary_tmpl_str = (
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "From this conent of our database and not prior knowledge, answer the query in German.\n"
    "Query : {query_str}\n"
    "Answer: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)


# In[10]:


query_engine = index.as_query_engine(response_mode="compact", similarity_top_k=4, streaming=True)
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": new_summary_tmpl}
)


# In[14]:


q="Was passiert wenn Astronauten rülpsen"


# In[15]:


response = query_engine.query(q)
response.print_response_stream()


# In[16]:


nodes = query_engine.retriever.retrieve(q)
for n in nodes:
    print (n)


# In[ ]:





