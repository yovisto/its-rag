import torch, datetime, csv, json, os
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index import load_index_from_storage
from llama_index import Document
from flask import Flask, Response, request, render_template
from pathlib import Path

#os.environ["TOKENIZERS_PARALLELISM"]="False"

print ("Welcome!", flush=True)

TEST=True

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


docs={}
texts={}
all_docs=[]

print ("Loading data", flush=True)

if TEST:
    f = 'test-data.csv'
else:
    f = 'workspace_data-public-only-chatbot-corrected.csv'

with open(f) as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar='"',doublequote = True, skipinitialspace = False)
    next(reader, None)
    for row in reader:
        did = row[0]
        dtitle = row[1]
        dtext = dtitle + ": " + row[2]
        docs[did]=dtitle
        texts[did]=row[2]
        ldoc = Document(text=dtext, id_=did)
        #print(ldoc.text)
        #print(dtext)
        all_docs.append(ldoc)
        #print(did, title)

print ("Loading model", flush=True)

#print (len(all_docs))
#for doc in all_docs:
#    print(doc["Text"])

MODEL = "HuggingFaceH4/zephyr-7b-beta"

llm = HuggingFaceLLM(
    model_name=MODEL,
    tokenizer_name=MODEL,
#    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    #max_new_tokens=64,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"do_sample":True, "temperature": 0.7, "top_k": 50, "top_p": 0.95},
#    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

print ("Loading embeddings", flush=True)

model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=model, chunk_size=512)

def createIndex():
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
            all_docs,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=True,
        )
    print ("index created")
    storage_context.persist(persist_dir=f"./storage/index" + str(int(datetime.datetime.now().timestamp())))
    print ("persisted")
    return cur_index

def loadIndex():
    print ("Loading index", flush=True)
    storage_context = StorageContext.from_defaults(persist_dir="./storage/index")
    print ("Storage created", flush=True)
    cur_index = load_index_from_storage(storage_context, service_context=service_context, show_progress=True)
    print ("Index loaded", flush=True)
    return cur_index

if TEST:
    cur_index=createIndex()
else:
    cur_index=loadIndex()

print ("Generating node map", flush=True)
node_map={}

#print (cur_index.ref_doc_info)
for key, item in cur_index.ref_doc_info.items():
    #print (item)
    for node_id in item.node_ids:
        #print (node_id)
        node_map[node_id]=key
    #print (docs[item])

print ("Nodes")
print (len(node_map), flush=True)


new_summary_tmpl_str = (
    "My database is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given my database information, answer the query in German.\n"
    "Query : {query_str}\n"
    "Answer: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

query_engine = cur_index.as_query_engine(response_mode="compact", similarity_top_k=4, streaming=True,)
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": new_summary_tmpl}
)

#q="Was ist die Gretchenfrage?" # ok

#response_stream = query_engine.query(q)
#response_stream.print_response_stream()


app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

breakList=['---------------------','Query', 'Query:']

def empty():
    yield "Du hast nichts eingegeben."

@app.route('/chain', methods=['POST'])
def query():

    # Get the payload from the request
    payload = request.json

    print ("PAYLOAD: ", payload, flush=True)
    
    question = request.json['prompt']

    if question.strip()=="":
        return Response(empty(), mimetype="text/event-stream")

    backload=[]

    nodes = query_engine.retriever.retrieve(question)
    for n in nodes:
        print (n)
        doc_id = node_map[n.node_id]
        print ("doc:", doc_id)
        print (docs[doc_id])
        dd={}
        dd["id"]=doc_id
        dd["title"]=docs[doc_id]
        dd["text"]=texts[doc_id]
        backload.append(dd)

    qb = QueryBundle(question)
    streaming_response=query_engine.synthesize(qb, nodes)


    # Now, query returns a StreamingResponse object
    #streaming_response = query_engine.query(question)

    def response_stream():
        yield "<<<>>>" + json.dumps({"results":backload}) + "<<<>>>"
        for text in streaming_response.response_gen:
            print("l: "+ text + "\n")
            if text.strip() in breakList:
                break
            yield text
        yield " "
        yield "EEEOOOFFF"
    return Response(response_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    #app.run(threaded=True, debug=True)
    app.run(threaded=True, host="0.0.0.0")
