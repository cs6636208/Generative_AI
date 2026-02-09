# !pip install -U langchain langchain-openai langchain-community langchain_classic langgraph pyowm
from langchain_openai import ChatOpenAI    ## openAI's chat model
from google.colab import userdata

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5)               ## พารามิเตอร์สำหรับตั้งค่าความคิดสร้างสรรในการตอบ

response = llm.invoke("กรุงเทพมหานครอยู่ที่ไหน")   ## รัน llm และเก็บผลลัพธ์ใน response
print(response)    ## return object
# print(response.content) ## return string

"""## Prompt Template"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate   ## สร้าง prompt template

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5)

# เตรียม Prompt Template โดยรับค่าจากตัวแปร {topic} และ {length} จากผู้ใช้
template = ChatPromptTemplate.from_template("อธิบายเกี่ยวกับ {topic} ความยาวไม่เกิน {length} ประโยค")

# ใส่ค่าตัวแปร {topic} และ {length} ให้ prompt
prompt = template.invoke({"topic": "machine learning", "length":5})

# รัน LLM
response = llm.invoke(prompt)
print(response.content)

"""## function calling"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5)

def summarize_text(text: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "ช่วยสรุปข้อความ: {text} ภายใน 1 ประโยค"
    )
    filled_prompt = prompt.invoke({"text": text})
    response = llm.invoke(filled_prompt)
    return response.content

input_text = "vLLM เป็นระบบที่ช่วยให้เราสามารถรันโมเดลภาษาขนาดใหญ่ได้อย่างมีประสิทธิภาพ โดยใช้เทคนิคต่างๆ เช่น การจัดการหน่วยความจำ GPU และการทำ Quantization"
summary = summarize_text(input_text)

print("สรุป:", summary)

"""## Streamming output
 stream ผลลัพธ์ (ค่อยๆ ทยอยแสดงตัวอักษรออกมาทันทีที่ประมวลผลเสร็จ) แทนที่จะรอให้สรุปจบทั้งหมดก่อน
"""

from openai import OpenAI
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5)

def summarize_text_streaming(text: str) -> str:
    prompt = f"ช่วยสรุปข้อความ: {text} ภายใน 1 ประโยค"
    final_text = []

    # รัน llm.stream
    for chunk in llm.stream(prompt):
        content = chunk.content
        print(content, end="", flush=True)
        final_text.append(content)
    return "".join(final_text)

input_text = "vLLM เป็นระบบที่ช่วยให้เราสามารถรันโมเดลภาษาขนาดใหญ่ได้อย่างมีประสิทธิภาพ โดยใช้เทคนิคต่างๆ เช่น การจัดการหน่วยความจำ GPU และการทำ Quantization"

summary = summarize_text_streaming(input_text)

"""## Chain & Langchain Expression Language (LCEL)
การนำองค์ประกอบต่างๆ มาเชื่อมต่อกันเพื่อกำหนดลำดับการทำงานของแอพพลิเคชั่น โดยใช้แนวคิด input, process, output
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5)

template = ChatPromptTemplate.from_template("อธิบายเกี่ยวกับ {topic} ความยาวไม่เกิน {length} ประโยค")

chain = template | llm            ## สร้าง chain โดยเริ่มจากเตรียม template เสร็จแล้วก็รัน llm
response = chain.invoke({"topic":"machine learning", "length":"3"})   ## ส่ง input ไปยัง chain
print(response.content)

"""### Message Type

System Message  กำหนดพฤติกรรม บทบาท ข้อมูลบริบทที่เกี่ยวข้องให้กับ AI

Human Message ข้อความที่ผู้ใช้งานส่งให้ AI เพื่อโต้ตอบ

AI Message  ข้อความตอบกลับจาก AI
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5)

# สร้าง Prompt Template โดยสร้างตัวแปร language และ text เพื่อรับข้อมูลจากผู้ใช้
template = ChatPromptTemplate.from_messages([
    ("system", "คุณเป็นผู้เชี่ยวชาญด้าน {expert}"),                ## role play
    ("human", "ช่วยอธิบายเกี่ยวกับเรื่อง {topic} ใน 3 ประโยค")     ## query
])

chain = template | llm         ## chain

# response = chain.invoke({"expert":"data science",  "topic":"LLM"})

for chunk in chain.stream({"expert": "Data Science", "topic": "LLM"}):       ### streaming output for chain
   print(chunk.content, end="", flush=True)

print(response.content)

"""## OutputParser
แปลงเอาท์พุตให้อยู่ใน format ที่ต้องการ เช่น string, list, json
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5)

parser = StrOutputParser()  ## แปลงผลลัพธ์ของ llm เป็นข้อความ

# สร้าง Prompt Template โดยสร้างตัวแปร language และ text เพื่อรับข้อมูลจากผู้ใช้
template = ChatPromptTemplate.from_messages([
    ("system", "คุณเป็นผู้เชี่ยวชาญด้าน {expert}"),
    ("human", "แนะนำเมนู {menu} จำนวน {amount} รายการ")
])

chain = template | llm | parser # input -> Process -> Output

response = chain.invoke({"expert":"เชฟอาหารเหนือ", "menu":"อาหารเหนือ", "amount": 3})
print(response) # เปลี่ยน format เป็น String

"""### Convert output to list"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5, max_tokens = 4096)

parser = CommaSeparatedListOutputParser()   ## แปลงผลลัพธ์เป็น list

# สร้าง prompt template
template = ChatPromptTemplate.from_messages([
    ("system", "คุณเป็นผู้เชี่ยวชาญด้าน {expert} "),
    ("human", "แนะนำเมนู {menu} จำนวน {amount} รายการ")
])

chain = template | llm | parser

response = chain.invoke({"expert":"เชฟอาหารเหนือ", "menu":"อาหารเหนือ", "amount": 3})

for i, item in enumerate(response):
    print(f"{i+1}. {item}")

"""## Prompt engineering"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5, max_tokens = 4096)

parser = CommaSeparatedListOutputParser()

# เตรียม prompt template โดยสั่งให้ตอบเฉพาะ ชื่อเมนู และลักษณะเด่นของเมนู เท่านั้น
template = ChatPromptTemplate.from_messages([
    ("system", "คุณเป็นผู้เชี่ยวชาญด้าน {expert} ตอบเฉพาะ ชื่อเมนู ตามด้วยเครื่องหมาย : และ ลักษณะเด่น เท่านั้น"),
    ("human", "แนะนำเมนู {menu} จำนวน {amount} รายการ")
])

chain = template | llm | parser

response = chain.invoke({"expert":"เชฟอาหารเหนือ", "menu":"อาหารเหนือ", "amount": 3})

for i, item in enumerate(response):
    print(f"{i+1}. {item}")

"""## Output as JSON format"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0)

parser = JsonOutputParser()

# สร้าง template ให้ตอบเป็น JSON format

template = ChatPromptTemplate.from_messages([
    ("system", "คุณเป็นผู้เชี่ยวชาญด้าน {expert}. ตอบกลับในรูปแบบ JSON เท่านั้น"),
    ("human", "แนะนำเมนู {menu} จำนวน {amount} รายการ")
])

chain = template | llm | parser

response = chain.invoke({"expert":"เชฟอาหารเหนือ", "menu":"อาหารเหนือ", "amount": 3})
print(response)

"""## Tools  
    ฟังก์ชันภายนอกที่อนุญาตให้ LLM เรียกใช้ได้
"""

pip install tavily-python   ## install tavily web search

"""## Generate context
search ผลลัพธ์ใน internet จากนั้นสร้างเป็นบริบทให้กับ LLM ตอบคำถาม
"""

from langchain_openai import ChatOpenAI
from google.colab import userdata
from langchain_community.tools.tavily_search import TavilySearchResults
import os

os.environ["TAVILY_API_KEY"] = userdata.get("TAVILY_API_KEY")

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5, max_tokens=8000)

search = TavilySearchResults(max_results=3)
# 1. Search
query = "นายกรัฐมนตรีคนล่าสุดของประเทศไทยคือใคร"
docs = search.invoke({"query": query})

# 2. Combine search results
context = "\n".join([d["content"] for d in docs])

prompt = f"""
ใช้ข้อมูลต่อไปนี้เพื่อตอบคำถาม:

{context}

คำถาม: {query}
"""

response = llm.invoke(prompt)
## print(context)
print(response.content)

"""## React agent
สร้าง tool ผูกเข้ากับ agent เพื่อให้คิดก่อนตอบ และเรียกใช้ tool ได้เอง เพื่อให้ตอบได้แม่นยำมากขึ้น
![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*kvQVlT9Wi3TX-hBDHA4DIQ.png)

"""

import os
from google.colab import userdata
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Setup
os.environ["TAVILY_API_KEY"] = userdata.get("TAVILY_API_KEY")

# 2. LLM Setup
llm = ChatOpenAI(
    base_url='https://api.opentyphoon.ai/v1',
    #model="typhoon-v2.5-30b-a3b-instruct",
    model="typhoon-v2.1-12b-instruct",
    api_key=userdata.get('TYPHOON_KEY'),
    temperature=0
)

# 3. Tool definition
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

# 4. สร้าง agent โดยผูก llm เข้ากับ tools
agent = create_react_agent(llm, tools)

query = "นายกรัฐมนตรีคนล่าสุดของประเทศไทยคือใคร "
system_msg = "คุณเป็นผู้ช่วยค้นหาข้อมูล ตอบกลับเป็นภาษาไทยที่สรุปใจความสำคัญ"

inputs = {"messages": [SystemMessage(content=system_msg), HumanMessage(content=query)]}

print("กำลังค้นหาข้อมูลและสรุปคำตอบ...")

# รัน agent แล้วรอคำตอบจาก agent
result = agent.invoke(inputs)

# ค้นหาข้อความสุดท้ายใน List ของ messages
final_message = result["messages"][-1]

print(final_message.content)

"""## Memory
   ช่วยให้ LLM สามารถจดจำบริบท (context) หรือบทสนทนา (conversation) ของผู้ใช้ได้

## chatbot โดยปราศจาก memory
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from google.colab import userdata

# 1. ---- Setup LLM ----
llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    model="typhoon-v2.5-30b-a3b-instruct",
    api_key=userdata.get('TYPHOON_KEY'),
    temperature=0.5,
    max_tokens=4096
)

# 2. ---- Prompt Setup (ตัด MessagesPlaceholder ออก) ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "คุณคือผู้ช่วยอัจฉริยะที่ตอบเป็นภาษาไทยอย่างสุภาพ"),
    ("human", "{input}")
])

# 3. # ส่ง input เข้า prompt และต่อไปยัง llm โดยตรง
chain = prompt | llm

# 4. ---- Interactive Chat Function ----
def start_chatbot():
     while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":        ##  พิมพ์ 'exit' เพื่อจบสนทนา
            break

        response = chain.invoke({"input": user_input})

        print(f"AI: {response.content}\n")

if __name__ == "__main__":
    start_chatbot()

"""## ConversationBufferMemory
เพิ่มหน่วยความจำเพื่อบันทึก conversation history ระหว่างผู้ใช้กับ AI ล่าสุด
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory  # Changed to BufferMemory
from langchain_core.runnables import RunnablePassthrough
from google.colab import userdata

# 1. ---- Setup LLM ----
llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    model="typhoon-v2.5-30b-a3b-instruct",
    api_key=userdata.get('TYPHOON_KEY'),
    temperature=0.5,
    max_tokens=4096
)

# 2. เตรียม Buffer Memory
memory = ConversationBufferMemory(memory_key="chat_history",  k=5, return_messages=True)  # k = 5 หมายถึง เก็บ 5 บทสนทนาล่าสุด

# 3. ---- Prompt Setup ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "คุณคือผู้ช่วยอัจฉริยะที่จดจำรายละเอียดได้แม่นยำ และตอบเป็นภาษาไทยอย่างสุภาพ"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 4. ฟังก์ชันโหลด chat history จาก memory
def get_chat_history(input_data):
    return memory.load_memory_variables({})["chat_history"]

chain = RunnablePassthrough.assign(chat_history=get_chat_history) | prompt | llm

def start_chatbot():
    print(f"--- Chatbot System Ready (Buffer Memory) ---")
    print("พิมพ์ 'exit' เพื่อเลิกคุย\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = chain.invoke({"input": user_input})
        memory.save_context({"input": user_input}, {"output": response.content})

        print(f"AI: {response.content}\n")

if __name__ == "__main__":
    start_chatbot()

    print("-" * 30)
    curr_memory = memory.load_memory_variables({})
    print(f"Current Memory (Exact Buffer):")
    for msg in curr_memory['chat_history']:
        print(f"{type(msg).__name__}: {msg.content}")
    print("-" * 30 + "\n")

"""## Retrieval Augmented Generation (RAG)
 - document loader
 - text splitter
 - Embedding & indexing
 - retrival
 ![](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F6uvncpzpbdvwgesdfv1t.jpeg)

### Document Loader
"""

from langchain_community.document_loaders import TextLoader
loader = TextLoader("/content/data.txt", encoding="utf-8")
documents = loader.load()
print(documents)

"""### Text Splitter"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("/content/data.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)
    print('*'*20)

"""### Embedding & Vector Stores"""

!pip install -U sentence-transformers faiss-cpu   ## ติดตั้ง transformer lib. และ faiss db

"""### Embedding & Vectorstore
    เลือกใช้ โมเดล BGE-M3 embedding หรือตัวอื่นเช่น openAIembedding
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# BGE-M3 embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True} )

data = embeddings.embed_documents(["luepol kmutnb"])  ## document sample
print(data)   ## print document vector

"""### Chuck indexing"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# BGE-M3 embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True} )

vectorstore = FAISS.from_documents(chunks, embeddings)  ## embedding chucks and indexing

retriever = vectorstore.as_retriever()    ## create chuck retriever

"""### Chunk Retreival"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough    # Corrected import path

prompt = ChatPromptTemplate.from_messages([
    ("system", "ใช้ข้อมูลจากเอกสาารเพิ่อตอบคำถามเท่านั้น ตอบแบบสุภาพเป็นกันเอง"),
    ("human", "คำถาม {question}, ข้อมูลที่เกี่ยวข้อง {context}")
])

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5, max_tokens = 2048)

parser = StrOutputParser()

chain = ({"context": retriever, "question": RunnablePassthrough()} ## รอรับคำถามจากผู้ใช้ และ context รอรับจาก retriver
         | prompt
         | llm
         | parser) # Using parser instead of StrOutputParser class
result = chain.invoke("เจ้าของบริษัท ABC คือใคร และจะติดต่อได้อย่างไร")
print(result)

"""## กรณีสกัดเอกสารจากไฟล์ pdf
   ยอดนิยม [pymupdf](https://pymupdf.readthedocs.io/en/latest/) หรือ [docling](https://www.docling.ai/)
   
"""

!pip install pymupdf pythainlp

import fitz
import unicodedata
import re
from pythainlp.util import normalize

doc = fitz.open("/content/4-2.pdf")
text = ""

for page_num in range(len(doc)):
    page = doc.load_page(page_num)  # โหลดหน้า
    t = page.get_text()         # ดึงข้อความ
    t = normalize(t)            # normalize text
    t += f"\n--- หน้าที่ {page_num + 1} ---\n"  ## insert page number
    text += t

doc.close()
print(text)

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

doc = fitz.open("/content/4-2.pdf")

docs = []

for i, page in enumerate(doc):
    text = page.get_text()
    docs.append(
        Document(                          ## create document object
            page_content=text,             ## content
            metadata={"page": i+1}         ## meta definition
        )
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)       ## convert docs to chunks

print(f"Total chunks: {len(chunks)}")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# BGE-M3 embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True} )

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # Corrected import path

prompt = ChatPromptTemplate.from_messages([
    ("system", "ใช้ข้อมูลจากเอกสาารเพิ่อตอบคำถามเท่านั้น ตอบแบบสุภาพเป็นกันเอง อ้างอิงเลขหน้าด้วย หากไม่มีข้อมูลให้ตอบว่า ไม่พบข้อมูล อย่าตอบมั่ว"),
    ("human", "คำถาม {question}, ข้อมูลที่เกี่ยวข้อง {context}")
])

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5, max_tokens=4096)

parser = StrOutputParser()

chain = ({"context": retriever, "question": RunnablePassthrough()} # Corrected typo
         | prompt
         | llm
         | parser) # Using parser instead of StrOutputParser class
# result = chain.invoke("เกณฑ์ในการพ้นสภาพนักศึกษามีอะไรบ้าง")
result = chain.invoke("เกียรตินิยมต้องมีเกรดเฉลี่ยเท่าไหร่")
print(result)

"""## บันทึกฐานข้อมูล"""

vectorstore.save_local(folder_path="faiss_index")

"""## โหลดฐานข้อมูล"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. ระบุ Embedding Model (ต้องเป็นตัวเดียวกับตอน Save)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True} )

# 2. โหลด Database จากโฟลเดอร์
new_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = new_db.as_retriever()

"""## ทดสอบ RAG"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "ใช้ข้อมูลจากเอกสาารเพิ่อตอบคำถามเท่านั้น ตอบแบบสุภาพเป็นกันเอง อ้างอิงเลขหน้าด้วย หากไม่มีข้อมูลให้ตอบว่า ไม่พบข้อมูล อย่าตอบมั่ว"),
    ("human", "คำถาม {question}, ข้อมูลที่เกี่ยวข้อง {context}")
])

llm = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model="typhoon-v2.5-30b-a3b-instruct",
                    api_key = userdata.get('TYPHOON_KEY'),
                    temperature=0.5, max_tokens=4096)

parser = StrOutputParser()

chain = ({"context": retriever, "question": RunnablePassthrough()} # Corrected typo
         | prompt
         | llm
         | parser) # Using parser instead of StrOutputParser class
result = chain.invoke("คุณสมบัติ ข้อ ๗ คือ ")
print(result)

"""## LangGraph
![](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F6xiawceubgvr23ai29ab.png)
"""

!pip install langgraph

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from google.colab import userdata

class ChatState(TypedDict):     ### state definition
    input: str                  ## นิยาม input message
    output: str                 ## นิยาม output message

llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key = userdata.get('TYPHOON_KEY'),
    model="typhoon-v2.5-30b-a3b-instruct"
)

def chat_node(state: ChatState):                      ### สร้าง node chat_node
    response = llm.invoke(state["input"])             ## รัน llm โดยอ่าน input จาก state
    return {
        "output": response.content                    ## เขียนผลลัพธ์ llm ไปยัง output ของ state
    }

builder = StateGraph(ChatState)                       #สร้าง graph
builder.add_node("chat", chat_node)                   # add node chat_node ตั้งชื่อ node "chat"
builder.set_entry_point("chat")                       # เริ่มต้นรันที่ "chat" node
builder.add_edge("chat", END)                         # จบการทำงานหลังจากรัน "chat_node เสร็จ

graph = builder.compile()

result = graph.invoke(                                # รัน graph โดยส่ง task ส่งกลับเป็น result
    {"input": "LangGraph คืออะไร"}
)

print(result["output"])                               # แสดงผลลัพธ์ที่ output ของ state

"""### ติดตั้ง pygraphviz เพื่อวาดกราฟของ langgraph"""

!apt-get update
!apt-get install -y graphviz graphviz-dev
!pip install pygraphviz

"""### Display graph"""

from IPython.display import Image, display
display(Image(graph.get_graph().draw_png()))

!pip install pyowm      ## insatll python open weather map

"""## Conditional Workflow with rule"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from google.colab import userdata
from langchain_community.utilities import OpenWeatherMapAPIWrapper
import os

os.environ["OPENWEATHERMAP_API_KEY"] = userdata.get('WEATHER_KEY')

class State(TypedDict):
    input: str
    output: str
    route: str             ## เพิ่ม router message

llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key = userdata.get('TYPHOON_KEY'),
    model="typhoon-v2.5-30b-a3b-instruct"
)

weather_tool = OpenWeatherMapAPIWrapper()

# ---------------- NODES ----------------
def classify_node(state):             ## จำแนกคำถาม
    if "อากาศ" in state["input"]:     ## ถ้า input มีคำว่า "อากาศ"
        return {"route": "weather"}   ## ให้ route = "weather"
    return {"route": "general"}       ## ให้ route = "general"

def route_decision(state):            ## อ่านค่าจาก state
    return state["route"]

def weather_node(state):              ## อ่านสภาพอากาศตามพื้นที่
    try:
        city_info = llm.invoke(f"สกัดชื่อเมือง และชื่อประเทศ แปลงเป็นภาษาอังกฤษจาก: {state['input']}. ตามตัวอย่าง: Bangkok,TH").content.strip()
        data = weather_tool.run(city_info)
        res = llm.invoke(f"สรุปสภาพอากาศเป็นภาษาไทยที่น่าฟัง: {data} เพื่อตอบคำถาม {state['input']}")
        return {"output": res.content}
    except:
        return {"output": "ขออภัยครับ ไม่สามารถระบุข้อมูลพยากรณ์อากาศในพื้นที่ดังกล่าวได้"}

def chat_node(state):                       ## อ่านค่าจาก state ส่งให้ llm ตอบคำถาม
    response = llm.invoke(state["input"])
    return {
        "output": response.content
    }

# -------- build graph --------
builder = StateGraph(State)

builder.add_node("classify", classify_node)
builder.add_node("weather", weather_node)
builder.add_node("chat", chat_node)

builder.set_entry_point("classify")     ## เริ่มต้นรันที่ classify node

builder.add_conditional_edges(          ## นิยาม conditional edges
    "classify",
    route_decision,
    {   "weather": "weather",          ## ถ้าผลลัพธ์จาก route_decision เป็น "weather" เรียก weather node
        "general": "chat"              ## ถ้าผลลัพธ์จาก route_decision เป็น "general" เรียก chat node
    }
)

builder.add_edge("weather", END)
builder.add_edge("chat", END)

graph = builder.compile()
# -------- run --------
result = graph.invoke(
    {"input": "เชียงใหม่ร้อนไหมวันนี้"}   ## เชียงใหม่อยู่ตรงไหน
)

print(result["output"])

from IPython.display import Image, display
display(Image(graph.get_graph().draw_png()))

"""## Conditional Routing with LLM-base"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from google.colab import userdata

os.environ["OPENWEATHERMAP_API_KEY"] = userdata.get('WEATHER_KEY')

class State(TypedDict):
    input: str
    output: str
    route: str         ## add router state

llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key = userdata.get('TYPHOON_KEY'),
    model="typhoon-v2.5-30b-a3b-instruct"
)

# ---------------- NODES ----------------

def classify_node(state):

    prompt = f"""จำแนกข้อความต่อไปนี้ให้เป็น: weather หรือ general
            Text: {state['input']}
            ตอบเพียงคำเดียว
            """
    label = llm.invoke(prompt).content.strip()

    return {"route": label}

def route_decision(state):
    return state["route"]

def weather_node(state):                ## อ่านสภาพอากาศตามพื้นที่
    try:
        city_info = llm.invoke(f"สกัดชื่อเมือง และชื่อประเทศ แปลงเป็นภาษาอังกฤษจาก: {state['input']}. ตามตัวอย่าง: Bangkok,TH").content.strip()
        data = weather_tool.run(city_info)
        res = llm.invoke(f"สรุปสภาพอากาศเป็นภาษาไทยที่น่าฟัง: {data} เพื่อตอบคำถาม {state['input'] }")
        return {"output": res.content}
    except:
        return {"output": "ขออภัยครับ ไม่สามารถระบุข้อมูลพยากรณ์อากาศในพื้นที่ดังกล่าวได้"}

def chat_node(state):
    response = llm.invoke(state["input"])
    return {
        "output": response.content
    }

# -------- build graph --------
builder = StateGraph(State)

builder.add_node("classify", classify_node)
builder.add_node("weather", weather_node)
builder.add_node("chat", chat_node)

builder.set_entry_point("classify")

builder.add_conditional_edges(
    "classify",
    route_decision,
    {   "weather": "weather",
        "general": "chat"
    }
)

builder.add_edge("weather", END)
builder.add_edge("chat", END)

graph = builder.compile()

# -------- run --------
query = input()

result = graph.invoke( {"input": query} )   ## ไมเกรนมีอาการอย่างไร

print(result["output"])

"""## Parallel workflows"""

import os
import requests
from typing import TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from google.colab import userdata

# ================= CONFIG =================

os.environ["TAVILY_API_KEY"] = userdata.get("TAVILY_API_KEY")
os.environ["OPENWEATHERMAP_API_KEY"] = userdata.get("WEATHER_KEY")

llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=userdata.get("TYPHOON_KEY"),
    model="typhoon-v2.5-30b-a3b-instruct",
    temperature=0.5,
    max_tokens=7000,
    streaming=True
)

weather_tool = OpenWeatherMapAPIWrapper()

# ================= STATE =================

class State(TypedDict):
    question: str
    keyword: str
    search: str
    weather: str
    final: str

# ================= NODES =================

def extract_node(state: State):
    prompt = f"""
สกัดชื่อเมืองจากคำถามผู้ใช้

User question:
{state['question']}

ตอบเฉพาะชื่อเมืองภาษาอังกฤษเท่านั้น
"""

    city = llm.invoke(prompt).content.strip()

    return {"keyword": city}


def search_node(state: State):
    search = TavilySearchResults(max_results=3)
    docs = search.invoke({"query": state["question"]})

    return {"search": docs}


def weather_node(state: State):
    if not state.get("keyword"):
        return {"weather": "ไม่พบชื่อเมือง"}

    try:
        data = weather_tool.run(state["keyword"])
        summary = llm.invoke(f"สรุปสภาพอากาศภาษาไทยแบบเป็นกันเอง:\n{data}").content
        return {"weather": summary}

    except Exception as e:
        return {"weather": f"Weather error: {e}"}


def combine_node(state: State):
    prompt = f"""
User question:
{state['question']}

Search info:
{state.get('search')}

Weather:
{state.get('weather')}

สรุปทั้งหมดเป็นคำตอบภาษาไทยแบบเป็นกันเอง
"""

    collected = ""

    print("\n🤖 Bot:\n")

    for chunk in llm.stream(prompt):
        token = chunk.content
        print(token, end="", flush=True)
        collected += token

    print("\n")

    return {"final": collected}

# ================= GRAPH =================

builder = StateGraph(State)

builder.add_node("extract", extract_node)
builder.add_node("search", search_node)
builder.add_node("weather", weather_node)
builder.add_node("combine", combine_node)

builder.add_edge(START, "extract")

builder.add_edge("extract", "search")
builder.add_edge("extract", "weather")

builder.add_edge("search", "combine")
builder.add_edge("weather", "combine")

builder.add_edge("combine", END)

graph = builder.compile()

# ================= RUN =================

if __name__ == "__main__":

    q = input("\nคุณ: ")
    for _ in graph.stream({"question": q}):
        pass

from IPython.display import Image, display
display(Image(graph.get_graph().draw_png()))

"""# พัฒนา simple chatbot ด้วย langgraph เพื่อตอบคำถามจากผู้ใช้ จำแนกเป็น 3 ประเภท ได้แก่
1. คำถามเกี่ยวกับ ระเบียบการศึกษา หรือข้อบังคับของมหาวิทยาลัย ใช้ RAG ในการตอบ
2. คำถามเกี่ยวกับ สภาพอากาศ อุณหภูมิ ยิงไปที่ OpenWeatherMap
3. นอกนั้นให้ llm เป็นคนตอบ
โดยเน้นการตอบเป็นกันเอง และควรมี memory เพื่อเก็บบทสนทนาการพูดคุย จนกระทั่งผู้ใช้พิมพ์ 'เลิก' หรือ 'exit' จบการทำงาน

"""

