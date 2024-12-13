from pathlib import Path

from Reader import Reader
from config import config

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import time


class RetrieverQA:


    def __init__(self, llm, prompts: dict, embeddings=None, vectorstore=None, chunk_size=750, chunk_overlap=100):
        self.llm = llm
        self.prompts = prompts
        self.embeddings = embeddings
        if embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        self.vectorstore = vectorstore
        if self.vectorstore is None:
            if Path(config.PROJECT_DIR / 'data/index.faiss').exists():
                self.vectorstore = FAISS.load_local(
                    folder_path=config.PROJECT_DIR / 'data',
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
            elif (Path(config.PROJECT_DIR / 'data/documents').exists()
                  and Path(config.PROJECT_DIR / 'data/documents').is_dir()
                  and any(Path(config.PROJECT_DIR / 'data/documents').iterdir())):
                reader = Reader(config.PROJECT_DIR / "data/documents")
                docs = reader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                    is_separator_regex=False,
                    length_function=len
                )
                docs = text_splitter.split_documents(docs)
                self.vectorstore = FAISS.from_documents(
                    documents=docs,
                    embedding=self.embeddings
                )
                self.vectorstore.save_local(config.PROJECT_DIR / 'data')
            else:
                raise FileNotFoundError(f"Не удалось создать Vectorstore по пути: {config.PROJECT_DIR / "data/index.faiss"}.")


    def ask(self, question: str, k: int=5) -> str:
        start_time = time.time()
        retriever = self.vectorstore.as_retriever(search_kwargs={'k': k})
        def format_docs(docs) -> str:
            """Форматирует документы для передачи их в LLM.

            Args:
                docs (List[Document]): Список документов.

            Returns:
                str: Текст, объединяющий содержимое всех документов.
            """
            res = "\n\n".join(doc.page_content for doc in docs)
            return res

        prompt = """Ты являешься ассистентом в чатботе компании ЦФТ (Центр финансовых технологий).
            Используй следующие документы для ответа на вопрос пользователя.
            Если ты не знаешь ответа на вопрос, то ответь пользователю: "Я не знаю ответа на ваш вопрос."
            Используй максимум 3 предложения для ответа и следи за тем, чтобы ответ был точным.
            Документы: {documents}
            Вопрос: {question}
            Ответ:
            Отвечай только на русском языке, не вставляя английские слова. 
            """
        qa_chain = (
            {"documents": retriever | format_docs, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(prompt)
            | self.llm
            | StrOutputParser()
        )
        result = qa_chain.invoke(question)
        end_time = time.time()
        print(end_time - start_time)
        return result

