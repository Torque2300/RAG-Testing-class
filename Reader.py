from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from markdown import markdown
from langchain.schema import Document


class Reader:
    def __init__(self, path: Path):
        self.path = Path(path)

    @staticmethod
    def extract_text_from_md(path: Path) -> str:
        """
        Читает файл .md, преобразует Markdown в HTML, а затем очищает текст от разметки.
        """
        with path.open('r', encoding='utf-8') as file:
            md_content = file.read()
            html_content = markdown(md_content)
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text().strip()

    @staticmethod
    def convert_to_documents(md_data: List[dict]) -> List[Document]:
        """
        Преобразует извлечённые данные .md в документы.
        """
        documents = []
        for item in md_data:
            document = Document(page_content=item['text'], metadata={"source": item['file_path']})
            documents.append(document)
        return documents

    def get_md_files_content(self) -> List[dict]:
        """
        Обходит все файлы .md в заданной папке и подпапках, извлекает текст.
        """
        data = []
        for file_path in self.path.rglob('*.md'):
            relative_path = file_path.relative_to(self.path)
            text = self.extract_text_from_md(file_path)
            if text:
                data.append({"file_path": str(relative_path), "text": text})
        return data

    def load(self) -> List[Document]:
        """
        Загружает данные из .md файлов и преобразует их в документы.
        """
        md_data = self.get_md_files_content()
        return self.convert_to_documents(md_data)
