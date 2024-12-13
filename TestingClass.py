from typing import Callable, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class TestLLM:


    def __init__(self, llm: Any, qa: Any, prompts: Dict):
        self.QA = qa
        self.llm = llm
        self.prompts = prompts
        self.threshold = 1

    async def check_answer(self, question: str, answer: str, context: str) -> str:
        """
        Critical LLM проверяет ответ от QA модели
        Args:
            question: Вопрос, на который нужно ответить
            answer: Ответ от QA модели
            context: Контекст вопроса
        Returns:
            Critical LLM answer: "1", если ответ соответствует вопросу, "0" иначе
        """
        critical_llm_chain = (
                ChatPromptTemplate.from_template(self.prompts['CriticalLLM'])
                | self.llm
                | StrOutputParser()
        )
        prompt_data = {
            "question": question,
            "answer": answer,
            "context": context
        }
        return critical_llm_chain.invoke(prompt_data)

    async def __call__(self, question: str, context: str, test_answer: str='', metric_callbacks: Dict[str, Callable]=None) -> \
    Dict[str, Any]:
        """
        Подсчет метрик.

        Args:
            question (str): Вопрос, на который нужно ответить.
            context (str): Контекст вопроса.
            test_answer (str): Тестовый ответ.
            metric_callbacks (Dict[str, Callable]): Словарь функций для подсчета метрик.

        Returns:
            Dict[str, Any]: Словарь, содержащий подсчитанные метрики.
        """
        test_report = {}
        answer = self.QA.ask(question)
        critical_llm_response = await self.check_answer(question, answer, context)
        counter = 0
        while counter < self.threshold:
            if critical_llm_response in ("1", "0"):
                test_report["critical_llm_answer"] = critical_llm_response
                break
            else:
                critical_llm_response = await self.check_answer(question, answer, context)
                counter += 1
        else:
            test_report["critical_llm_answer"] = "LLM не смогла оценить выход модели"

        if test_answer:
            for metric_name, metric_function in metric_callbacks.items():
                if callable(metric_function):
                    try:
                        test_report[metric_name] = metric_function(answer, test_answer)
                    except Exception as e:
                        test_report[f"{metric_name}_error"] = str(e)
                else:
                    test_report[f"{metric_name}_error"] = "Функция для подсчета метрики не вызываема"

        return test_report
