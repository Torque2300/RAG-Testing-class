import asyncio
from metrics import *
from config import config
from QA import RetrieverQA
from TestingClass import TestLLM
from langchain_ollama.llms import OllamaLLM


async def main():
    qa = RetrieverQA(llm=OllamaLLM(model="llama3.2", temperature=0.4), prompts=config.PROMPTS)
    critical_llm = TestLLM(llm=OllamaLLM(model="akdengi/saiga-llama3-8b:latest", temperature=0.3), qa=qa,
                           prompts=config.PROMPTS)
    context = """Центр Финансовых Технологий – провайдер современных ИТ-решений для участников финансового рынка РФ и СНГ.
    Более 10 лет входит в ТОП-5 крупнейших разработчиков программного обеспечения, действующих на российском рынке,
    ТОП-10 рейтинга высокотехнологичных компаний РФ «ТехУспех».
    - Более 25 лет на рынке ИТ: автоматизация банковской и платежной индустрии, сфер ЖКХ и транспорта
    - 12 офисов в крупных городах России, в том числе: в Москве, Новосибирске, Санкт-Петербурге
    - Топ-3 крупнейших разработчиков ПО (Рейтинг РА «Эксперт»).
    - Топ-3 поставщиков услуг в области IT (Рейтинг CNews Analytics)"""
    metric_callbacks = {"f1": calculate_f1}

    test_report = await critical_llm(
        question="Что такое Центр финансовых технологий?",
        context=context,
        test_answer="Центр Финансовых Технологий – провайдер современных ИТ-решений для участников финансового рынка РФ и СНГ.",
        metric_callbacks=metric_callbacks
    )
    print(test_report)


if __name__ == "__main__":
    asyncio.run(main())
