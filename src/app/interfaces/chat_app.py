#!/usr/bin/env python3
"""
Streamlit веб-интерфейс для AI ассистента трейдера

Использование:
    poetry run streamlit run src/app/chat_app.py
    streamlit run src/app/chat_app.py
"""

import json

import streamlit as st

from src.app.adapters import FinamAPIClient
from src.app.core import call_llm, get_settings


def create_system_prompt() -> str:
    """Создать системный промпт для AI ассистента"""
    return """Ты - AI ассистент трейдера, работающий с Finam TradeAPI.

Когда пользователь задает вопрос о рынке, портфеле или хочет совершить действие:
1. Определи нужный API endpoint
2. Укажи запрос в формате: API_REQUEST: METHOD /path
3. После получения данных - проанализируй их и дай понятный ответ

Доступные endpoints:
- GET /v1/instruments/{symbol}/quotes/latest - котировка
- GET /v1/instruments/{symbol}/orderbook - стакан
- GET /v1/instruments/{symbol}/bars - свечи
- GET /v1/accounts/{account_id} - счет и позиции
- GET /v1/accounts/{account_id}/orders - ордера
- POST /v1/accounts/{account_id}/orders - создать ордер
- DELETE /v1/accounts/{account_id}/orders/{order_id} - отменить ордер

Отвечай на русском, кратко и по делу."""


def extract_api_request(text: str) -> tuple[str | None, str | None]:
    """Извлечь API запрос из ответа LLM"""
    if "API_REQUEST:" not in text:
        return None, None

    lines = text.split("\n")
    for line in lines:
        if line.strip().startswith("API_REQUEST:"):
            request = line.replace("API_REQUEST:", "").strip()
            parts = request.split(maxsplit=1)
            if len(parts) == 2:
                return parts[0], parts[1]
    return None, None


def main() -> None:  # noqa: C901
    """Главная функция Streamlit приложения"""
    st.set_page_config(page_title="AI Трейдер (Finam)", page_icon="🤖", layout="wide")

    # Заголовок
    st.title("🤖 AI Ассистент Трейдера")
    st.caption("Интеллектуальный помощник для работы с Finam TradeAPI")

    # Sidebar с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        settings = get_settings()
        st.info(f"**Модель:** {settings.openrouter_model}")

        # Finam API настройки
        with st.expander("🔑 Finam API", expanded=False):
            api_token = st.text_input(
                "Access Token",
                type="password",
                help="Токен доступа к Finam TradeAPI (или используйте FINAM_ACCESS_TOKEN)",
            )
            api_base_url = st.text_input("API Base URL", value="https://api.finam.ru", help="Базовый URL API")

        account_id = st.text_input("ID счета", value="", help="Оставьте пустым если не требуется")

        if st.button("🔄 Очистить историю"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 💡 Примеры вопросов:")
        st.markdown("""
        - Какая цена Сбербанка?
        - Покажи мой портфель
        - Что в стакане по Газпрому?
        - Покажи свечи YNDX за последние дни
        - Какие у меня активные ордера?
        - Детали моей сессии
        """)

    # Инициализация состояния
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Инициализация Finam API клиента
    finam_client = FinamAPIClient(access_token=api_token or None, base_url=api_base_url if api_base_url else None)

    # Проверка токена
    if not finam_client.access_token:
        st.sidebar.warning(
            "⚠️ Finam API токен не установлен. Установите в переменной окружения FINAM_ACCESS_TOKEN или введите выше."
        )
    else:
        st.sidebar.success("✅ Finam API токен установлен")

    # Отображение истории сообщений
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Показываем API запросы
            if "api_request" in message:
                with st.expander("🔍 API запрос"):
                    st.code(f"{message['api_request']['method']} {message['api_request']['path']}", language="http")
                    st.json(message["api_request"]["response"])

    # Поле ввода
    if prompt := st.chat_input("Напишите ваш вопрос..."):
        # Добавляем сообщение пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Формируем историю для LLM
        conversation_history = [{"role": "system", "content": create_system_prompt()}]
        for msg in st.session_state.messages:
            conversation_history.append({"role": msg["role"], "content": msg["content"]})

        # Получаем ответ от ассистента
        with st.chat_message("assistant"), st.spinner("Думаю..."):
            try:
                response = call_llm(conversation_history, temperature=0.3)
                assistant_message = response["choices"][0]["message"]["content"]

                # Проверяем API запрос
                method, path = extract_api_request(assistant_message)

                api_data = None
                if method and path:
                    # Подставляем account_id если есть
                    if account_id and "{account_id}" in path:  # noqa: RUF027
                        path = path.replace("{account_id}", account_id)

                    # Показываем что делаем запрос
                    st.info(f"🔍 Выполняю запрос: `{method} {path}`")

                    # Выполняем API запрос
                    api_response = finam_client.execute_request(method, path)

                    # Проверяем на ошибки
                    if "error" in api_response:
                        st.error(f"⚠️ Ошибка API: {api_response.get('error')}")
                        if "details" in api_response:
                            st.error(f"Детали: {api_response['details']}")

                    # Показываем результат
                    with st.expander("📡 Ответ API", expanded=False):
                        st.json(api_response)

                    api_data = {"method": method, "path": path, "response": api_response}

                    # Добавляем результат в контекст
                    conversation_history.append({"role": "assistant", "content": assistant_message})
                    conversation_history.append({
                        "role": "user",
                        "content": f"Результат API: {json.dumps(api_response, ensure_ascii=False)}\n\nПроанализируй.",
                    })

                    # Получаем финальный ответ
                    response = call_llm(conversation_history, temperature=0.3)
                    assistant_message = response["choices"][0]["message"]["content"]

                st.markdown(assistant_message)

                # Сохраняем сообщение ассистента
                message_data = {"role": "assistant", "content": assistant_message}
                if api_data:
                    message_data["api_request"] = api_data
                st.session_state.messages.append(message_data)

            except Exception as e:
                st.error(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
