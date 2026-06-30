const messages = document.querySelector("#messages");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const button = form.querySelector("button");
const statusNode = document.querySelector("#status");

const sessionIdKey = "sommelier_session_id";
const DEFAULT_GREETING = `Привет! Круто, что зашёл в наш бар. Меня зовут Бакард-ИИ. Я профессиональный бармен и так давно работаю в этом баре, что помню времена, когда коктейль «Апероль Шприц» считался экзотикой.

Чем я могу быть полезен:
1. Могу детально рассказать про каждый напиток: какие у него вкусовые особенности, с какой едой его лучше сочетать и в какие коктейли добавлять.
2. Могу подготовить персональную подборку напитков в зависимости от твоих предпочтений и критериев поиска.
3. Могу подобрать и составить список вкуснейших коктейлей в зависимости от повода, твоего настроения и вкусовых предпочтений.
4. Могу рассказать тебе истории из нашего бара.

Давай начнём наше знакомство. Как тебя зовут? Что хотелось бы сегодня?`;

function createSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

let sessionId = localStorage.getItem(sessionIdKey);
if (!sessionId) {
  sessionId = createSessionId();
  localStorage.setItem(sessionIdKey, sessionId);
}

function stripUnsupportedMarkdown(text) {
  return text
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1");
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderAssistantText(text) {
  const cleanText = stripUnsupportedMarkdown(text);
  return escapeHtml(cleanText)
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br>");
}

function appendMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  if (role.includes("assistant")) {
    node.innerHTML = renderAssistantText(text);
  } else {
    node.textContent = text;
  }
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

function setFormEnabled(enabled) {
  input.disabled = !enabled;
  button.disabled = !enabled;
}

async function loadChatHistory() {
  setFormEnabled(false);
  messages.replaceChildren();
  try {
    const response = await fetch(
      `/api/sessions/${encodeURIComponent(sessionId)}/messages`
    );
    if (!response.ok) {
      throw new Error(`History request failed: ${response.status}`);
    }
    const payload = await response.json();
    if (payload.messages.length === 0) {
      appendMessage("assistant", DEFAULT_GREETING);
      return;
    }
    for (const message of payload.messages) {
      appendMessage(message.role, message.content);
    }
  } catch (error) {
    appendMessage("assistant", DEFAULT_GREETING);
    console.warn("Chat history is unavailable", error);
  } finally {
    setFormEnabled(true);
    input.focus();
  }
}

async function loadCatalogStatus() {
  try {
    const response = await fetch("/api/catalog/status");
    if (!response.ok) {
      throw new Error(`Catalog status failed: ${response.status}`);
    }
    const status = await response.json();
    statusNode.textContent = `${status.product_profiles_count} ромов · ${status.cocktail_profiles_count} коктейлей`;
  } catch (error) {
    statusNode.textContent = "Каталог недоступен";
  }
}

loadCatalogStatus();
loadChatHistory();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  appendMessage("user", text);
  input.value = "";
  input.disabled = true;
  button.disabled = true;
  const loading = appendMessage("assistant loading", "Думаю...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }
    loading.className = "message assistant";
    loading.innerHTML = renderAssistantText(payload.answer);
  } catch (error) {
    loading.className = "message assistant";
    loading.textContent = "Сейчас не могу получить ответ. Проверь сервер и ключи API.";
  } finally {
    input.disabled = false;
    button.disabled = false;
    input.focus();
  }
});
