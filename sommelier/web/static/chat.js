const messages = document.querySelector("#messages");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const button = form.querySelector("button");
const statusNode = document.querySelector("#status");

const sessionIdKey = "sommelier_session_id";

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

appendMessage(
  "assistant",
  "Спроси меня о роме, сочетании с едой или коктейле. Я использую локальный каталог Bacardi и помню эту сессию."
);
loadCatalogStatus();

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
