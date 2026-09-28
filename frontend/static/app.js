/* ==================================================
   AI KNOWLEDGE ASSISTANT - CLIENT
   ================================================== */

/* Served by FastAPI at http://localhost:8000.
   When the page is opened straight from disk (file://)
   the API still lives on the Uvicorn server. */
const API_BASE =
    window.location.protocol === "file:"
        ? "http://localhost:8000"
        : "";

const API = {
    ask: `${API_BASE}/ask`,
    stream: `${API_BASE}/ask/stream`,
    upload: `${API_BASE}/upload`,
    documents: `${API_BASE}/documents`,
    health: `${API_BASE}/health`,
};

const ALLOWED = [".pdf", ".docx", ".txt"];

const STORAGE_KEY = "aka.conversations.v1";

const THINKING_STAGES = [
    "Reading your question…",
    "Searching your documents…",
    "Pulling the most relevant passages…",
    "Writing the answer…",
];

const state = {
    documents: [],
    selected: new Set(),
    busy: false,
    uploading: false,
    loadingDocuments: true,
    abort: null,
    conversations: [],
    activeId: null,
    messages: [],
};

const el = {};

const $ = (id) => document.getElementById(id);

function cacheElements() {
    [
        "app", "sidebar", "sidebarOpen", "sidebarClose", "scrim", "newChat",
        "docCount", "docList", "docListEmpty", "dropzone", "fileInput",
        "uploadSlot", "uploadName", "uploadPct", "uploadBar", "uploadStatus",
        "selectAll", "selectNone", "status", "statusText",
        "messages", "welcome", "suggestions", "chatMeta", "scopeChip",
        "timingChip", "chatSubtitle", "composer", "input", "send",
        "attachButton", "scrollBottom", "toasts", "docTemplate",
        "historyList", "historyEmpty", "historyTemplate", "clearHistory",
    ].forEach((id) => { el[id] = $(id); });
}

/* ==================================================
   API
   ================================================== */

async function apiRequest(path, options = {}) {
    const response = await fetch(path, options);

    let payload = null;

    try {
        payload = await response.json();
    } catch {
        payload = null;
    }

    if (!response.ok) {
        const detail =
            (payload && (payload.detail || payload.message)) ||
            `Request failed (${response.status})`;

        throw new Error(
            typeof detail === "string" ? detail : JSON.stringify(detail),
        );
    }

    return payload;
}

const api = {
    listDocuments: () => apiRequest(API.documents),

    deleteDocument: (id) =>
        apiRequest(`${API.documents}/${id}`, { method: "DELETE" }),

    upload: (file, onProgress) =>
        new Promise((resolve, reject) => {
            const form = new FormData();
            form.append("file", file);

            const xhr = new XMLHttpRequest();
            xhr.open("POST", API.upload);

            xhr.upload.addEventListener("progress", (event) => {
                if (event.lengthComputable && onProgress) {
                    onProgress(
                        Math.min(
                            96,
                            Math.round(
                                (event.loaded / event.total) * 100,
                            ),
                        ),
                    );
                }
            });

            xhr.addEventListener("load", () => {
                let payload = null;

                try {
                    payload = JSON.parse(xhr.responseText);
                } catch {
                    payload = null;
                }

                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(payload);
                    return;
                }

                const detail =
                    (payload && (payload.detail || payload.message)) ||
                    `Upload failed (${xhr.status})`;

                reject(
                    new Error(
                        typeof detail === "string"
                            ? detail
                            : JSON.stringify(detail),
                    ),
                );
            });

            xhr.addEventListener("error", () =>
                reject(
                    new Error(
                        "Network error while uploading. Is the server running?",
                    ),
                ),
            );

            xhr.send(form);
        }),

    ask: (question, documentIds, history) =>
        apiRequest(API.ask, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                document_ids:
                    documentIds && documentIds.length
                        ? documentIds
                        : null,
                history: history && history.length
                    ? history
                    : null,
            }),
        }),

    /**
     * Server-sent events. Each event is handed to
     * onEvent as soon as it arrives, so the first words
     * appear without waiting for the full answer.
     */
    stream(question, documentIds, history, onEvent) {
        const controller = new AbortController();
        state.abort = controller;

        fetch(API.stream, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                document_ids:
                    documentIds && documentIds.length
                        ? documentIds
                        : null,
                history: history && history.length
                    ? history
                    : null,
            }),
            signal: controller.signal,
        })
            .then(async (response) => {
                if (!response.ok) {
                    let detail = `Server error (${response.status})`;

                    try {
                        const body = await response.json();
                        detail = body.detail || detail;
                    } catch {
                        /* keep the default */
                    }

                    onEvent({ type: "error", message: String(detail) });
                    return;
                }

                const reader = response.body
                    .getReader();

                const decoder = new TextDecoder();
                let buffer = "";

                for (;;) {
                    const { value, done } = await reader.read();

                    if (done) break;

                    buffer += decoder.decode(value, {
                        stream: true,
                    });

                    const frames = buffer.split("\n\n");
                    buffer = frames.pop() || "";

                    for (const frame of frames) {
                        const line = frame
                            .split("\n")
                            .find((item) => item.startsWith("data: "));

                        if (!line) continue;

                        const payload = line.slice(6).trim();

                        if (!payload || payload === "[DONE]") continue;

                        try {
                            onEvent(JSON.parse(payload));
                        } catch {
                            /* ignore malformed frame */
                        }
                    }
                }

                onEvent({ type: "closed" });
            })
            .catch((error) => {
                if (error.name === "AbortError") {
                    onEvent({ type: "aborted" });
                    return;
                }

                onEvent({
                    type: "error",
                    message: error.message,
                });
            });
    },

    stop: () => {
        if (state.abort) {
            state.abort.abort();
            state.abort = null;
        }
    },
};

/* ==================================================
   TOASTS
   ================================================== */

const TOAST_ICONS = {
    ok: '<path d="M20 6L9 17l-5-5"/>',
    error: '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
};

function toast(message, type = "info", ttl = 4200) {
    const node = document.createElement("div");
    node.className = `toast toast--${type}`;

    node.innerHTML = `
        <svg class="toast__icon" viewBox="0 0 24 24" aria-hidden="true">${TOAST_ICONS[type] || TOAST_ICONS.info}</svg>
        <span class="toast__text"></span>
    `;

    node.querySelector(".toast__text").textContent = message;
    el.toasts.appendChild(node);

    const remove = () => {
        node.classList.add("is-leaving");
        node.addEventListener(
            "animationend",
            () => node.remove(),
            { once: true },
        );
    };

    const timer = setTimeout(remove, ttl);
    node.addEventListener("click", () => {
        clearTimeout(timer);
        remove();
    });
}

/* ==================================================
   SAFE MARKDOWN-LITE RENDERER
   ================================================== */

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function inline(text) {
    return text
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(
            /(^|[\s(])\*([^*\n]+)\*(?=[\s.,!?):]|$)/g,
            "$1<em>$2</em>",
        );
}

function renderMarkdown(source) {
    const lines = escapeHtml(String(source || "")).split("\n");
    const out = [];
    let list = null;
    let code = null;

    const closeList = () => {
        if (list) {
            out.push(`</${list}>`);
            list = null;
        }
    };

    lines.forEach((line) => {
        const fence = line.match(/^```(.*)$/);

        if (fence) {
            if (code !== null) {
                out.push(`${code.join("\n")}</code></pre>`);
                code = null;
            } else {
                closeList();

                const language = fence[1]
                    .trim()
                    .toLowerCase()
                    .replace(/[^a-z0-9+#-]/g, "");

                out.push(
                    language
                        ? `<pre><code class="language-${language}">`
                        : "<pre><code>",
                );

                code = [];
            }

            return;
        }

        if (code !== null) {
            code.push(line);
            return;
        }

        const heading = line.match(/^(#{1,3})\s+(.*)$/);

        if (heading) {
            closeList();
            const level = heading[1].length;
            out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
            return;
        }

        if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
            closeList();
            out.push("<hr>");
            return;
        }

        const quote = line.match(/^&gt;\s?(.*)$/);

        if (quote) {
            closeList();
            out.push(`<blockquote>${inline(quote[1])}</blockquote>`);
            return;
        }

        const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
        const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);

        if (bullet || ordered) {
            const wanted = bullet ? "ul" : "ol";

            if (list !== wanted) {
                closeList();
                out.push(`<${wanted}>`);
                list = wanted;
            }

            out.push(
                `<li>${inline((bullet || ordered)[1])}</li>`,
            );
            return;
        }

        if (!line.trim()) {
            closeList();
            return;
        }

        closeList();
        out.push(`<p>${inline(line)}</p>`);
    });

    if (code !== null) {
        out.push(`${code.join("\n")}</code></pre>`);
    }

    closeList();

    return out.join("");
}

/* ==================================================
   STATUS
   ================================================== */

function setStatus(kind, text) {
    el.status.dataset.state = kind;
    el.statusText.textContent = text;
}

/* ==================================================
   DOCUMENTS
   ================================================== */

function extensionOf(name) {
    const match = String(name).match(/\.([a-z0-9]+)$/i);
    return match ? match[1].toLowerCase() : "file";
}

function formatBytes(value) {
    if (!value) return "";
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
    return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function renderDocuments() {
    el.docCount.textContent = state.documents.length;
    el.docList.innerHTML = "";

    const hasDocuments = state.documents.length > 0;

    el.docListEmpty.hidden = hasDocuments;

    // The sidebar starts with two shimmering placeholder
    // rows. They must only be visible while the first
    // fetch is in flight; once we know the real answer,
    // show either the documents or a real empty message.
    if (state.loadingDocuments) {
        el.docListEmpty.innerHTML = `
            <div class="skeleton-row"></div>
            <div class="skeleton-row"></div>
        `;
    } else {
        el.docListEmpty.innerHTML = `
            <svg class="doclist__emptyicon" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M14 3v5h5M6 3h8l5 5v13H6z"/>
            </svg>
            <p class="doclist__emptytitle">No documents yet</p>
            <p class="doclist__emptytext">
                Upload a PDF, DOCX or TXT file and your
                questions will be answered from it.
            </p>
        `;
    }

    state.documents.forEach((doc, index) => {
        const node = el.docTemplate.content
            .firstElementChild
            .cloneNode(true);

        const type = extensionOf(doc.filename);
        const checked = state.selected.has(doc.document_id);

        node.dataset.id = doc.document_id;
        node.style.animationDelay = `${Math.min(index * 45, 400)}ms`;
        node.classList.toggle("is-unselected", !checked);

        const toggle = node.querySelector(".doc__toggle");
        toggle.checked = checked;
        toggle.addEventListener("change", (event) => {
            event.stopPropagation();
            toggleDocument(doc.document_id, event.target.checked);
        });

        const icon = node.querySelector(".doc__icon");
        icon.dataset.type = type;
        icon.textContent = type.slice(0, 4).toUpperCase();

        node.querySelector(".doc__name").textContent = doc.filename;
        node.querySelector(".doc__meta").textContent = doc.chunks
            ? `${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}`
            : "not indexed";

        node.addEventListener("click", () => {
            toggleDocument(doc.document_id, !state.selected.has(doc.document_id));
        });

        node.querySelector(".doc__delete").addEventListener(
            "click",
            (event) => {
                event.stopPropagation();
                removeDocument(doc, node);
            },
        );

        el.docList.appendChild(node);
    });

    updateScopeUi();
}

function toggleDocument(id, enabled) {
    if (enabled) {
        state.selected.add(id);
    } else {
        state.selected.delete(id);
    }

    const node = el.docList.querySelector(`[data-id="${id}"]`);

    if (node) {
        node.classList.toggle("is-unselected", !enabled);
        const box = node.querySelector(".doc__toggle");
        if (box) box.checked = enabled;
    }

    updateScopeUi();
}

function updateScopeUi() {
    const count = state.selected.size;
    const total = state.documents.length;

    el.scopeChip.textContent =
        count === total && total > 0
            ? `all ${total} docs`
            : count === 1
              ? "1 doc"
              : `${count} doc${count === 1 ? "" : "s"}`;

    el.chatMeta.hidden = total === 0;

    el.chatSubtitle.textContent =
        total === 0
            ? "Upload a document to start"
            : count === 0
              ? "No documents selected — answers will be document-free"
              : `Answering from ${count} selected document${count === 1 ? "" : "s"}`;
}

async function loadDocuments({ silent = false } = {}) {
    state.loadingDocuments = true;
    renderDocuments();

    try {
        const data = await api.listDocuments();
        state.documents = data.documents || [];

        const valid = new Set(
            state.documents.map((doc) => doc.document_id),
        );

        state.selected = new Set(
            [...state.selected].filter((id) => valid.has(id)),
        );

        if (state.selected.size === 0) {
            state.documents.forEach((doc) =>
                state.selected.add(doc.document_id),
            );
        }

        state.loadingDocuments = false;
        renderDocuments();
        setStatus("ready", "Model ready · RAG online");

        if (!silent && state.documents.length) {
            toast(
                `${state.documents.length} document${state.documents.length === 1 ? "" : "s"} ready to query.`,
                "ok",
            );
        }
    } catch (error) {
        state.loadingDocuments = false;
        el.docListEmpty.hidden = true;
        setStatus("error", "Cannot reach the API");
        toast(error.message, "error", 6000);
    }
}

async function removeDocument(doc, node) {
    if (state.uploading) return;

    if (!window.confirm(`Remove "${doc.filename}" from the knowledge base?`)) {
        return;
    }

    node.classList.add("is-removing");
    setStatus("busy", "Removing document…");

    try {
        const result = await api.deleteDocument(doc.document_id);
        state.documents = result.documents || state.documents.filter(
            (item) => item.document_id !== doc.document_id,
        );
        state.selected.delete(doc.document_id);
        renderDocuments();
        setStatus("ready", "Model ready · RAG online");
        toast(`"${doc.filename}" removed.`, "ok");
    } catch (error) {
        node.classList.remove("is-removing");
        setStatus("ready", "Model ready · RAG online");
        toast(error.message, "error", 6000);
    }
}

/* ==================================================
   UPLOAD
   ================================================== */

function validateFile(file) {
    const name = file.name || "";
    const dot = name.lastIndexOf(".");

    if (dot < 0) {
        return "Files need an extension (.pdf, .docx or .txt).";
    }

    const extension = name.slice(dot).toLowerCase();

    if (!ALLOWED.includes(extension)) {
        return `"${extension}" is not supported. Use PDF, DOCX or TXT.`;
    }

    return null;
}

async function uploadFiles(fileList) {
    const files = [...fileList];

    if (!files.length) return;
    if (state.uploading) {
        toast("Still processing the previous upload.", "info");
        return;
    }

    const problems = files
        .map((file) => ({ file, problem: validateFile(file) }))
        .filter((item) => item.problem);

    problems.forEach((item) =>
        toast(item.problem, "error", 5200),
    );

    const valid = files.filter(
        (file) => !validateFile(file),
    );

    if (!valid.length) return;

    state.uploading = true;
    el.dropzone.classList.add("is-busy");
    el.attachButton.classList.add("is-spinning");
    setStatus("busy", "Indexing documents…");

    for (const file of valid) {
        showUploadProgress(file);

        try {
            const result = await api.upload(file, (percent) =>
                setUploadProgress(percent, "Chunking & embedding…"),
            );

            setUploadProgress(100, "Indexed successfully");
            toast(
                result.replaced
                    ? `${file.name} updated - re-indexed ${result.chunks} chunk${result.chunks === 1 ? "" : "s"}, previous version removed.`
                    : `${file.name} indexed - ${result.chunks} chunk${result.chunks === 1 ? "" : "s"} from ${result.characters} characters.`,
                "ok",
            );
        } catch (error) {
            setUploadProgress(0, `Failed: ${error.message}`);
            toast(`${file.name}: ${error.message}`, "error", 6500);
        }

        await sleep(700);
    }

    hideUploadProgress();
    el.dropzone.classList.remove("is-busy");
    el.attachButton.classList.remove("is-spinning");
    state.uploading = false;

    await loadDocuments({ silent: true });
}

function showUploadProgress(file) {
    el.uploadSlot.hidden = false;
    el.uploadName.textContent = file.name;
    el.uploadPct.textContent = "0%";
    el.uploadStatus.textContent = `Uploading ${formatBytes(file.size)}…`;

    el.uploadBar.classList.remove("progress__bar--done");
    el.uploadBar.classList.add("progress__bar--indeterminate");
    el.uploadBar.style.width = "";
}

function setUploadProgress(percent, status) {
    el.uploadPct.textContent = `${percent}%`;
    el.uploadStatus.textContent = status;

    if (percent >= 100) {
        el.uploadBar.classList.remove("progress__bar--indeterminate");
        el.uploadBar.classList.add("progress__bar--done");
    }
}

function hideUploadProgress() {
    el.uploadSlot.hidden = true;
}

/* ==================================================
   MESSAGES
   ================================================== */

function timeLabel(date = new Date()) {
    return date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
    });
}

function addUserMessage(text, at) {
    hideWelcome();

    const node = document.createElement("div");
    node.className = "msg msg--user";

    node.innerHTML = `
        <div class="avatar">You</div>
        <div class="msg__body">
            <div class="bubble"></div>
            <div class="msg__time">${timeLabel(at ? new Date(at) : new Date())}</div>
        </div>
    `;

    node.querySelector(".bubble").textContent = text;
    el.messages.appendChild(node);
    scrollToBottom(true);
}

function addBotShell() {
    hideWelcome();

    const node = document.createElement("div");
    node.className = "msg msg--bot";

    node.innerHTML = `
        <div class="avatar">AI</div>
        <div class="msg__body">
            <div class="bubble">
                <div class="thinking">
                    <span class="thinking__dots"><span></span><span></span><span></span></span>
                    <span class="thinking__label">${THINKING_STAGES[0]}</span>
                </div>
            </div>
        </div>
    `;

    el.messages.appendChild(node);
    scrollToBottom(true);

    return node;
}

function advanceThinking(labelEl) {
    let stage = 0;

    const timer = setInterval(() => {
        stage = (stage + 1) % THINKING_STAGES.length;

        const next = document.createElement("span");
        next.textContent = THINKING_STAGES[stage];
        labelEl.replaceChildren(next);
    }, 2600);

    return () => clearInterval(timer);
}

function addErrorMessage(message, at) {
    hideWelcome();

    const node = document.createElement("div");
    node.className = "msg msg--bot msg--error";

    node.innerHTML = `
        <div class="avatar">AI</div>
        <div class="msg__body">
            <div class="bubble"></div>
            <div class="actions">
                <button class="action" data-action="retry">
                    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 10-2.3 5.7M20 5v6h-6"/></svg>
                    Retry
                </button>
            </div>
            <div class="msg__time">${timeLabel(at ? new Date(at) : new Date())}</div>
        </div>
    `;

    node.querySelector(".bubble").textContent = message;
    el.messages.appendChild(node);
    scrollToBottom(true);
}

function hideWelcome() {
    el.welcome.hidden = true;
}

function buildSources(sources) {
    if (!sources || !sources.length) return "";

    const items = sources
        .map((source, index) => {
            const score = Number(source.score ?? 0);
            const low = score < 0.45;
            const snippet = String(source.text || "").trim();

            return `
                <div class="source" style="animation-delay:${index * 60}ms">
                    <div class="source__head">
                        <span class="source__name"></span>
                        <span class="score" data-low="${low}">${(score * 100).toFixed(0)}% match</span>
                    </div>
                    ${snippet ? `<p class="source__text"></p>` : ""}
                </div>
            `;
        })
        .join("");

    return `
        <div class="sources">
            <button class="sources__toggle" type="button">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>
                <span>${sources.length} source${sources.length === 1 ? "" : "s"} used</span>
            </button>
            <div class="sources__list">
                <div class="sources__inner">${items}</div>
            </div>
        </div>
    `;
}

/* ==================================================
   CONVERSATIONS
   ================================================== */

function newId() {
    return `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
}

function loadConversations() {
    let items = [];
    let activeId = null;

    try {
        const raw = window.localStorage.getItem(STORAGE_KEY);

        if (raw) {
            const parsed = JSON.parse(raw);

            if (Array.isArray(parsed)) {
                items = parsed;
            } else {
                items = parsed.items || [];
                activeId = parsed.activeId || null;
            }
        }
    } catch {
        items = [];
    }

    state.conversations = items
        .filter((item) => item && typeof item.id === "string")
        .slice(0, 40)
        .map((item) => ({
            id: item.id,
            title: item.title || "Untitled",
            created: item.created || Date.now(),
            messages: Array.isArray(item.messages) ? item.messages : [],
        }));

    const known = state.conversations.find(
        (item) => item.id === activeId,
    );

    if (known) {
        state.activeId = known.id;
        state.messages = known.messages;
    } else if (state.conversations.length) {
        state.activeId = state.conversations[0].id;
        state.messages = state.conversations[0].messages;
    } else {
        startConversation();
        return;
    }

    renderHistory();
    renderConversation();
}

function saveConversations() {
    try {
        window.localStorage.setItem(
            STORAGE_KEY,
            JSON.stringify({
                activeId: state.activeId,
                items: state.conversations.slice(0, 40),
            }),
        );
    } catch {
        /* storage full or blocked */
    }
}

function startConversation() {
    const conversation = {
        id: newId(),
        title: "New conversation",
        created: Date.now(),
        messages: [],
    };

    state.conversations.unshift(conversation);
    state.activeId = conversation.id;
    state.messages = conversation.messages;

    saveConversations();
    renderHistory();
    renderConversation();

    return conversation;
}

function activeConversation() {
    return state.conversations.find(
        (item) => item.id === state.activeId,
    );
}

function titleFrom(text) {
    const clean = text.replace(/\s+/g, " ").trim();

    return clean.length > 46
        ? `${clean.slice(0, 46)}…`
        : clean;
}

function pushMessage(role, payload) {
    const message = {
        role,
        ...payload,
        at: Date.now(),
    };

    state.messages.push(message);

    const conversation = activeConversation();

    if (conversation) {
        conversation.messages = state.messages;

        if (conversation.title === "New conversation") {
            conversation.title = titleFrom(
                payload.question || payload.text || "",
            );

            renderHistory();
        }
    }

    saveConversations();

    return message;
}

function switchConversation(id) {
    if (state.busy) {
        toast("Wait for the current answer to finish.", "info");
        return;
    }

    const conversation = state.conversations.find(
        (item) => item.id === id,
    );

    if (!conversation) return;

    state.activeId = id;
    state.messages = conversation.messages || [];

    saveConversations();
    renderHistory();
    renderConversation();
    closeSidebar();
}

function deleteConversation(id) {
    state.conversations = state.conversations.filter(
        (item) => item.id !== id,
    );

    if (!state.conversations.length) {
        startConversation();
        saveConversations();
        renderHistory();
        return;
    }

    if (state.activeId === id) {
        state.activeId = state.conversations[0].id;
        state.messages = state.conversations[0].messages || [];
        renderConversation();
    }

    saveConversations();
    renderHistory();
}

function clearHistory() {
    if (!window.confirm("Delete every saved conversation?")) {
        return;
    }

    state.conversations = [];
    startConversation();
    renderHistory();
    renderConversation();
    toast("Conversation history cleared.", "ok");
}

function relativeTime(stamp) {
    const seconds = Math.round(
        (Date.now() - stamp) / 1000,
    );

    if (seconds < 60) return "just now";
    if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;

    return new Date(stamp).toLocaleDateString();
}

function renderHistory() {
    el.historyList.innerHTML = "";

    el.historyEmpty.hidden = state.conversations.length > 0;

    state.conversations.forEach((conversation, index) => {
        const node = el.historyTemplate.content
            .firstElementChild
            .cloneNode(true);

        node.dataset.id = conversation.id;
        node.style.animationDelay = `${Math.min(index * 40, 300)}ms`;
        node.classList.toggle(
            "is-active",
            conversation.id === state.activeId,
        );

        node.querySelector(".history__title").textContent =
            conversation.title;

        const count = (conversation.messages || []).filter(
            (item) => item.role === "user",
        ).length;

        node.querySelector(".history__meta").textContent =
            `${count} question${count === 1 ? "" : "s"} · ${relativeTime(conversation.created)}`;

        node.querySelector(".history__open").addEventListener(
            "click",
            () => switchConversation(conversation.id),
        );

        node.querySelector(".history__delete").addEventListener(
            "click",
            () => deleteConversation(conversation.id),
        );

        el.historyList.appendChild(node);
    });
}

/* ==================================================
   RENDERING A WHOLE CONVERSATION
   ================================================== */

function renderConversation() {
    el.messages.innerHTML = "";
    el.timingChip.textContent = "";

    if (!state.messages.length) {
        el.welcome.hidden = false;
        return;
    }

    el.welcome.hidden = true;

    state.messages.forEach((message) => {
        if (message.role === "user") {
            addUserMessage(message.text, message.at);
            return;
        }

        if (message.role === "error") {
            addErrorMessage(message.text, message.at);
            return;
        }

        const shell = addBotShell();

        shell.classList.remove("is-streaming");

        const bubble = shell.querySelector(".bubble");
        bubble.innerHTML = renderMarkdown(message.text || "");

        attachSources(bubble, message.sources);
        attachActions(shell, message);
        attachPerf(shell, message.timing);

        if (message.at) {
            const time = document.createElement("div");
            time.className = "msg__time";
            time.textContent = timeLabel(new Date(message.at));
            shell.querySelector(".msg__body").appendChild(time);
        }
    });

    scrollToBottom(true);
}

function attachSources(bubble, sources) {
    if (!sources || !sources.length) return;

    const wrapper = document.createElement("div");
    wrapper.innerHTML = buildSources(sources);

    const node = wrapper.firstElementChild;
    bubble.appendChild(node);

    node.querySelectorAll(".source__name").forEach((item, index) => {
        item.textContent = sources[index]?.source || "Document";
    });

    node.querySelectorAll(".source__text").forEach((item, index) => {
        item.textContent = sources[index]?.text || "";
    });

    node.querySelector(".sources__toggle").addEventListener("click", () => {
        node.classList.toggle("is-open");
    });
}

function attachPerf(shell, timing) {
    if (!timing) return;

    const rows = [];

    if (timing.first_token_seconds != null) {
        rows.push(`first token <strong>${timing.first_token_seconds}s</strong>`);
    }

    rows.push(`total <strong>${timing.total_seconds}s</strong>`);

    if (timing.output_tokens) {
        rows.push(`<strong>${timing.output_tokens}</strong> tokens`);
    }

    if (timing.tokens_per_second) {
        rows.push(`<strong>${timing.tokens_per_second}</strong> tok/s`);
    }

    if (timing.search_seconds != null) {
        rows.push(`search <strong>${timing.search_seconds}s</strong>`);
    }

    const node = document.createElement("div");
    node.className = "perf";

    node.innerHTML = rows
        .join('<span class="perf__dot"></span>');

    shell.querySelector(".msg__body").appendChild(node);
}

/* ==================================================
   STREAMING CHAT FLOW
   ================================================== */

function sendMessage(rawText) {
    const text = (rawText ?? el.input.value).trim();

    if (!text || state.busy) return;

    if (state.documents.length === 0) {
        toast(
            "Upload a document first — answers are grounded in your own files.",
            "info",
            5000,
        );
        return;
    }

    if (state.selected.size === 0) {
        toast(
            "Select at least one document in the sidebar to ask a question.",
            "info",
            5000,
        );
        return;
    }

    setBusy(true);
    el.input.value = "";
    autoGrow();

    addUserMessage(text);

    const question = {
        role: "user",
        text,
        question: text,
    };

    pushMessage("user", question);

    const shell = addBotShell();
    shell.classList.add("is-streaming");

    const bubble = shell.querySelector(".bubble");
    const label = bubble.querySelector(".thinking__label");
    const stopStages = advanceThinking(label);

    const stream = document.createElement("div");
    const caret = document.createElement("span");
    caret.className = "stream-caret";

    let received = "";
    let finished = false;
    let sources = null;
    let timing = null;

    const turn = pushMessage("assistant", {
        text: "",
        sources: null,
        timing: null,
    });

    setStatus("busy", "Thinking…");

    const historyForModel = state.messages
        .slice(0, -2)
        .filter((item) => item.role === "user" || item.role === "assistant")
        .filter((item) => item.text)
        .slice(-6)
        .map((item) => ({ role: item.role, content: item.text }));

    const finalize = () => {
        if (finished) return;
        finished = true;

        stopStages();
        state.abort = null;

        shell.classList.remove("is-streaming");

        const text = received.trim();

        bubble.innerHTML = "";

        if (text) {
            bubble.appendChild(stream);
            stream.innerHTML = renderMarkdown(text);
        } else {
            bubble.innerHTML = `<p class="text-dim">No answer was produced.</p>`;
        }

        if (sources) attachSources(bubble, sources);
        if (timing) attachPerf(shell, timing);
        attachActions(shell, { answer: text });

        const time = document.createElement("div");
        time.className = "msg__time";
        time.textContent = timeLabel();
        shell.querySelector(".msg__body").appendChild(time);

        turn.text = text;
        turn.sources = sources;
        turn.timing = timing;
        saveConversations();

        if (timing) {
            el.timingChip.textContent = `${timing.total_seconds}s`;
        }

        setStatus("ready", "Model ready · RAG online");
        setBusy(false);
        scrollToBottom();
    };

    const showStream = (piece) => {
        if (!stream.parentNode) {
            stopStages();

            bubble.innerHTML = "";
            bubble.appendChild(stream);
            stream.appendChild(caret);
        }

        received += piece;
        stream.textContent = received;
        stream.appendChild(caret);

        scrollToBottom();
    };

    api.stream(
        text,
        [...state.selected],
        historyForModel,
        (event) => {
            if (event.type === "delta") {
                showStream(event.text || "");
                return;
            }

            if (event.type === "stage") {
                if (event.stage === "generate") {
                    const next = document.createElement("span");
                    next.textContent =
                        THINKING_STAGES[THINKING_STAGES.length - 1];
                    label.replaceChildren(next);
                }
                return;
            }

            if (event.type === "done") {
                received = event.answer || received;
                sources = event.sources;
                timing = event.timing;
                return;
            }

            if (event.type === "error") {
                stopStages();
                shell.remove();
                addErrorMessage(`Something went wrong: ${event.message}`);
                pushMessage("error", {
                    text: `Something went wrong: ${event.message}`,
                });
                setStatus("error", "Request failed");
                finished = true;
                setBusy(false);
                return;
            }

            if (event.type === "aborted") {
                finalize();
                toast("Generation stopped.", "info", 2400);
                return;
            }

            if (event.type === "closed") {
                finalize();
            }
        },
    );
}

function attachActions(shell, data) {
    const body = shell.querySelector(".msg__body");

    if (body.querySelector(".actions")) return;

    const actions = document.createElement("div");
    actions.className = "actions";

    actions.innerHTML = `
        <button class="action" data-action="copy">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 9h10v10H9zM5 5v12h2"/></svg>
            Copy
        </button>
        <button class="action" data-action="again">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 10-2.3 5.7M20 5v6h-6"/></svg>
            Ask again
        </button>
    `;

    actions
        .querySelector('[data-action="copy"]')
        .addEventListener("click", async (event) => {
            const button = event.currentTarget;
            const label = button.lastChild;

            try {
                await navigator.clipboard.writeText(
                    data.answer || data.text || "",
                );

                label.textContent = "Copied";
                toast("Answer copied.", "ok", 2000);

                setTimeout(() => {
                    label.textContent = " Copy";
                }, 1600);
            } catch {
                toast("Clipboard blocked by the browser.", "error");
            }
        });

    actions
        .querySelector('[data-action="again"]')
        .addEventListener("click", () => {
            const last = [...state.messages]
                .reverse()
                .find((item) => item.role === "user");

            if (last) sendMessage(last.question || last.text);
        });

    body.appendChild(actions);
}

function setBusy(busy) {
    state.busy = busy;
    el.input.disabled = busy;
    el.composer.classList.toggle("is-busy", busy);
    el.send.classList.toggle("is-stop", busy);
    el.send.disabled = busy ? false : !el.input.value.trim();
    el.input.placeholder = busy
        ? "Assistant is answering…"
        : "Ask a question about your documents…";
}

/* ==================================================
   COMPOSER
   ================================================== */

function autoGrow() {
    el.input.style.height = "auto";
    el.input.style.height = `${Math.min(el.input.scrollHeight, 180)}px`;
}

/* ==================================================
   SCROLLING
   ================================================== */

function scrollToBottom(instant = false) {
    if (typeof el.messages.scrollTo === "function") {
        el.messages.scrollTo({
            top: el.messages.scrollHeight,
            behavior: instant ? "auto" : "smooth",
        });
        return;
    }

    el.messages.scrollTop = el.messages.scrollHeight;
}

function isNearBottom() {
    const gap =
        el.messages.scrollHeight -
        el.messages.scrollTop -
        el.messages.clientHeight;

    return gap < 120;
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

/* ==================================================
   EVENTS
   ================================================== */

function openSidebar() {
    el.sidebar.classList.add("is-open");
    el.scrim.hidden = false;
}

function closeSidebar() {
    el.sidebar.classList.remove("is-open");
    el.scrim.hidden = true;
}

function bindEvents() {
    el.input.addEventListener("input", () => {
        autoGrow();
        el.send.disabled = state.busy || !el.input.value.trim();
    });

    el.input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    el.send.addEventListener("click", () => {
        if (state.busy) {
            api.stop();
            return;
        }

        sendMessage();
    });

    el.messages.addEventListener("click", (event) => {
        const retry = event.target.closest('[data-action="retry"]');

        if (retry) {
            const failed = retry.closest(".msg");
            const last = [...state.messages]
                .reverse()
                .find((item) => item.role === "user");

            if (failed) failed.remove();
            if (last) sendMessage(last.question || last.text);
        }
    });

    el.messages.addEventListener("scroll", () => {
        el.scrollBottom.hidden = isNearBottom();
    });

    el.scrollBottom.addEventListener("click", () => scrollToBottom());

    el.suggestions.addEventListener("click", (event) => {
        const button = event.target.closest(".suggestion");
        if (button) sendMessage(button.dataset.prompt);
    });

    el.newChat.addEventListener("click", () => {
        if (state.busy) {
            toast("Wait for the current answer to finish.", "info");
            return;
        }

        startConversation();
        el.input.focus();
        toast("New conversation started.", "info", 2200);
    });

    el.clearHistory.addEventListener("click", clearHistory);

    el.selectAll.addEventListener("click", () => {
        state.documents.forEach((doc) => {
            state.selected.add(doc.document_id);
        });
        renderDocuments();
    });

    el.selectNone.addEventListener("click", () => {
        state.selected.clear();
        renderDocuments();
    });

    el.dropzone.addEventListener("click", () => el.fileInput.click());
    el.dropzone.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            el.fileInput.click();
        }
    });

    el.attachButton.addEventListener("click", () => el.fileInput.click());

    el.fileInput.addEventListener("change", (event) => {
        uploadFiles(event.target.files);
        event.target.value = "";
    });

    ["dragenter", "dragover"].forEach((type) => {
        el.dropzone.addEventListener(type, (event) => {
            event.preventDefault();
            el.dropzone.classList.add("is-dragging");
        });
    });

    ["dragleave", "dragend"].forEach((type) => {
        el.dropzone.addEventListener(type, () =>
            el.dropzone.classList.remove("is-dragging"),
        );
    });

    el.dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        el.dropzone.classList.remove("is-dragging");
        uploadFiles(event.dataTransfer.files);
    });

    document.addEventListener("dragover", (event) => event.preventDefault());
    document.addEventListener("drop", (event) => event.preventDefault());

    el.sidebarOpen.addEventListener("click", openSidebar);
    el.sidebarClose.addEventListener("click", closeSidebar);
    el.scrim.addEventListener("click", closeSidebar);

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeSidebar();
    });
}

/* ==================================================
   BOOT
   ================================================== */

async function init() {
    cacheElements();
    bindEvents();
    autoGrow();
    loadConversations();

    setStatus("connecting", "Connecting to model…");
    await loadDocuments();
    el.input.focus();
}

document.addEventListener("DOMContentLoaded", init);
