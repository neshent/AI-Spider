/* ── State ───────────────────────────────────────────────────────────────── */
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('userInput');
const sendBtn    = document.getElementById('sendBtn');
const stopBtn    = document.getElementById('stopBtn');
const modelSel   = document.getElementById('modelSelect');

/* ── Model loading ───────────────────────────────────────────────────────── */
async function loadModels() {
  try {
    const res  = await fetch('/api/models');
    const data = await res.json();
    const models = data.models || [];

    modelSel.innerHTML = '';

    if (models.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No models available';
      modelSel.appendChild(opt);
      modelSel.disabled = true;
    } else {
      modelSel.disabled = false;
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.label;
        modelSel.appendChild(opt);
      }
    }
  } catch (e) {
    console.warn('Could not load models:', e);
  }
}

loadModels();

/* ── DOM helpers ─────────────────────────────────────────────────────────── */
function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

function hideWelcome() {
  const card = document.querySelector('.welcome-card');
  if (card) card.remove();
}

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = `message ${cls}`;
  d.textContent = text;
  messagesEl.appendChild(d);
  scrollBottom();
  return d;
}

function createStreamGroup() {
  const group  = document.createElement('div');
  group.className = 'msg-group';

  const bubble = document.createElement('div');
  bubble.className = 'response-bubble';
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  bubble.appendChild(cursor);

  group.appendChild(bubble);
  messagesEl.appendChild(group);
  scrollBottom();
  return { bubble, cursor };
}

/* ── Stop ────────────────────────────────────────────────────────────────── */
function stopGeneration() {
  stopBtn.classList.remove('visible');
  sendBtn.disabled = false;
  inputEl.focus();
}

/* ── Send message ────────────────────────────────────────────────────────── */
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || sendBtn.disabled) return;

  const model = modelSel.value;
  if (!model) {
    addMsg('No model selected. Please add a model first.', 'error');
    return;
  }

  inputEl.value = '';
  sendBtn.disabled = true;
  stopBtn.classList.add('visible');
  hideWelcome();

  addMsg(text, 'user');

  const { bubble, cursor } = createStreamGroup();

  try {
    const resp = await fetch('/api/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, model }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      cursor.remove();
      bubble.textContent = 'Error: ' + (err.error || resp.statusText);
      bubble.classList.add('error');
      finishStream();
      return;
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const messages = buf.split(/\r?\n\r?\n/);
      buf = messages.pop();

      for (const msg of messages) {
        let type = '';
        let dataLines = [];
        for (const line of msg.split(/\r?\n/)) {
          if (line.startsWith('event:'))     type = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5));
        }
        if (type) handleSSEEvent(type, dataLines.join('\n'));
      }
    }
  } catch (err) {
    cursor.remove();
    bubble.textContent = 'Stream error: ' + err.message;
  } finally {
    finishStream();
  }

  function handleSSEEvent(type, data) {
    if (type === 'content') {
      const isFirst = bubble.childNodes.length === 1;
      const txt = isFirst ? data.trimStart() : data;
      if (!txt) return;
      txt.length > 20
        ? typewriterAnimate(txt, bubble, cursor)
        : bubble.insertBefore(document.createTextNode(txt), cursor);
      scrollBottom();
    } else if (type === 'error') {
      cursor.remove();
      bubble.textContent = 'Error: ' + data;
      bubble.classList.add('error');
    }
  }

  function typewriterAnimate(text, bbl, csr) {
    let i = 0;
    const delay = Math.max(4, Math.min(18, 1200 / text.length));
    function tick() {
      if (i >= text.length) return;
      const batch = text.length > 400 ? 4 : 1;
      bbl.insertBefore(document.createTextNode(text.slice(i, i + batch)), csr);
      i += batch;
      scrollBottom();
      if (i < text.length) setTimeout(tick, delay);
    }
    tick();
  }

  function finishStream() {
    cursor.remove();
    stopBtn.classList.remove('visible');
    sendBtn.disabled = false;
    inputEl.focus();
    scrollBottom();
  }
}

/* ── Knowledge ───────────────────────────────────────────────────────────── */
async function addKnowledge() {
  const doc_id = document.getElementById('kDocId').value.trim();
  const text   = document.getElementById('kText').value.trim();
  if (!doc_id || !text) { alert('Both fields are required.'); return; }
  const res  = await fetch('/api/knowledge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ doc_id, text }),
  });
  const data = await res.json();
  if (data.status === 'ok') {
    document.getElementById('kDocId').value = '';
    document.getElementById('kText').value  = '';
    addMsg(`Knowledge added: "${doc_id}"`, 'assistant');
  } else {
    alert('Error: ' + (data.error || 'unknown'));
  }
}
