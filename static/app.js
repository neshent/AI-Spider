/* ── State ───────────────────────────────────────────────────────────────── */
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('userInput');
const sendBtn    = document.getElementById('sendBtn');
const stopBtn    = document.getElementById('stopBtn');
const modelSel   = document.getElementById('modelSelect');
const keyBadge   = document.getElementById('keyBadge');
const hfWrap     = document.getElementById('hfTokenWrap');
const hfInput    = document.getElementById('hfTokenInput');
const lmsWrap    = document.getElementById('lmsPanelWrap');
const lmsUrlEl   = document.getElementById('lmsUrl');
const lmsModelEl = document.getElementById('lmsModel');
const lmsStatus  = document.getElementById('lmsStatus');
const lmsHint    = document.getElementById('lmsModelHint');

let _keys    = {};
let _lmsData = { running: false, models: [] };

/* ── Model loading ───────────────────────────────────────────────────────── */
async function loadModels() {
  try {
    const res  = await fetch('/api/models');
    const data = await res.json();
    _keys    = data.keys;
    _lmsData = data.lmstudio || { running: false, models: [] };

    const providerLabels = {
      mock:       'Offline',
      lmstudio:   'LM Studio (local server)',
      'hf-local': 'HuggingFace Local (no account)',
      'hf-api':   'HuggingFace API (free token)',
      anthropic:  'Anthropic',
      openai:     'OpenAI',
      google:     'Google',
    };
    const groups = {};
    for (const m of data.models) {
      (groups[m.provider] = groups[m.provider] || []).push(m);
    }
    modelSel.innerHTML = '';
    for (const [provider, models] of Object.entries(groups)) {
      // Use a disabled option as a plain text section header (no optgroup = no colored squares)
      const header = document.createElement('option');
      header.disabled = true;
      header.textContent = '-- ' + (providerLabels[provider] || provider) + ' --';
      header.style.cssText = 'color:#64748b;font-size:10px;font-weight:700;';
      modelSel.appendChild(header);
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.id; opt.textContent = m.label;
        modelSel.appendChild(opt);
      }
    }
    modelSel.value = 'mock';
    modelSel.addEventListener('change', updateUI);
    updateUI();
  } catch (e) { console.warn('Could not load models:', e); }
}

function providerOf(id) {
  if (!id || id === 'mock') return 'mock';
  if (id === 'lmstudio' || id.startsWith('lmstudio/')) return 'lmstudio';
  if (id.startsWith('hf-local/'))  return 'hf-local';
  if (id.startsWith('hf-api/'))    return 'hf-api';
  if (id.startsWith('claude'))     return 'anthropic';
  if (id.startsWith('gpt') || id.startsWith('o1') || id.startsWith('o3')) return 'openai';
  if (id.startsWith('gemini'))     return 'google';
  return 'unknown';
}

function updateUI() {
  const p = providerOf(modelSel.value);
  hfWrap.style.display  = p === 'hf-api'   ? 'flex'  : 'none';
  lmsWrap.style.display = p === 'lmstudio' ? 'block' : 'none';

  if      (p === 'mock')      setBadge('offline', 'local');
  else if (p === 'lmstudio')  setBadge(_lmsData.running ? 'server running' : 'server offline', _lmsData.running ? 'set' : 'unset');
  else if (p === 'hf-local')  setBadge('local model', 'local');
  else if (p === 'hf-api')    setBadge((_keys['hf-api'] || hfInput.value.trim()) ? 'token ready' : 'token needed', (_keys['hf-api'] || hfInput.value.trim()) ? 'set' : 'unset');
  else                        setBadge(_keys[p] ? 'key set' : 'no key', _keys[p] ? 'set' : 'unset');

  if      (p === 'lmstudio' && !_lmsData.running) setWarn('LM Studio server not detected. Open LM Studio - Local Server tab - Start Server.');
  else if (p === 'hf-api')                         setWarn('Requires internet connection to HuggingFace servers.');
  else if (p === 'anthropic' && !_keys.anthropic)  setWarn('Set ANTHROPIC_API_KEY in .env to use this model.');
  else if (p === 'openai'    && !_keys.openai)     setWarn('Set OPENAI_API_KEY in .env to use this model.');
  else if (p === 'google'    && !_keys.google)     setWarn('Set GOOGLE_API_KEY in .env to use this model.');
  else                                              setWarn(null);

  if (p === 'lmstudio') updateLMSPanel(_lmsData);
}

function setBadge(t, c) { keyBadge.textContent = t; keyBadge.className = `key-badge ${c}`; }
function setWarn(msg) {
  const w = document.getElementById('modelWarning');
  if (msg) { w.textContent = msg; w.style.display = 'block'; }
  else      { w.style.display = 'none'; }
}
hfInput.addEventListener('input', updateUI);
loadModels();

/* ── LM Studio probe ─────────────────────────────────────────────────────── */
function updateLMSPanel(data) {
  if (data.running) {
    lmsStatus.textContent = '● Connected'; lmsStatus.className = 'lms-status online';
    lmsHint.textContent   = data.models.length ? `Loaded: ${data.models.join(', ')}` : '';
    if (!lmsModelEl.value && data.models.length) lmsModelEl.placeholder = data.models[0];
  } else {
    lmsStatus.textContent = '● Offline'; lmsStatus.className = 'lms-status offline';
    lmsHint.textContent   = 'Start LM Studio → Local Server tab → Start Server';
  }
}

async function probeLMStudio() {
  lmsStatus.textContent = '…checking'; lmsStatus.className = 'lms-status';
  lmsHint.textContent   = '';
  try {
    const res  = await fetch('/api/lmstudio/probe', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url: lmsUrlEl.value.trim() || 'http://localhost:1234' }),
    });
    const data = await res.json();
    _lmsData = data; updateLMSPanel(data);
    setBadge(data.running ? 'server running' : 'server offline', data.running ? 'set' : 'unset');
    setWarn(data.running ? null : 'LM Studio server not detected. Open LM Studio - Local Server tab - Start Server.');
  } catch (err) {
    lmsStatus.textContent = '● Error'; lmsStatus.className = 'lms-status offline';
    lmsHint.textContent = err.message;
  }
}

/* ── DOM helpers ─────────────────────────────────────────────────────────── */
function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

function fillInput(text) {
  inputEl.value = text;
  inputEl.focus();
}

function hideWelcome() {
  const card = document.querySelector('.welcome-card');
  if (card) card.remove();
}

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = `message ${cls}`; d.textContent = text;
  messagesEl.appendChild(d); scrollBottom(); return d;
}

function createStreamGroup() {
  const group = document.createElement('div');
  group.className = 'msg-group';

  const thinkBlock = document.createElement('div');
  thinkBlock.className = 'thinking-block';
  thinkBlock.innerHTML = `
    <div class="thinking-header" onclick="toggleThinking(this)">
      <span class="thinking-dot pulsing"></span>
      <span class="thinking-label">Thinking</span>
      <span class="thinking-token-count">0 tokens</span>
      <span class="thinking-chevron">▼</span>
    </div>
    <div class="thinking-body"></div>`;
  thinkBlock.style.display = 'none';

  const bubble = document.createElement('div');
  bubble.className = 'response-bubble';
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  bubble.appendChild(cursor);

  group.appendChild(thinkBlock);
  group.appendChild(bubble);
  messagesEl.appendChild(group);
  scrollBottom();
  return { group, thinkBlock, bubble, cursor };
}

function toggleThinking(header) {
  header.closest('.thinking-block').classList.toggle('collapsed');
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
  inputEl.value = '';
  sendBtn.disabled = true;
  stopBtn.classList.add('visible');
  hideWelcome();

  const model    = modelSel.value;
  const provider = providerOf(model);
  addMsg(text, 'user');

  const { thinkBlock, bubble, cursor } = createStreamGroup();
  let thinkTokens  = 0;
  let thinkingDone = false;
  const thinkBody  = thinkBlock.querySelector('.thinking-body');
  const thinkDot   = thinkBlock.querySelector('.thinking-dot');
  const thinkCount = thinkBlock.querySelector('.thinking-token-count');

  const body = { message: text, model };
  if (provider === 'hf-api')   body.hf_token       = hfInput.value.trim();
  if (provider === 'lmstudio') {
    body.lmstudio_url   = lmsUrlEl.value.trim()   || 'http://localhost:1234';
    body.lmstudio_model = lmsModelEl.value.trim() || '';
  }

  try {
    const resp = await fetch('/api/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      cursor.remove();
      bubble.textContent = 'Error: ' + (err.error || resp.statusText);
      finishStream(); return;
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
        let type = ''; let dataLines = [];
        for (const line of msg.split(/\r?\n/)) {
          if (line.startsWith('event:'))      type = line.slice(6).trim();
          else if (line.startsWith('data:'))  dataLines.push(line.slice(5));
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
    if (type === 'thinking') {
      if (thinkBlock.style.display === 'none') thinkBlock.style.display = '';
      thinkBody.textContent += data;
      thinkTokens += data.length;
      thinkCount.textContent = `${thinkTokens} chars`;
      thinkBody.scrollTop = thinkBody.scrollHeight;
      scrollBottom();

    } else if (type === 'content') {
      if (!thinkingDone && thinkBlock.style.display !== 'none') {
        thinkingDone = true;
        thinkDot.classList.remove('pulsing');
        thinkBlock.classList.add('collapsed');
        thinkCount.textContent = `${thinkTokens} chars · done`;
      }
      const isFirst = bubble.childNodes.length === 1;
      const txt = isFirst ? data.trimStart() : data;
      if (!txt) return;
      txt.length > 20
        ? typewriterAnimate(txt, bubble, cursor)
        : bubble.insertBefore(document.createTextNode(txt), cursor);
      scrollBottom();

    } else if (type === 'pipeline') {
      // pipeline trace removed
    } else if (type === 'error') {
      cursor.remove();
      bubble.textContent = 'Error: ' + data;
      bubble.style.cssText = 'background:#3b1f1f;color:#fca5a5;border:1px solid #7f1d1d';
    }
  }

  function typewriterAnimate(text, bbl, csr) {
    let i = 0;
    const delay = Math.max(4, Math.min(18, 1200 / text.length));
    function tick() {
      if (i >= text.length) return;
      const batch = text.length > 400 ? 4 : 1;
      bbl.insertBefore(document.createTextNode(text.slice(i, i + batch)), csr);
      i += batch; scrollBottom();
      if (i < text.length) setTimeout(tick, delay);
    }
    tick();
  }

  function finishStream() {
    cursor.remove();
    thinkDot.classList.remove('pulsing');
    if (thinkTokens > 0) thinkCount.textContent = `${thinkTokens} chars · done`;
    stopBtn.classList.remove('visible');
    sendBtn.disabled = false;
    inputEl.focus(); scrollBottom();
  }
}

/* ── Knowledge ───────────────────────────────────────────────────────────── */
async function addKnowledge() {
  const doc_id = document.getElementById('kDocId').value.trim();
  const text   = document.getElementById('kText').value.trim();
  if (!doc_id || !text) { alert('Both fields are required.'); return; }
  const res  = await fetch('/api/knowledge', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
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
