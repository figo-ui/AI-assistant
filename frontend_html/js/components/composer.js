/**
 * composer.js — Chat input composer with attachment, voice, and auto-resize.
 */

let _attachedFile = null;
let _attachedPreviewUrl = null;

function renderComposer() {
  return `
    <form class="composer" id="composer-form" onsubmit="App.submitAnalysis(event)">

      <!-- Attachment preview (only shown after user picks a file) -->
      <div id="attachment-preview" class="attachment-preview" style="display:none">
        <img id="att-img" src="" alt="" />
        <div class="att-info">
          <strong id="att-name"></strong>
          <small id="att-size"></small>
        </div>
        <button type="button" class="msg-action" onclick="removeAttachment()">✕ Remove</button>
      </div>

      <!-- Main input row: [📎] [textarea.....................] [▶] -->
      <div class="composer-row">

        <!-- Left: file picker icon -->
        <label class="composer-attach" title="Attach a photo of a skin condition (optional)">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
               fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66L9.41 17.41a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
          </svg>
          <input type="file" accept="image/png,image/jpeg,image/webp"
                 onchange="handleAttachment(this)" hidden />
        </label>

        <!-- Centre: growing textarea -->
        <textarea id="composer-input" rows="1"
          placeholder="Describe your symptoms — what, where, how long, how severe..."
          oninput="onComposerInput(this)"
          onkeydown="composerKeyDown(event)"></textarea>

        <!-- Right: send button -->
        <button type="submit" class="composer-send" id="send-btn"
                title="Analyze symptoms (Enter)">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
               fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>

      <!-- Writing hints — appear while typing -->
      <div id="composer-hints" class="composer-hints" style="display:none">
        <span class="hint-item">📍 Location?</span>
        <span class="hint-item">⏱ Duration?</span>
        <span class="hint-item">📊 Severity?</span>
        <span class="hint-item">➕ Other symptoms?</span>
      </div>

      <!-- Secondary options row -->
      <div class="composer-options">
        <label class="checkbox-line" title="I consent to this analysis being processed">
          <input type="checkbox" id="consent-check" checked /> I consent to analysis
        </label>
        <label class="checkbox-line"
               title="Allow the AI to search current medical guidelines for your query">
          <input type="checkbox" id="search-consent" /> Allow web search
        </label>
        <button type="button" class="composer-more-btn" id="more-btn"
                onclick="toggleComposerExtra()">⋯ More options</button>
      </div>

      <!-- Expandable extra options -->
      <div id="composer-extra" class="composer-extra" style="display:none">
        <label>
          Additional context tags <span class="muted" style="font-weight:400">(optional — improves accuracy)</span>
          <input id="tags-input" placeholder="e.g. diabetes, pregnancy, recent travel, allergies" />
        </label>
        <div class="extra-actions">
          <button type="button" class="ghost" onclick="App.switchTab('guidance')">📋 View Guidance</button>
          <button type="button" class="ghost" onclick="App.switchTab('facilities')">🏥 Find Facilities</button>
        </div>
      </div>

    </form>`;
}

function autoResize(el) {
  el.style.height = '0';
  el.style.height = Math.min(el.scrollHeight, 220) + 'px';
}

function onComposerInput(el) {
  autoResize(el);
  // Show writing hints once user starts typing, hide when empty
  const hints = document.getElementById('composer-hints');
  if (hints) hints.style.display = el.value.trim().length > 0 ? 'flex' : 'none';
}

function composerKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const btn = document.getElementById('send-btn');
    if (!btn.disabled) App.submitAnalysis(e);
  }
}

function toggleComposerExtra() {
  const el = document.getElementById('composer-extra');
  const btn = document.getElementById('more-btn');
  const open = el.style.display === 'none';
  el.style.display = open ? 'grid' : 'none';
  btn.textContent = open ? 'Less ▴' : 'More ▾';
}

function handleAttachment(input) {
  const file = input.files?.[0];
  if (!file) return;
  if (_attachedPreviewUrl) URL.revokeObjectURL(_attachedPreviewUrl);
  _attachedFile = file;
  _attachedPreviewUrl = URL.createObjectURL(file);
  document.getElementById('att-img').src = _attachedPreviewUrl;
  document.getElementById('att-name').textContent = file.name;
  document.getElementById('att-size').textContent = formatBytes(file.size);
  document.getElementById('attachment-preview').style.display = 'flex';
}

function removeAttachment() {
  if (_attachedPreviewUrl) URL.revokeObjectURL(_attachedPreviewUrl);
  _attachedFile = null;
  _attachedPreviewUrl = null;
  document.getElementById('attachment-preview').style.display = 'none';
  document.getElementById('att-img').src = '';
}

function handleVoiceInput() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { showToast('Voice input not supported in this browser.'); return; }
  const rec = new SR();
  rec.lang = 'en-US';
  rec.onresult = e => {
    const t = e.results?.[0]?.[0]?.transcript || '';
    const ta = document.getElementById('composer-input');
    if (ta) { ta.value = (ta.value ? ta.value + ' ' : '') + t; autoResize(ta); }
  };
  rec.start();
}

function setComposerBusy(busy) {
  const btn = document.getElementById('send-btn');
  const ta  = document.getElementById('composer-input');
  if (btn) {
    btn.disabled = busy;
    btn.innerHTML = busy
      ? `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
              fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
           <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="32">
             <animate attributeName="stroke-dashoffset" values="32;0;32" dur="1.2s" repeatCount="indefinite"/>
           </circle>
         </svg>`
      : `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24"
              fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
           <line x1="22" y1="2" x2="11" y2="13"/>
           <polygon points="22 2 15 22 11 13 2 9 22 2"/>
         </svg>`;
  }
  if (ta) ta.disabled = busy;
}

function clearComposer() {
  const ta = document.getElementById('composer-input');
  if (ta) { ta.value = ''; autoResize(ta); }
  const tags = document.getElementById('tags-input');
  if (tags) tags.value = '';
  removeAttachment();
}

function getComposerValues() {
  return {
    symptomText:      (document.getElementById('composer-input')?.value || '').trim(),
    tags:             csvToArray(document.getElementById('tags-input')?.value || ''),
    consentGiven:     document.getElementById('consent-check')?.checked ?? true,
    searchConsent:    document.getElementById('search-consent')?.checked ?? false,
    imageFile:        _attachedFile,
  };
}
