/**
 * chatPage.js — Chat tab: onboarding guide, message stream, composer.
 *
 * No hardcoded prompts. The welcome card teaches users HOW to use the AI
 * through clear instructions, tips, and feature explanations.
 */

function renderChatPage() {
  const el = document.getElementById('main-content');
  el.innerHTML = `
    <div class="chat-stage">

      <!-- Onboarding guide (shown only when no messages exist) -->
      <section id="welcome-card" class="welcome-card">
        <div class="onboarding-hero">
          <span class="onboarding-icon">🩺</span>
          <h3>Welcome to Your AI Health Assistant</h3>
          <p class="onboarding-subtitle">
            I can help you understand your symptoms, assess your risk level,
            and guide you to the right care — in plain language.
          </p>
        </div>

        <!-- How to use — step by step -->
        <div class="how-to-grid">
          <div class="how-to-step">
            <span class="step-num">1</span>
            <div>
              <strong>Describe what you feel</strong>
              <p>Type your symptoms in the box below. Be as specific as you can —
                mention where it hurts, how long it has been happening, and how severe it is.</p>
            </div>
          </div>
          <div class="how-to-step">
            <span class="step-num">2</span>
            <div>
              <strong>Add context if you have it</strong>
              <p>You can mention your age, any existing conditions, or recent events
                that might be relevant. The more detail you give, the better the analysis.</p>
            </div>
          </div>
          <div class="how-to-step">
            <span class="step-num">3</span>
            <div>
              <strong>Attach an image (optional)</strong>
              <p>If you have a skin concern or visible symptom, tap the 📎 button
                to attach a photo. The AI will include it in the analysis.</p>
            </div>
          </div>
          <div class="how-to-step">
            <span class="step-num">4</span>
            <div>
              <strong>Review your results</strong>
              <p>You will receive a risk level, possible conditions, a recommended
                next step, and nearby facilities if needed.</p>
            </div>
          </div>
        </div>

        <!-- Tips panel -->
        <div class="tips-panel">
          <p class="tips-heading">💡 Tips for better results</p>
          <ul class="tips-list">
            <li>Mention <strong>duration</strong> — e.g. "for 3 days" or "since this morning"</li>
            <li>Mention <strong>location</strong> — e.g. "pain in my lower right abdomen"</li>
            <li>Mention <strong>severity</strong> — e.g. "mild", "moderate", or "severe"</li>
            <li>Mention <strong>associated symptoms</strong> — e.g. "also have nausea and no appetite"</li>
            <li>Use the <strong>Facilities tab</strong> to find a clinic or hospital near you</li>
            <li>Enable <strong>Allow search</strong> for the latest medical guidelines</li>
          </ul>
        </div>

        <!-- What the AI can and cannot do -->
        <div class="capability-row">
          <div class="capability-box can-do">
            <p class="cap-heading">✅ What I can do</p>
            <ul>
              <li>Analyze symptoms and suggest possible conditions</li>
              <li>Score your risk level (Low / Medium / High)</li>
              <li>Recommend whether to seek urgent care</li>
              <li>Find nearby hospitals, clinics, and pharmacies</li>
              <li>Provide prevention and self-care guidance</li>
            </ul>
          </div>
          <div class="capability-box cannot-do">
            <p class="cap-heading">⚠️ What I cannot do</p>
            <ul>
              <li>Replace a doctor's examination or diagnosis</li>
              <li>Prescribe medication or treatment</li>
              <li>Access your medical records</li>
              <li>Provide emergency services — call 911 / 907 if in danger</li>
            </ul>
          </div>
        </div>

        <p class="welcome-disclaimer">
          This system is for informational and decision-support purposes only.
          Always consult a qualified healthcare professional for medical advice.
        </p>
      </section>

      <!-- Message stream -->
      <div class="message-stream" id="message-stream" onscroll="checkScrollFab()"></div>

      <!-- Typing indicator -->
      <div id="typing-indicator" class="message-row assistant" style="display:none">
        <div class="msg-avatar">AI</div>
        <div class="bubble typing-bubble">
          <div class="typing-dots"><span></span><span></span><span></span></div>
          <p class="muted" style="font-size:0.82rem;margin:0">Analyzing your symptoms...</p>
        </div>
      </div>

      <!-- Scroll FAB -->
      <button id="scroll-fab" class="scroll-fab" style="display:none"
              onclick="scrollToBottom()">Latest ↓</button>

      <!-- Composer -->
      <div id="composer-container"></div>
    </div>`;

  document.getElementById('composer-container').innerHTML = renderComposer();
}

function checkScrollFab() {
  const stream = document.getElementById('message-stream');
  const fab    = document.getElementById('scroll-fab');
  if (!stream || !fab) return;
  const dist = stream.scrollHeight - stream.scrollTop - stream.clientHeight;
  fab.style.display = dist > 200 ? 'block' : 'none';
}
