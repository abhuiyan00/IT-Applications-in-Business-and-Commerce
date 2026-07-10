// IT Apps Knowledge Test — quiz engine.
// Learning model: active recall + immediate feedback + spaced retrieval (retry incorrect).
// Progress is DERIVED from `answered` (no separate score counters that can desync).

const STORAGE_KEY = 'itAppsQuizState';
const STORAGE_VERSION = 7; // bump when quizData shape changes -> stale saves auto-reset

// ---- State -----------------------------------------------------------------
// answered is keyed by `${set.id}-${question.id}` => { selected, isCorrect }
// timeSpent is keyed the same way => milliseconds studied on that question.
// Total elapsed is DERIVED from timeSpent (no separate counter to desync).
let state = {
    version: STORAGE_VERSION,
    currentSetIndex: 0,
    currentQuestionIndex: 0, // position within `currentOrder`; === order.length => summary view
    answered: {},
    timeSpent: {},
    setPositions: {},      // set.id -> last position within that set's order (resume where you left off)
    shuffle: false,
    theme: 'auto',         // 'auto' (follow OS) | 'light' | 'dark'
    view: 'quiz',          // 'quiz' | 'learn' | 'search' | 'mistakes'
    learnSetIndex: 0,      // which set is shown in the Learn view ('all' = every topic)
    learnReveal: true,     // show answers in the Learn view (false = flashcard mode)
};

// Module-local view state (not persisted)
let currentOrder = [];     // array of indices into the current set's questions[]
const orderCache = {};     // setIndex -> order array (stable within a session)
let isRetry = false;       // true when drilling only the previously-wrong questions

// ---- DOM -------------------------------------------------------------------
const el = (id) => document.getElementById(id);
const setsNav = el('setsNav');
const questionCard = el('questionCard');
const setTag = el('setTag');
const qIndex = el('qIndex');
const questionText = el('questionText');
const optionsContainer = el('optionsContainer');
const reasoningPanel = el('reasoningPanel');
const reasoningText = el('reasoningText');
const navButtons = el('navButtons');
const prevBtn = el('prevBtn');
const nextBtn = el('nextBtn');
const progressText = el('progressText');
const progressFill = el('progressFill');
const scoreText = el('scoreText');
const overallAnswered = el('overallAnswered');
const overallAccuracy = el('overallAccuracy');
const shuffleToggle = el('shuffleToggle');
const themeBtn = el('themeBtn');
const resetBtn = el('resetBtn');
const resetSetBtn = el('resetSetBtn');
const summaryCard = el('summaryCard');
const summaryTitle = el('summaryTitle');
const summaryRing = el('summaryRing');
const summaryPercent = el('summaryPercent');
const summaryDetail = el('summaryDetail');
const retryWrongBtn = el('retryWrongBtn');
const reviewBtn = el('reviewBtn');
const nextSetBtn = el('nextSetBtn');
const timerBar = document.querySelector('.timer-bar');
const questionTimer = el('questionTimer');
const totalTimer = el('totalTimer');
const pauseBtn = el('pauseBtn');

// Views (Quiz / Learn / Search)
const quizTopBar = el('quizTopBar');
const views = { quiz: el('quizView'), learn: el('learnView'), search: el('searchView'), mistakes: el('mistakesView') };
const viewTabs = document.querySelectorAll('.view-tab');
const learnSetSelect = el('learnSetSelect');
const learnRevealToggle = el('learnRevealToggle');
const learnList = el('learnList');
const searchInput = el('searchInput');
const searchResults = el('searchResults');
const searchCount = el('searchCount');
const mistakesList = el('mistakesList');
const mistakesCount = el('mistakesCount');

const LETTERS = ['A', 'B', 'C', 'D'];

// ---- Helpers ---------------------------------------------------------------
function currentSet() { return quizData.sets[state.currentSetIndex]; }
function keyFor(set, q) { return `${set.id}-${q.id}`; }

function shuffled(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// Full natural/shuffled order for a set, cached so navigation stays stable.
function fullOrder(setIndex) {
    if (!orderCache[setIndex]) {
        const idxs = quizData.sets[setIndex].questions.map((_, i) => i);
        orderCache[setIndex] = state.shuffle ? shuffled(idxs) : idxs;
    }
    return orderCache[setIndex];
}

// Stats for one set (always over ALL its questions, independent of retry subset).
function setStats(setIndex) {
    const set = quizData.sets[setIndex];
    let answered = 0, correct = 0;
    set.questions.forEach(q => {
        const a = state.answered[keyFor(set, q)];
        if (a) { answered++; if (a.isCorrect) correct++; }
    });
    return { answered, correct, total: set.questions.length };
}

function overallStats() {
    let answered = 0, correct = 0, total = 0;
    quizData.sets.forEach((_, i) => {
        const s = setStats(i);
        answered += s.answered; correct += s.correct; total += s.total;
    });
    return { answered, correct, total };
}

// ---- Persistence -----------------------------------------------------------
function saveProgress() {
    // Remember where we are in the current set so switching sets and coming back
    // resumes there (not back at Q1). Skip during a retry drill (its order is a
    // subset, so the position index wouldn't map back to the full set).
    if (!isRetry && state.view === 'quiz' && currentOrder.length) {
        (state.setPositions || (state.setPositions = {}))[currentSet().id] = state.currentQuestionIndex;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function loadProgress() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return;
    try {
        const parsed = JSON.parse(saved);
        if (parsed.version !== STORAGE_VERSION) return; // stale schema -> ignore
        state = { ...state, ...parsed };
    } catch (e) {
        // corrupt storage -> start fresh
    }
}

function resetAll() {
    if (!confirm('Reset ALL progress across every set? This cannot be undone.')) return;
    localStorage.removeItem(STORAGE_KEY);
    // Preserve view/learn prefs and the shuffle setting; wipe only progress.
    state = {
        version: STORAGE_VERSION, currentSetIndex: 0, currentQuestionIndex: 0,
        answered: {}, timeSpent: {}, setPositions: {}, shuffle: state.shuffle,
        theme: state.theme, view: state.view, learnSetIndex: 0, learnReveal: state.learnReveal,
    };
    Object.keys(orderCache).forEach(k => delete orderCache[k]);
    loadSet(0);
    renderSidebar();
    if (state.view === 'mistakes') renderMistakes();
}

function resetSet() {
    const set = currentSet();
    if (!confirm(`Reset progress for "${set.title}"?`)) return;
    set.questions.forEach(q => {
        const k = keyFor(set, q);
        delete state.answered[k];
        delete state.timeSpent[k];
    });
    delete state.setPositions[set.id];
    delete orderCache[state.currentSetIndex];
    saveProgress();
    loadSet(state.currentSetIndex);
    renderSidebar();
}

// ---- Set loading -----------------------------------------------------------
function loadSet(setIndex, position) {
    state.currentSetIndex = setIndex;
    isRetry = false;
    currentOrder = fullOrder(setIndex);
    // No explicit position -> resume where the learner left off in this set.
    if (position === undefined) {
        position = (state.setPositions && state.setPositions[quizData.sets[setIndex].id]) || 0;
    }
    // clamp position into [0, order.length] (length === summary view)
    state.currentQuestionIndex = Math.min(Math.max(position, 0), currentOrder.length);
    saveProgress();
    render();
}

// ---- Rendering -------------------------------------------------------------
const GROUP_LABELS = { study: 'Study sets', exam: 'Testownik' };

function renderSidebar() {
    setsNav.innerHTML = '';
    let lastGroup = null;
    quizData.sets.forEach((set, index) => {
        const group = set.group || 'study';
        if (group !== lastGroup) {
            const label = document.createElement('div');
            label.className = 'set-group-label';
            label.textContent = GROUP_LABELS[group] || group;
            setsNav.appendChild(label);
            lastGroup = group;
        }
        const s = setStats(index);
        const btn = document.createElement('button');
        btn.className = `set-btn ${index === state.currentSetIndex ? 'active' : ''}`;
        const pct = Math.round((s.answered / s.total) * 100);
        btn.innerHTML = `
            <div class="set-btn-top">
                <span>${set.title}</span>
                <span class="set-score">${s.correct}/${s.total}</span>
            </div>
            <div class="mastery-bar"><div class="mastery-fill" style="width:${pct}%"></div></div>
        `;
        btn.addEventListener('click', () => { loadSet(index); if (state.view !== 'quiz') showView('quiz'); });
        setsNav.appendChild(btn);
    });

    const o = overallStats();
    overallAnswered.textContent = `${o.answered}/${o.total}`;
    overallAccuracy.textContent = o.answered ? `${Math.round((o.correct / o.answered) * 100)}%` : '—';
}

function render() {
    const atSummary = state.currentQuestionIndex >= currentOrder.length;
    questionCard.hidden = atSummary;
    navButtons.hidden = atSummary;
    reasoningPanel.style.display = atSummary ? 'none' : '';
    summaryCard.hidden = !atSummary;
    if (atSummary) renderSummary();
    else renderQuestion();
    timer.lastTs = Date.now(); // clean boundary so a partial tick isn't misattributed
    renderTimer();
}

function renderQuestion() {
    const set = currentSet();
    const question = set.questions[currentOrder[state.currentQuestionIndex]];
    const key = keyFor(set, question);
    const prev = state.answered[key];

    setTag.textContent = cleanTitle(set.title);
    qIndex.textContent = `Q${question.id}`;
    questionText.textContent = question.question;

    const pos = state.currentQuestionIndex + 1;
    progressText.textContent = `${isRetry ? 'Retry · ' : ''}Question ${pos} of ${currentOrder.length}`;
    progressFill.style.width = `${(pos / currentOrder.length) * 100}%`;

    optionsContainer.innerHTML = '';
    question.options.forEach((opt, i) => {
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.innerHTML = `<span class="option-letter">${LETTERS[i]}</span> <span>${opt}</span>`;
        if (prev) {
            btn.classList.add('disabled');
            if (i === question.correct) btn.classList.add('correct');
            if (i === prev.selected && !prev.isCorrect) btn.classList.add('incorrect');
        } else {
            btn.addEventListener('click', () => selectOption(i, question, key));
        }
        optionsContainer.appendChild(btn);
    });

    if (prev) {
        reasoningText.textContent = question.reasoning;
        reasoningPanel.classList.add('show');
    } else {
        reasoningPanel.classList.remove('show');
    }

    prevBtn.disabled = state.currentQuestionIndex === 0 && (state.currentSetIndex === 0 || isRetry);
    nextBtn.disabled = !prev;
    nextBtn.textContent = (state.currentQuestionIndex === currentOrder.length - 1) ? 'Finish ✓' : 'Next →';

    updateScore();
}

function selectOption(selectedIndex, question, key) {
    if (state.answered[key]) return; // already answered -> ignore further clicks
    const isCorrect = selectedIndex === question.correct;
    state.answered[key] = { selected: selectedIndex, isCorrect };
    saveProgress();

    optionsContainer.querySelectorAll('.option-btn').forEach((btn, i) => {
        btn.classList.add('disabled');
        if (i === question.correct) btn.classList.add('correct');
        if (i === selectedIndex && !isCorrect) btn.classList.add('incorrect');
    });

    reasoningText.textContent = question.reasoning;
    reasoningPanel.classList.add('show');
    nextBtn.disabled = false;
    renderSidebar();
    updateScore();
}

function renderSummary() {
    const set = currentSet();
    // Score over the questions currently in view (full set, or retry subset)
    let correct = 0;
    currentOrder.forEach(i => {
        const a = state.answered[keyFor(set, set.questions[i])];
        if (a && a.isCorrect) correct++;
    });
    const total = currentOrder.length;
    const pct = total ? Math.round((correct / total) * 100) : 0;

    summaryTitle.textContent = isRetry ? 'Retry complete' : `${set.title} — complete`;
    summaryPercent.textContent = `${pct}%`;
    summaryRing.style.background =
        `conic-gradient(var(--correct) 0% ${pct}%, var(--border) ${pct}% 100%)`;
    summaryDetail.textContent = `You got ${correct} of ${total} correct.`;

    const wrong = countWrong();
    retryWrongBtn.disabled = wrong === 0;
    retryWrongBtn.textContent = wrong ? `Retry incorrect (${wrong})` : 'All correct 🎉';
    nextSetBtn.style.display = state.currentSetIndex < quizData.sets.length - 1 ? '' : 'none';

    progressText.textContent = isRetry ? 'Retry complete' : 'Set complete';
    progressFill.style.width = '100%';
    updateScore();
}

// Wrong answers among the questions currently in view.
function countWrong() {
    const set = currentSet();
    return currentOrder.reduce((n, i) => {
        const a = state.answered[keyFor(set, set.questions[i])];
        return n + (a && !a.isCorrect ? 1 : 0);
    }, 0);
}

function updateScore() {
    const s = setStats(state.currentSetIndex);
    const acc = s.answered ? ` (${Math.round((s.correct / s.answered) * 100)}%)` : '';
    scoreText.textContent = `${s.correct} / ${s.answered}${acc}`;
}

// ---- Study timer -----------------------------------------------------------
// Count-up elapsed time (per question + derived total), with a pause control.
// Active time is only counted on a live question while not paused and the tab
// is visible — idle/background time isn't billed to the learner's study clock.
let timer = { paused: false, lastTs: Date.now(), saveCounter: 0 };

function fmt(ms) {
    const total = Math.floor(ms / 1000);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const p = n => String(n).padStart(2, '0');
    return h ? `${h}:${p(m)}:${p(s)}` : `${p(m)}:${p(s)}`;
}

function totalElapsed() {
    return Object.values(state.timeSpent).reduce((a, b) => a + b, 0);
}

// Key of the question on screen, or null at the summary view.
function currentTimerKey() {
    if (state.currentQuestionIndex >= currentOrder.length) return null;
    const set = currentSet();
    return keyFor(set, set.questions[currentOrder[state.currentQuestionIndex]]);
}

function renderTimer() {
    const key = currentTimerKey();
    questionTimer.textContent = key ? fmt(state.timeSpent[key] || 0) : '—';
    totalTimer.textContent = fmt(totalElapsed());
}

function timerTick() {
    const now = Date.now();
    const delta = now - timer.lastTs;
    timer.lastTs = now;
    // Bill time only while actively studying a question; guard stray large gaps.
    if (state.view === 'quiz' && !timer.paused && !document.hidden && delta > 0 && delta < 60000) {
        const key = currentTimerKey();
        if (key) {
            state.timeSpent[key] = (state.timeSpent[key] || 0) + delta;
            if (++timer.saveCounter >= 10) { timer.saveCounter = 0; saveProgress(); }
        }
    }
    renderTimer();
}

function setPaused(paused) {
    timer.paused = paused;
    timer.lastTs = Date.now(); // don't bill the paused gap on resume
    pauseBtn.setAttribute('aria-pressed', String(paused));
    pauseBtn.textContent = paused ? '▶ Resume' : '⏸ Pause';
    if (timerBar) timerBar.classList.toggle('is-paused', paused);
    saveProgress();
}

function togglePause() { setPaused(!timer.paused); }

// ---- Navigation actions ----------------------------------------------------
function nextAction() {
    const atSummary = state.currentQuestionIndex >= currentOrder.length;
    if (atSummary) { goNextSet(); return; }
    const set = currentSet();
    const answered = state.answered[keyFor(set, set.questions[currentOrder[state.currentQuestionIndex]])];
    if (!answered) return; // must answer before advancing
    if (state.currentQuestionIndex < currentOrder.length - 1) {
        state.currentQuestionIndex++;
    } else {
        state.currentQuestionIndex = currentOrder.length; // -> summary
    }
    saveProgress();
    render();
}

function prevAction() {
    if (state.currentQuestionIndex > 0) {
        state.currentQuestionIndex--;
    } else if (state.currentSetIndex > 0 && !isRetry) {
        const target = state.currentSetIndex - 1;
        loadSet(target, fullOrder(target).length - 1);
        renderSidebar();
        return;
    } else {
        return;
    }
    saveProgress();
    render();
}

function goNextSet() {
    if (state.currentSetIndex < quizData.sets.length - 1) {
        loadSet(state.currentSetIndex + 1);
        renderSidebar();
    } else {
        alert(`You have completed all ${quizData.sets.length} sets. Use "Retry incorrect" on any set to drill the ones you missed.`);
    }
}

function retryWrong() {
    const set = currentSet();
    const wrongIdx = currentOrder.filter(i => {
        const a = state.answered[keyFor(set, set.questions[i])];
        return a && !a.isCorrect;
    });
    if (!wrongIdx.length) return;
    // Clear their answers AND their accumulated time so a re-attempt starts from a
    // clean 00:00 clock (otherwise the per-question timer resumes from the prior try).
    wrongIdx.forEach(i => {
        const k = keyFor(set, set.questions[i]);
        delete state.answered[k];
        delete state.timeSpent[k];
    });
    isRetry = true;
    currentOrder = state.shuffle ? shuffled(wrongIdx) : wrongIdx;
    state.currentQuestionIndex = 0;
    saveProgress();
    renderSidebar();
    render();
}

function reviewAnswers() {
    // Go back through the questions in view from the start (answers stay shown).
    state.currentQuestionIndex = 0;
    saveProgress();
    render();
}

// ---- Views: Quiz / Learn / Search ------------------------------------------
function showView(name) {
    state.view = name;
    Object.entries(views).forEach(([k, node]) => { if (node) node.hidden = (k !== name); });
    viewTabs.forEach(t => {
        const on = t.dataset.view === name;
        t.classList.toggle('active', on);
        t.setAttribute('aria-selected', String(on));
        t.tabIndex = on ? 0 : -1; // roving tabindex: only the active tab is in tab order
    });
    if (quizTopBar) quizTopBar.style.display = name === 'quiz' ? '' : 'none';
    if (name === 'learn') renderLearn();
    if (name === 'search') { searchInput.focus(); renderSearch(); }
    if (name === 'mistakes') renderMistakes();
    timer.lastTs = Date.now(); // don't bill view-switch gap to the current question
    saveProgress();
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function cleanTitle(t) { return t.replace(/^\d+\s·\s/, ''); }

// Build one card's HTML (used by both Learn and Search). `hi` optionally highlights a term.
function cardHtml(set, q, hi) {
    const mark = (txt) => {
        const safe = escapeHtml(txt);
        if (!hi) return safe;
        return safe.replace(new RegExp('(' + hi.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi'),
            '<mark>$1</mark>');
    };
    const opts = q.options.map((opt, i) =>
        `<div class="learn-opt ${i === q.correct ? 'correct' : ''}">
            <span class="opt-letter">${LETTERS[i]}</span><span>${mark(opt)}</span>
        </div>`).join('');
    return `<div class="learn-card revealed">
        <div class="learn-card-meta">
            <span class="learn-card-tag">${escapeHtml(cleanTitle(set.title))}</span>
            <span class="learn-card-id">Q${q.id}</span>
        </div>
        <div class="learn-card-q">${mark(q.question)}</div>
        ${opts}
        <div class="flip-hint">Tap to reveal answer ▾</div>
        <div class="learn-explain">💡 ${mark(q.reasoning)}</div>
    </div>`;
}

function renderLearn() {
    // Populate the topic selector once: an "All topics" entry + every set, grouped.
    if (!learnSetSelect.options.length) {
        const all = document.createElement('option');
        all.value = 'all'; all.textContent = '★ All topics (every question)';
        learnSetSelect.appendChild(all);
        let lastGroup = null;
        let optgroup = null;
        quizData.sets.forEach((set, i) => {
            const g = set.group || 'study';
            if (g !== lastGroup) {
                optgroup = document.createElement('optgroup');
                optgroup.label = GROUP_LABELS[g] || g;
                learnSetSelect.appendChild(optgroup);
                lastGroup = g;
            }
            const o = document.createElement('option');
            o.value = i; o.textContent = set.title;
            optgroup.appendChild(o);
        });
    }
    // Resolve the selection -> a flat list of {set, q}. 'all' spans every topic.
    let cards;
    if (state.learnSetIndex === 'all') {
        learnSetSelect.value = 'all';
        cards = [];
        quizData.sets.forEach(set => set.questions.forEach(q => cards.push({ set, q })));
    } else {
        const idx = Math.min(state.learnSetIndex, quizData.sets.length - 1);
        learnSetSelect.value = idx;
        const set = quizData.sets[idx];
        cards = set.questions.map(q => ({ set, q }));
    }
    learnRevealToggle.checked = state.learnReveal;
    learnList.innerHTML = cards.map(({ set, q }) => cardHtml(set, q, null)).join('');
    applyLearnReveal();
    // Flashcard mode: when answers are hidden, each card can be flipped on its own
    // (click / Enter / Space) so you self-test before seeing the answer.
    learnList.querySelectorAll('.learn-card').forEach(c => {
        c.addEventListener('click', () => { if (!state.learnReveal) c.classList.toggle('revealed'); });
        c.addEventListener('keydown', (e) => {
            if (!state.learnReveal && (e.key === 'Enter' || e.key === ' ')) {
                e.preventDefault(); c.classList.toggle('revealed');
            }
        });
    });
}

// Apply the global reveal state to every Learn card and set its flashcard affordance
// (focusable + button role only while in flip mode).
function applyLearnReveal() {
    learnList.querySelectorAll('.learn-card').forEach(c => {
        c.classList.toggle('revealed', state.learnReveal);
        if (state.learnReveal) {
            if (c.removeAttribute) { c.removeAttribute('tabindex'); c.removeAttribute('role'); }
        } else {
            c.setAttribute('tabindex', '0'); c.setAttribute('role', 'button');
        }
    });
}

let searchTimer = null;
function renderSearch() {
    const term = searchInput.value.trim();
    if (term.length < 2) {
        searchResults.innerHTML = '<div class="search-empty">Type at least 2 characters to search across every question, answer and explanation.</div>';
        searchCount.textContent = '';
        return;
    }
    const low = term.toLowerCase();
    const hits = [];
    quizData.sets.forEach(set => {
        set.questions.forEach(q => {
            const hay = (q.question + ' ' + q.options.join(' ') + ' ' + q.reasoning + ' ' + set.title).toLowerCase();
            if (hay.includes(low)) hits.push({ set, q });
        });
    });
    const CAP = 100;
    const shown = Math.min(hits.length, CAP);
    searchCount.textContent = hits.length
        ? `${hits.length} match${hits.length === 1 ? '' : 'es'}${hits.length > CAP ? ` · showing first ${CAP}` : ''}`
        : '0 matches';
    searchResults.innerHTML = hits.length
        ? hits.slice(0, CAP).map(h => cardHtml(h.set, h.q, term)).join('')
            + (hits.length > CAP ? `<div class="search-empty">${hits.length - shown} more match${hits.length - shown === 1 ? '' : 'es'} hidden — refine your search to narrow it down.</div>` : '')
        : '<div class="search-empty">No questions match that keyword.</div>';
}

// ---- Mistakes review -------------------------------------------------------
// A running list of every question answered WRONG (this device's saved progress),
// so the learner can revisit, re-read the explanation, and jump back to fix it.
function collectMistakes() {
    const out = [];
    quizData.sets.forEach((set, setIndex) => {
        set.questions.forEach((q, qArrIndex) => {
            const a = state.answered[keyFor(set, q)];
            if (a && !a.isCorrect) out.push({ set, setIndex, q, qArrIndex, selected: a.selected });
        });
    });
    return out;
}

function mistakeCardHtml(m) {
    const { set, q, selected } = m;
    const opts = q.options.map((opt, i) => {
        const cls = i === q.correct ? 'correct' : (i === selected ? 'your-wrong' : '');
        const tag = i === q.correct ? ' <span class="opt-flag ok">✓ correct</span>'
            : (i === selected ? ' <span class="opt-flag bad">✗ your answer</span>' : '');
        return `<div class="learn-opt ${cls}">
            <span class="opt-letter">${LETTERS[i]}</span><span>${escapeHtml(opt)}${tag}</span>
        </div>`;
    }).join('');
    return `<div class="learn-card revealed mistake-card">
        <div class="learn-card-meta">
            <span class="learn-card-tag">${escapeHtml(cleanTitle(set.title))}</span>
            <span class="learn-card-id">Q${q.id}</span>
            <button class="mistake-jump" data-set="${m.setIndex}" data-qidx="${m.qArrIndex}">Review in quiz →</button>
        </div>
        <div class="learn-card-q">${escapeHtml(q.question)}</div>
        ${opts}
        <div class="learn-explain">💡 ${escapeHtml(q.reasoning)}</div>
    </div>`;
}

// Re-drill just one set's saved mistakes, straight from the Mistakes view.
// Clears their answers so they can be re-attempted, then runs a retry subset.
function retrySetMistakes(setIndex) {
    const set = quizData.sets[setIndex];
    const wrongIdx = set.questions
        .map((_, i) => i)
        .filter(i => {
            const a = state.answered[keyFor(set, set.questions[i])];
            return a && !a.isCorrect;
        });
    if (!wrongIdx.length) return;
    // Clear answers AND accumulated time so each re-attempt starts at 00:00.
    wrongIdx.forEach(i => {
        const k = keyFor(set, set.questions[i]);
        delete state.answered[k];
        delete state.timeSpent[k];
    });
    state.currentSetIndex = setIndex;
    isRetry = true;
    currentOrder = state.shuffle ? shuffled(wrongIdx) : wrongIdx;
    state.currentQuestionIndex = 0;
    renderSidebar();
    showView('quiz');
    render();
}

function renderMistakes() {
    const items = collectMistakes();
    mistakesCount.textContent = items.length
        ? `${items.length} to review` : '';
    if (!items.length) {
        mistakesList.innerHTML = '<div class="search-empty">No mistakes saved yet. Wrong answers you give in the Quiz collect here so you can revisit and fix them.</div>';
        return;
    }
    // Group mistakes by set (preserving first-seen order) so each section can be
    // retaken on its own.
    const groups = [];
    const byIndex = {};
    items.forEach(m => {
        if (byIndex[m.setIndex] === undefined) {
            byIndex[m.setIndex] = groups.length;
            groups.push({ setIndex: m.setIndex, set: m.set, items: [] });
        }
        groups[byIndex[m.setIndex]].items.push(m);
    });
    mistakesList.innerHTML = groups.map(g => `
        <div class="mistake-group">
            <div class="mistake-group-head">
                <span class="mistake-group-title">${escapeHtml(cleanTitle(g.set.title))}</span>
                <span class="mistake-group-count">${g.items.length} wrong</span>
                <button class="mistake-retake" data-set="${g.setIndex}">Retake these ${g.items.length} →</button>
            </div>
            ${g.items.map(mistakeCardHtml).join('')}
        </div>`).join('');
    mistakesList.querySelectorAll('.mistake-jump').forEach(btn => {
        btn.addEventListener('click', () => {
            const setIndex = +btn.dataset.set;
            const qArrIndex = +btn.dataset.qidx;
            const pos = fullOrder(setIndex).indexOf(qArrIndex);
            loadSet(setIndex, pos < 0 ? 0 : pos);
            renderSidebar();
            showView('quiz');
        });
    });
    mistakesList.querySelectorAll('.mistake-retake').forEach(btn => {
        btn.addEventListener('click', () => retrySetMistakes(+btn.dataset.set));
    });
}

// ---- Theme (light / dark / auto) -------------------------------------------
// 'auto' follows the OS via the CSS @media query; 'light'/'dark' force it through a
// [data-theme] attribute on <html> that the CSS overrides. Persisted in state.
const THEME_CYCLE = ['auto', 'light', 'dark'];
const THEME_LABEL = { auto: '🌗 Auto', light: '☀️ Light', dark: '🌙 Dark' };

function applyTheme() {
    const root = (typeof document !== 'undefined') && document.documentElement;
    if (root && root.setAttribute) {
        if (state.theme === 'auto') root.removeAttribute('data-theme');
        else root.setAttribute('data-theme', state.theme);
    }
    if (themeBtn) {
        themeBtn.textContent = THEME_LABEL[state.theme] || THEME_LABEL.auto;
        themeBtn.setAttribute('aria-label', `Theme: ${state.theme}. Click to change.`);
        themeBtn.setAttribute('title', `Theme: ${state.theme} — click to change`);
    }
}

function cycleTheme() {
    const i = THEME_CYCLE.indexOf(state.theme);
    state.theme = THEME_CYCLE[(i + 1) % THEME_CYCLE.length];
    applyTheme();
    saveProgress();
}

// ---- Init ------------------------------------------------------------------
function attachEventListeners() {
    if (themeBtn) themeBtn.addEventListener('click', cycleTheme);
    nextBtn.addEventListener('click', nextAction);
    prevBtn.addEventListener('click', prevAction);
    resetBtn.addEventListener('click', resetAll);
    resetSetBtn.addEventListener('click', resetSet);
    retryWrongBtn.addEventListener('click', retryWrong);
    reviewBtn.addEventListener('click', reviewAnswers);
    nextSetBtn.addEventListener('click', goNextSet);
    pauseBtn.addEventListener('click', togglePause);

    // View tabs
    viewTabs.forEach(t => t.addEventListener('click', () => showView(t.dataset.view)));
    // Arrow-key navigation across tabs (WAI-ARIA tablist pattern).
    const tabsArr = Array.from(viewTabs);
    tabsArr.forEach((t, i) => {
        t.addEventListener('keydown', (e) => {
            let next = -1;
            if (e.key === 'ArrowRight') next = (i + 1) % tabsArr.length;
            else if (e.key === 'ArrowLeft') next = (i - 1 + tabsArr.length) % tabsArr.length;
            else if (e.key === 'Home') next = 0;
            else if (e.key === 'End') next = tabsArr.length - 1;
            if (next < 0) return;
            e.preventDefault();
            showView(tabsArr[next].dataset.view);
            tabsArr[next].focus();
        });
    });
    learnSetSelect.addEventListener('change', (e) => {
        state.learnSetIndex = e.target.value === 'all' ? 'all' : +e.target.value;
        saveProgress();
        renderLearn();
    });
    learnRevealToggle.addEventListener('change', (e) => {
        state.learnReveal = e.target.checked;
        saveProgress();
        applyLearnReveal();
    });
    searchInput.addEventListener('input', () => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(renderSearch, 120); // debounce
    });

    // Returning to the tab: drop the away-gap so it isn't billed as study time.
    document.addEventListener('visibilitychange', () => { timer.lastTs = Date.now(); });
    // Persist the latest study time when leaving the page.
    window.addEventListener('beforeunload', saveProgress);

    shuffleToggle.addEventListener('change', (e) => {
        state.shuffle = e.target.checked;
        Object.keys(orderCache).forEach(k => delete orderCache[k]);
        loadSet(state.currentSetIndex, 0); // rebuild order, back to Q1
        renderSidebar();
    });

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (state.view !== 'quiz') return; // keyboard shortcuts only drive the quiz
        // Don't hijack browser/OS chords: Ctrl/⌘+C must copy the question, not pick
        // option C (same for any modified key). Plain keys still drive the quiz.
        if (e.ctrlKey || e.metaKey || e.altKey) return;

        if (e.key.toLowerCase() === 'p') { e.preventDefault(); togglePause(); return; }

        const atSummary = state.currentQuestionIndex >= currentOrder.length;

        // Answer selection 1-4 / a-d (only on an unanswered question)
        if (!atSummary) {
            let pick = -1;
            if (e.key >= '1' && e.key <= '4') pick = +e.key - 1;
            const lower = e.key.toLowerCase();
            if (['a', 'b', 'c', 'd'].includes(lower)) pick = lower.charCodeAt(0) - 97;
            if (pick >= 0) {
                const set = currentSet();
                const q = set.questions[currentOrder[state.currentQuestionIndex]];
                const key = keyFor(set, q);
                if (!state.answered[key] && pick < q.options.length) {
                    e.preventDefault();
                    selectOption(pick, q, key);
                }
                return;
            }
        }

        if (e.key === 'ArrowRight' || e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            nextAction();
        } else if (e.key === 'ArrowLeft') {
            e.preventDefault();
            prevAction();
        }
    });
}

function init() {
    loadProgress();
    // Defensive clamp: a saved index can point past the bank if sets were removed
    // (or a save predates a bank edit). Never let it crash the first render.
    const lastSet = quizData.sets.length - 1;
    state.currentSetIndex = Math.min(Math.max(state.currentSetIndex | 0, 0), lastSet);
    if (state.learnSetIndex !== 'all')
        state.learnSetIndex = Math.min(Math.max(state.learnSetIndex | 0, 0), lastSet);
    if (!['quiz', 'learn', 'search', 'mistakes'].includes(state.view)) state.view = 'quiz';
    if (!THEME_CYCLE.includes(state.theme)) state.theme = 'auto';
    applyTheme();
    const totalQ = quizData.sets.reduce((n, s) => n + s.questions.length, 0);
    const sub = el('appSubtitle');
    if (sub) sub.textContent = `Knowledge Test · ${quizData.sets.length} sets · ${totalQ} Q`;
    shuffleToggle.checked = state.shuffle;
    const startPos = state.currentQuestionIndex;
    currentOrder = fullOrder(state.currentSetIndex);
    state.currentQuestionIndex = Math.min(Math.max(startPos, 0), currentOrder.length);
    renderSidebar();
    render();
    attachEventListeners();
    showView(state.view || 'quiz'); // restore last view (Quiz/Learn/Search)
    timer.lastTs = Date.now();
    renderTimer();
    setInterval(timerTick, 500);
}

init();
