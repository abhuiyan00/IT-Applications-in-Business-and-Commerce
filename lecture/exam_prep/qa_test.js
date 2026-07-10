// Headless QA harness for the IT Apps quiz app.
// No browser/jsdom needed: a minimal DOM shim loads the REAL chapters.js + app.js,
// then drives user flows through real event handlers and asserts via the DOM + the
// persisted state (app.js writes full state to localStorage on every change).
//
//   node qa_test.js          # run all checks, exits non-zero on failure
//
// Re-run this after any change to app.js / chapters.js / index.html.
const fs = require('fs');
const vm = require('vm');

// ---------- tiny DOM shim ----------------------------------------------------
const ALL = [];
function matchSel(node, sel) {
    if (sel[0] === '.') return node._class.has(sel.slice(1));
    return node.tag === sel.toLowerCase();
}
class El {
    constructor(tag) {
        this.tag = (tag || 'div').toLowerCase();
        this.children = []; this.parentNode = null;
        this._class = new Set(); this._html = ''; this._text = '';
        this.style = {}; this.dataset = {}; this.attrs = {};
        this.handlers = {}; this.hidden = false; this.disabled = false;
        this.checked = false; this.value = '';
        ALL.push(this);
        const self = this;
        this.classList = {
            add: (...c) => c.forEach(x => self._class.add(x)),
            remove: (...c) => c.forEach(x => self._class.delete(x)),
            contains: (c) => self._class.has(c),
            toggle: (c, force) => {
                const on = force === undefined ? !self._class.has(c) : !!force;
                on ? self._class.add(c) : self._class.delete(c);
                return on;
            },
        };
    }
    get className() { return [...this._class].join(' '); }
    set className(v) { this._class = new Set(String(v).split(/\s+/).filter(Boolean)); }
    get textContent() { return this._text; }
    set textContent(v) { this._text = String(v); }
    get innerHTML() { return this._html; }
    set innerHTML(v) { this._html = String(v); this.children = []; } // string render: drop child nodes
    get options() { return this._descend().filter(n => n.tag === 'option'); }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    getAttribute(k) { return this.attrs[k]; }
    removeAttribute(k) { delete this.attrs[k]; }
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
    addEventListener(type, fn) { (this.handlers[type] = this.handlers[type] || []).push(fn); }
    _descend(out = []) { this.children.forEach(c => { out.push(c); c._descend(out); }); return out; }
    querySelectorAll(sel) { return this._descend().filter(n => matchSel(n, sel)); }
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
    dispatch(type, ev = {}) { (this.handlers[type] || []).forEach(fn => fn(ev)); }
    focus() {}
}

const byId = {};
const IDS = `appSubtitle learnList learnRevealToggle learnSetSelect learnView navButtons nextBtn
nextSetBtn optionsContainer overallAccuracy overallAnswered pauseBtn prevBtn progressFill progressText
qIndex questionCard questionText questionTimer quizTopBar quizView reasoningPanel reasoningText resetBtn
resetSetBtn retryWrongBtn reviewBtn scoreText searchCount searchInput searchResults searchView setTag
setsNav shuffleToggle themeBtn summaryCard summaryDetail summaryPercent summaryRing summaryTitle totalTimer
mistakesView mistakesList mistakesCount`.split(/\s+/);
IDS.forEach(id => { const e = new El(id === 'learnSetSelect' ? 'select' : 'div'); e.id = id; byId[id] = e; });

// view tabs + timer-bar (queried by selector, not id)
['quiz', 'learn', 'search', 'mistakes'].forEach(v => { const t = new El('button'); t._class.add('view-tab'); t.dataset.view = v; });
const timerBar = new El('div'); timerBar._class.add('timer-bar');

const docHandlers = {};
const htmlEl = new El('html');
const document = {
    hidden: false,
    documentElement: htmlEl,
    getElementById: id => byId[id] || null,
    createElement: tag => new El(tag),
    querySelector: sel => ALL.find(n => matchSel(n, sel)) || null,
    querySelectorAll: sel => ALL.filter(n => matchSel(n, sel)),
    addEventListener: (t, fn) => { (docHandlers[t] = docHandlers[t] || []).push(fn); },
};
const store = {};
const localStorage = {
    getItem: k => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: k => { delete store[k]; },
};
// Optionally poison the saved state to prove init() can't be crashed by a stale
// out-of-range index (the likely cause of the original "previous section" error).
if (process.env.QA_POISON) {
    store['itAppsQuizState'] = JSON.stringify({
        version: 5, currentSetIndex: 9999, currentQuestionIndex: 9999,
        learnSetIndex: 9999, view: 'bogus', answered: {}, timeSpent: {}, shuffle: false,
    });
}
let confirmReturn = true;
const sandbox = {
    document, localStorage, console,
    window: { addEventListener() {} },
    confirm: () => confirmReturn, alert: () => {},
    setInterval: () => 0, clearTimeout: () => {},
    setTimeout: (fn) => { fn(); return 0; },     // run debounce synchronously
};

// ---------- load the real app ------------------------------------------------
const src = fs.readFileSync('chapters.js', 'utf8') + '\n' + fs.readFileSync('app.js', 'utf8');

// ---------- assertions -------------------------------------------------------
let pass = 0, fail = 0; const fails = [];
function ok(cond, msg) { if (cond) { pass++; } else { fail++; fails.push(msg); console.log('  ✗ ' + msg); } }
function section(s) { console.log('\n' + s); }
function getState() { return JSON.parse(store['itAppsQuizState'] || '{}'); }
function setBtns() { return byId.setsNav.querySelectorAll('.set-btn'); }
function optBtns() { return byId.optionsContainer.querySelectorAll('.option-btn'); }
function tab(v) { return ALL.find(n => n._class.has('view-tab') && n.dataset.view === v); }

let threw = null;
try { vm.runInNewContext(src, sandbox, { filename: 'app-bundle.js' }); }
catch (e) { threw = e; }

section('Load & initial render');
ok(!threw, 'app loads without throwing' + (threw ? ': ' + threw.message : ''));
// A poisoned/stale save must not crash init; it may legitimately land on a summary
// view, so normalize to set 0 / Q1 before the question-level checks.
if (process.env.QA_POISON && setBtns().length) setBtns()[0].dispatch('click');
ok(byId.questionText.textContent.length > 0, 'first question text rendered');
ok(optBtns().length === 4, 'first question shows 4 options (got ' + optBtns().length + ')');
ok(getState().view === 'quiz', 'starts in quiz view');

section('Answering a question');
try {
    const correct = sandbox.quizData ? null : null; // state is internal; assert via DOM/storage
    optBtns()[0].dispatch('click');
    ok(byId.reasoningPanel._class.has('show'), 'explanation panel shown after answering');
    ok(!byId.nextBtn.disabled, 'Next enabled after answering');
    const st = getState();
    ok(Object.keys(st.answered).length === 1, 'one answer persisted');
} catch (e) { ok(false, 'answering threw: ' + e.message); }

section('Previous-section navigation (reported bug)');
try {
    const btns = setBtns();
    ok(btns.length >= 2, 'multiple sets in sidebar (' + btns.length + ')');
    btns[2].dispatch('click');                 // jump to 3rd set, question 1
    const before = byId.progressText.textContent;
    byId.prevBtn.dispatch('click');            // <-- the action that used to error
    const after = byId.progressText.textContent;
    ok(true, 'clicking Previous at a set boundary did not throw');
    ok(/Question\s+\d+\s+of\s+\d+/.test(after), 'previous set renders a valid question ("' + after + '")');
    ok(getState().view === 'quiz', 'stayed in quiz view after crossing set boundary');
} catch (e) { ok(false, 'previous-section navigation threw: ' + e.message); }

section('Cross-boundary sweep: Prev from every set\'s first question');
try {
    let bad = 0;
    const n = setBtns().length;
    for (let i = 0; i < n; i++) {
        setBtns()[i].dispatch('click');        // set i, Q1
        byId.prevBtn.dispatch('click');        // go back across boundary (or no-op at set 0)
        if (!/Question\s+\d+\s+of\s+\d+|complete/i.test(byId.progressText.textContent)) bad++;
    }
    ok(bad === 0, 'Prev from every set boundary renders cleanly (' + bad + ' bad)');
} catch (e) { ok(false, 'boundary sweep threw: ' + e.message); }

section('Learn view');
try {
    tab('learn').dispatch('click');
    ok(byId.learnView.hidden === false, 'learn view visible');
    ok(byId.quizView.hidden === true, 'quiz view hidden in learn mode');
    ok(/learn-card/.test(byId.learnList.innerHTML), 'learn cards rendered');
    ok(byId.learnSetSelect.options.length > 0, 'topic selector populated (' + byId.learnSetSelect.options.length + ')');
    // switch topic
    byId.learnSetSelect.value = '3';
    byId.learnSetSelect.dispatch('change', { target: byId.learnSetSelect });
    ok(/learn-card/.test(byId.learnList.innerHTML), 'learn cards re-render on topic change');
    // hide answers
    byId.learnRevealToggle.checked = false;
    byId.learnRevealToggle.dispatch('change', { target: byId.learnRevealToggle });
    ok(getState().learnReveal === false, 'reveal toggle persisted');
} catch (e) { ok(false, 'learn view threw: ' + e.message); }

section('Search view');
try {
    tab('search').dispatch('click');
    ok(byId.searchView.hidden === false, 'search view visible');
    byId.searchInput.value = 'blockchain';
    byId.searchInput.dispatch('input', { target: byId.searchInput });
    ok(/learn-card/.test(byId.searchResults.innerHTML), 'search returns results for "blockchain"');
    ok(/match/.test(byId.searchCount.textContent), 'match count shown ("' + byId.searchCount.textContent + '")');
    // short query guard
    byId.searchInput.value = 'a';
    byId.searchInput.dispatch('input', { target: byId.searchInput });
    ok(/at least 2/.test(byId.searchResults.innerHTML), 'short query shows guidance, not results');
    // XSS-safety: angle brackets must be escaped in output
    byId.searchInput.value = '<script>';
    byId.searchInput.dispatch('input', { target: byId.searchInput });
    ok(!/<script>/.test(byId.searchResults.innerHTML), 'search output escapes raw HTML');
} catch (e) { ok(false, 'search view threw: ' + e.message); }

section('Keyboard guarded outside quiz view');
try {
    // currently in search view; a number key must not throw or record an answer
    (docHandlers.keydown || []).forEach(fn => fn({ key: '1', target: { tagName: 'DIV' }, preventDefault() {} }));
    ok(true, 'keydown in non-quiz view ignored without error');
} catch (e) { ok(false, 'keydown outside quiz threw: ' + e.message); }

section('Position resume across set switches');
try {
    tab('quiz').dispatch('click');
    setBtns()[0].dispatch('click');                 // set 0 (resumes saved pos)
    for (let i = 0; i < 60; i++) byId.prevBtn.dispatch('click'); // rewind to Q1
    ok(/Question\s+1\s+of/.test(byId.progressText.textContent), 'rewound to Q1 of set 0');
    optBtns()[0].dispatch('click');                 // answer Q1
    byId.nextBtn.dispatch('click');                 // advance to Q2
    const at = byId.progressText.textContent;
    ok(/Question\s+2\s+of/.test(at), 'advanced to Q2 ("' + at + '")');
    setBtns()[1].dispatch('click');                 // jump to set 1
    setBtns()[0].dispatch('click');                 // come back to set 0
    ok(byId.progressText.textContent === at, 'set 0 resumed at Q2, not restarted ("' + byId.progressText.textContent + '")');
} catch (e) { ok(false, 'position resume threw: ' + e.message); }

section('Ctrl+C does not pick option C');
try {
    setBtns()[2].dispatch('click');                 // fresh set, Q1
    for (let i = 0; i < 60; i++) byId.prevBtn.dispatch('click');
    const beforeAnswered = Object.keys(getState().answered).length;
    (docHandlers.keydown || []).forEach(fn => fn({ key: 'c', ctrlKey: true, target: { tagName: 'DIV' }, preventDefault() {} }));
    ok(Object.keys(getState().answered).length === beforeAnswered, 'Ctrl+C records no answer (copy, not pick C)');
} catch (e) { ok(false, 'ctrl+c guard threw: ' + e.message); }

section('Mistakes view');
try {
    tab('quiz').dispatch('click');
    setBtns()[0].dispatch('click');
    for (let i = 0; i < 60; i++) byId.prevBtn.dispatch('click'); // back to Q1 of set 0
    // Answer several questions all picking option A; with shuffled correct index this
    // guarantees at least one wrong answer lands in the mistakes list.
    for (let i = 0; i < 8; i++) {
        const opts = optBtns();
        if (opts.length && !opts[0]._class.has('disabled')) opts[0].dispatch('click');
        byId.nextBtn.dispatch('click');
    }
    tab('mistakes').dispatch('click');
    ok(byId.mistakesView.hidden === false, 'mistakes view visible');
    ok(/mistake-card/.test(byId.mistakesList.innerHTML), 'mistake cards rendered');
    ok(/review/i.test(byId.mistakesCount.textContent), 'mistake count shown ("' + byId.mistakesCount.textContent + '")');
    // (innerHTML-shim drops child nodes, so assert the jump button via the markup)
    ok(/class="mistake-jump" data-set="\d+" data-qidx="\d+"/.test(byId.mistakesList.innerHTML),
        'each mistake has a Review-in-quiz button with jump targets');
    ok(/your-wrong|✗ your answer/.test(byId.mistakesList.innerHTML), 'the wrong answer is marked on the card');
} catch (e) { ok(false, 'mistakes view threw: ' + e.message); }

section('Back to quiz via tab + reset');
try {
    tab('quiz').dispatch('click');
    ok(byId.quizView.hidden === false, 'quiz view restored');
    confirmReturn = true;
    byId.resetBtn.dispatch('click');
    ok(Object.keys(getState().answered || {}).length === 0, 'reset all clears answers');
} catch (e) { ok(false, 'reset threw: ' + e.message); }

section('Theme toggle (auto → light → dark → auto)');
try {
    ok(getState().theme === 'auto', 'theme starts on auto');
    byId.themeBtn.dispatch('click');
    ok(getState().theme === 'light', 'first click -> light');
    ok(htmlEl.getAttribute('data-theme') === 'light', '<html data-theme="light"> set');
    byId.themeBtn.dispatch('click');
    ok(getState().theme === 'dark' && htmlEl.getAttribute('data-theme') === 'dark', 'second click -> dark');
    byId.themeBtn.dispatch('click');
    ok(getState().theme === 'auto', 'third click -> auto');
    ok(htmlEl.getAttribute('data-theme') === undefined, 'auto removes the data-theme attribute (follows OS)');
} catch (e) { ok(false, 'theme toggle threw: ' + e.message); }

section('Exam bank is de-duplicated & well-formed');
try {
    // chapters.js declares `const quizData` (not attached to the sandbox), so load a
    // fresh copy here just for this data-integrity check.
    const bank = {};
    vm.runInNewContext(fs.readFileSync('chapters.js', 'utf8').replace('const quizData =', 'this.quizData ='), bank);
    const sets = bank.quizData.sets;
    const exam = sets.filter(s => s.group === 'exam');
    ok(exam.length === 8, 'eight Testownik sets (' + exam.length + ')');
    const seen = {}, dup = [];
    let bad = 0;
    exam.forEach(s => s.questions.forEach(q => {
        const k = q.question.trim().toLowerCase();
        if (seen[k]) dup.push(q.question); seen[k] = 1;
        if (q.options.length !== 4 || q.correct < 0 || q.correct > 3) bad++;
    }));
    ok(dup.length === 0, 'no duplicate exam question text (' + dup.slice(0, 2).join(' | ') + ')');
    ok(bad === 0, 'every exam question has 4 options & a valid correct index');
} catch (e) { ok(false, 'exam-bank check threw: ' + e.message); }

// ---------- result -----------------------------------------------------------
console.log(`\n${'='.repeat(48)}\nQA: ${pass} passed, ${fail} failed`);
if (fail) { console.log('FAILURES:\n - ' + fails.join('\n - ')); process.exit(1); }
console.log('ALL GREEN ✓');
