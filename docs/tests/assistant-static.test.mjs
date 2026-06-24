import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const syncedFiles = [
  '_static/js/assistant-utils.js',
  '_static/js/config-manager.js',
  '_static/js/typesense-client.js',
  '_static/js/message-manager.js',
  '_static/js/ai-chat-service.js',
  '_static/sphinx-modal-widget.js',
  '_static/css/sphinx-modal.css',
  '_static/css/mode-selection.css'
];

test('assistant runtime files stay synced across Sphinx locale trees', () => {
  for (const relativePath of syncedFiles) {
    const en = fs.readFileSync(`docs/source-en/${relativePath}`, 'utf8');
    const zh = fs.readFileSync(`docs/source-zh/${relativePath}`, 'utf8');
    assert.equal(zh, en, relativePath);
  }
});

test('assistant sync script includes every synced runtime file', () => {
  const syncScript = fs.readFileSync('docs/assistant/sync-static.mjs', 'utf8');

  for (const relativePath of syncedFiles) {
    assert.match(syncScript, new RegExp(`'${relativePath.replaceAll('/', '\\/')}'`));
  }
});

test('widget initializes services through the window config getter', () => {
  const widget = fs.readFileSync('docs/source-en/_static/sphinx-modal-widget.js', 'utf8');
  const initializeServices = widget.match(
    /function initializeServices\(\) \{[\s\S]*?\n  \}/
  )?.[0] || '';

  assert.match(widget, /const config = window\.SphinxAIConfig/);
  assert.match(initializeServices, /const config = getRuntimeConfig\(\)/);
  assert.match(initializeServices, /new SphinxTypesenseClient\(config\.typesense\)/);
  assert.match(initializeServices, /new SphinxAIChatService\(typesenseClient, config\)/);
  assert.doesNotMatch(initializeServices, /SphinxAIConfig\.typesense/);
});

test('Sphinx Typesense search key is runtime-configured', () => {
  for (const confPath of ['docs/source-en/conf.py', 'docs/source-zh/conf.py']) {
    const conf = fs.readFileSync(confPath, 'utf8');
    assert.match(conf, /RLINF_TYPESENSE_SEARCH_API_KEY/);
    assert.doesNotMatch(conf, /"typesense_api_key",\s*"[^\"]+"/);
  }
});

test('Sphinx Typesense config only exposes the RLinf docs collection', () => {
  for (const confPath of ['docs/source-en/conf.py', 'docs/source-zh/conf.py']) {
    const conf = fs.readFileSync(confPath, 'utf8');
    assert.doesNotMatch(conf, /RLINF_TYPESENSE_FAQ_COLLECTION/);
    assert.doesNotMatch(conf, /typesense_faq_collection/);
  }

  for (const layoutPath of ['docs/source-en/_templates/layout.html', 'docs/source-zh/_templates/layout.html']) {
    const layout = fs.readFileSync(layoutPath, 'utf8');
    assert.doesNotMatch(layout, /faqCollectionName/);
  }
});

test('assistant enables Typesense streaming by default', () => {
  const config = fs.readFileSync('docs/source-en/_static/js/config-manager.js', 'utf8');
  const service = fs.readFileSync('docs/source-en/_static/js/ai-chat-service.js', 'utf8');

  assert.match(config, /enableTypesenseStreaming:\s*true/);
  assert.match(service, /streamConfig/);
  assert.match(service, /conversation_stream/);
});

test('Sphinx sitemap uses canonical base URL without duplicating locale/version', () => {
  for (const confPath of ['docs/source-en/conf.py', 'docs/source-zh/conf.py']) {
    const conf = fs.readFileSync(confPath, 'utf8');
    assert.match(conf, /sitemap_url_scheme\s*=\s*"\{link\}"/);
  }
});

test('widget persists conversation state and sources on completion', () => {
  const widget = fs.readFileSync('docs/source-en/_static/sphinx-modal-widget.js', 'utf8');
  const closeHandler = widget.match(/function handleModalClose\(\) \{[\s\S]*?\n    \}/)?.[0] || '';

  assert.match(widget, /messageManager\.setConversationId\(result\.conversationId\)/);
  assert.match(widget, /onComplete: \(sources, response = \{\}\)/);
  assert.match(widget, /finalizeAIMessage\(aiMessage\.id, container, sources/);
  assert.doesNotMatch(closeHandler, /clearChatHistory/);
});

test('widget exposes explicit new chat control', () => {
  const widget = fs.readFileSync('docs/source-en/_static/sphinx-modal-widget.js', 'utf8');

  assert.match(widget, /sphinx-modal-reset/);
  assert.match(widget, /clearChatHistory\(messagesContainer\);/);
});

test('assistant UI follows the Sphinx theme instead of standalone chatbot styling', () => {
  const widget = fs.readFileSync('docs/source-en/_static/sphinx-modal-widget.js', 'utf8');
  const modalCss = fs.readFileSync('docs/source-en/_static/css/sphinx-modal.css', 'utf8');
  const modeCss = fs.readFileSync('docs/source-en/_static/css/mode-selection.css', 'utf8');
  const combinedCss = `${modalCss}\n${modeCss}`;

  assert.match(modalCss, /--sphinx-primary:\s*var\(--pst-color-primary/);
  assert.match(modalCss, /--sphinx-background:\s*var\(--pst-color-background/);
  assert.match(modalCss, /--sphinx-border:\s*var\(--pst-color-border/);

  assert.doesNotMatch(widget, /function injectModalStyles/);
  assert.doesNotMatch(widget, /const avatar =/);
  assert.doesNotMatch(widget, /[👤🤖💡⚠️👋]/u);
  assert.doesNotMatch(combinedCss, /[💡🔗]/u);
  assert.doesNotMatch(combinedCss, /prefers-color-scheme/);
  assert.doesNotMatch(combinedCss, /--sphinx-primary-hue/);
});

test('assistant detail polish keeps modal controls stable and modalized', () => {
  const widget = fs.readFileSync('docs/source-en/_static/sphinx-modal-widget.js', 'utf8');
  const modalCss = fs.readFileSync('docs/source-en/_static/css/sphinx-modal.css', 'utf8');
  const modeCss = fs.readFileSync('docs/source-en/_static/css/mode-selection.css', 'utf8');
  const customCss = fs.readFileSync('docs/source-en/_static/css/custom.css', 'utf8');
  const combinedCss = `${modalCss}\n${modeCss}\n${customCss}`;

  assert.match(combinedCss, /--assistant-control-height:\s*40px/);
  assert.match(combinedCss, /box-sizing:\s*border-box/);
  assert.match(customCss, /\.rlinf-ai-modal-open\s+\.rlinf-ask-ai-fab/);
  assert.match(widget, /classList\.add\('rlinf-ai-modal-open'\)/);
  assert.match(widget, /classList\.remove\('rlinf-ai-modal-open'\)/);
  assert.match(widget, /lastFocusedTrigger/);
  assert.match(widget, /function trapModalFocus/);
  assert.match(widget, /aria-live="polite"/);
  assert.match(widget, /placeholder="Ask RLinf docs\.\.\."/);
  assert.match(widget, /textContent = isBusy \? 'Asking' : 'Ask AI'/);
  assert.doesNotMatch(widget, /Ask about installation, examples, APIs/);
  assert.doesNotMatch(widget, /Thinking\.\.\./);
  assert.doesNotMatch(widget, /textarea\.style\.height = '44px'/);
});

test('assistant source hover underlines only source text, not citation number', () => {
  const modalCss = fs.readFileSync('docs/source-en/_static/css/sphinx-modal.css', 'utf8');
  const markerHover = modalCss.match(/\.sphinx-source:hover a::before\s*\{([\s\S]*?)\}/)?.[1] || '';

  assert.match(
    modalCss,
    /\.sphinx-source:hover a\s*\{[\s\S]*?text-decoration:\s*none/
  );
  assert.match(
    modalCss,
    /\.sphinx-source:hover \.sphinx-source-title,\s*\.sphinx-source:hover \.sphinx-source-excerpt\s*\{[\s\S]*?text-decoration:\s*underline/
  );
  assert.match(markerHover, /text-decoration:\s*none/);
  assert.doesNotMatch(markerHover, /text-decoration:\s*underline/);
});

test('AI chat service uses Typesense multiSearch and not direct document conversation search', () => {
  const service = fs.readFileSync('docs/source-en/_static/js/ai-chat-service.js', 'utf8');

  assert.match(service, /multiSearch\.perform/);
  assert.match(service, /buildAIChatMultiSearchParameters/);
  assert.doesNotMatch(service, /\.documents\(\)\s*\.search\(searchParameters\)/);
});

test('AI chat service reads Typesense conversation answer from top-level response', () => {
  const service = fs.readFileSync('docs/source-en/_static/js/ai-chat-service.js', 'utf8');

  assert.match(
    service,
    /response\?\.conversation\?\.answer\s*\|\|\s*primary\?\.conversation\?\.answer/
  );
});
