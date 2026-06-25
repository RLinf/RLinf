import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function loadUtils() {
  const code = fs.readFileSync('docs/source-en/_static/js/assistant-utils.js', 'utf8');
  const context = {
    console,
    URL,
    setTimeout,
    clearTimeout,
    window: {
      location: {
        href: 'https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html'
      }
    }
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(code, context, { filename: 'assistant-utils.js' });
  return context.window.RLinfAssistantUtils;
}

test('markdown rendering escapes HTML and blocks unsafe links', () => {
  const utils = loadUtils();
  const html = utils.renderMarkdown('**safe** <img src=x onerror=alert(1)> [bad](javascript:alert(1))');

  assert.match(html, /<strong>safe<\/strong>/);
  assert.doesNotMatch(html, /<img/i);
  assert.doesNotMatch(html, /<img/i);
  assert.doesNotMatch(html, /javascript:/i);
  assert.match(html, /href="#"/);
});

test('source normalization accepts service and Typesense hit shapes', () => {
  const utils = loadUtils();

  const simple = utils.normalizeSource({
    title: 'Install',
    excerpt: 'Install RLinf',
    url: '/en/latest/start.html'
  });
  assert.equal(simple.title, 'Install');
  assert.equal(simple.excerpt, 'Install RLinf');
  assert.equal(simple.url, 'https://rlinf.readthedocs.io/en/latest/start.html');

  const normalized = utils.normalizeSource({
    document: {
      'hierarchy.lvl1': 'Configuration',
      content: 'Use YAML config files.',
      url: 'https://rlinf.readthedocs.io/en/latest/configuration.html',
      anchor: 'runner'
    }
  });

  assert.equal(normalized.title, 'Configuration');
  assert.equal(
    normalized.url,
    'https://rlinf.readthedocs.io/en/latest/configuration.html#runner'
  );
});

test('AI chat multiSearch parameters keep q in common params and use vector retrieval', () => {
  const utils = loadUtils();
  const built = utils.buildAIChatMultiSearchParameters({
    message: 'BGE-M3 embedding API',
    modelId: 'rlinf-quick-response-model',
    conversationId: 'conv-1',
    config: {
      collectionName: 'infini-RL',
      enableVectorSearch: true
    }
  });

  assert.equal(built.commonParams.q, 'BGE-M3 embedding API');
  assert.equal(built.commonParams.conversation, true);
  assert.equal(built.commonParams.conversation_model_id, 'rlinf-quick-response-model');
  assert.equal(built.commonParams.conversation_id, 'conv-1');
  assert.equal(built.requestBody.searches[0].collection, 'infini-RL');
  assert.match(built.requestBody.searches[0].query_by, /embedding/);
  assert.match(built.requestBody.searches[0].vector_query, /embedding/);
  assert.equal(built.requestBody.searches[0].exclude_fields, 'embedding');
});

test('AI chat multiSearch parameters keep FAQ wording in the docs collection', () => {
  const utils = loadUtils();
  const queryPlan = utils.createQueryPlan({ message: 'FAQ: how do I configure RLinf?' });
  const built = utils.buildAIChatMultiSearchParameters({
    message: queryPlan.searchQuery,
    modelId: 'rlinf-quick-response-model',
    queryPlan,
    config: {
      collectionName: 'infini-RL',
      enableVectorSearch: false
    }
  });

  assert.equal(queryPlan.sourceScope, 'docs');
  assert.equal(queryPlan.display.sourceScope, 'Documentation');
  assert.equal(built.requestBody.searches[0].collection, 'infini-RL');
  assert.doesNotMatch(built.requestBody.searches[0].query_by, /embedding/);
});

test('overview questions reset conversation and use exact text retrieval', () => {
  const utils = loadUtils();
  const queryPlan = utils.createQueryPlan({ message: 'What is RLinf?' });
  const built = utils.buildAIChatMultiSearchParameters({
    message: queryPlan.searchQuery,
    modelId: 'rlinf-quick-response-model',
    conversationId: 'stale-conversation',
    queryPlan,
    config: {
      collectionName: 'infini-RL',
      enableVectorSearch: true
    }
  });

  assert.equal(queryPlan.searchQuery, 'RLinf');
  assert.equal(queryPlan.resetConversation, true);
  assert.equal(queryPlan.retrievalMode, 'text');
  assert.equal(built.commonParams.q, 'RLinf');
  assert.equal(built.commonParams.conversation_id, undefined);
  assert.doesNotMatch(built.requestBody.searches[0].query_by, /embedding/);
  assert.equal(built.requestBody.searches[0].vector_query, undefined);
});

test('request control lets callers ignore stale completions', async () => {
  const utils = loadUtils();
  const control = utils.createSearchRequestControl();
  const applied = [];
  let releaseFirst;
  let releaseSecond;

  void control.runNow(async ({ requestId }) => {
    await new Promise((resolve) => {
      releaseFirst = resolve;
    });
    if (control.isLatest(requestId)) applied.push('first');
  });

  void control.runNow(async ({ requestId }) => {
    await new Promise((resolve) => {
      releaseSecond = resolve;
    });
    if (control.isLatest(requestId)) applied.push('second');
  });

  releaseSecond();
  await new Promise((resolve) => setTimeout(resolve, 0));
  releaseFirst();
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(applied, ['second']);
});
