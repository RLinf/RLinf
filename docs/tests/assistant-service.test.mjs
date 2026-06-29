import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function loadServiceContext(perform, overrides = {}) {
  const context = {
    console,
    URL,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    navigator: { onLine: true },
    Typesense: {
      SearchClient: class FakeSearchClient {
        constructor() {
          this.multiSearch = { perform };
        }
      }
    },
    window: {
      location: {
        href: 'https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html'
      },
      addEventListener() {}
    },
    ...overrides
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(
    fs.readFileSync('docs/source-en/_static/js/assistant-utils.js', 'utf8'),
    context,
    { filename: 'assistant-utils.js' }
  );
  vm.runInContext(
    fs.readFileSync('docs/source-en/_static/js/ai-chat-service.js', 'utf8'),
    context,
    { filename: 'ai-chat-service.js' }
  );
  return context;
}

test('AI chat service streams Typesense chunks through callbacks', async () => {
  const calls = [];
  const context = loadServiceContext(async (requestBody, commonParams) => {
    calls.push({ requestBody, commonParams });

    assert.equal(commonParams.conversation_stream, true);
    assert.equal(typeof commonParams.streamConfig?.onChunk, 'function');

    commonParams.streamConfig.onChunk('Install ');
    commonParams.streamConfig.onChunk({
      message: 'RLinf from source.',
      conversation_id: 'conv-stream'
    });
    commonParams.streamConfig.onComplete({
      conversation: {
        conversation_id: 'conv-stream'
      },
      results: [{
        conversation: {
          conversation_id: 'conv-stream'
        },
        hits: [{
          document: {
            'hierarchy.lvl1': 'Installation',
            content: 'Install RLinf from source.',
            url: 'https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html'
          }
        }]
      }]
    });
  });

  const service = new context.window.SphinxAIChatService(
    { isReady: () => true },
    {
      debug: false,
      typesense: {
        host: 'typesense.example.com',
        port: 443,
        protocol: 'https',
        apiKey: 'search-key',
        collectionName: 'infini-RL',
        enableVectorSearch: false
      },
      chat: {
        enableTypesenseStreaming: true,
        defaultMode: 'quick',
        modes: {
          quick: {
            model: 'rlinf-quick-response-model'
          }
        }
      }
    }
  );

  const chunks = [];
  let completed;
  const result = await service.sendMessageStreaming(
    'How do I install RLinf?',
    undefined,
    {
      onChunk: (chunk) => chunks.push(chunk),
      onComplete: (sources, response) => {
        completed = { sources, response };
      }
    },
    { mode: 'quick' }
  );

  assert.equal(calls.length, 1);
  assert.equal(calls[0].commonParams.conversation_model_id, 'rlinf-quick-response-model');
  assert.equal(calls[0].requestBody.searches[0].collection, 'infini-RL');
  assert.deepEqual(chunks, ['Install ', 'RLinf from source.']);
  assert.equal(completed.response.conversationId, 'conv-stream');
  assert.equal(completed.sources[0].title, 'Installation');
  assert.equal(result.conversationId, 'conv-stream');
});

test('AI chat service parses browser SSE stream and keeps reasoning private', async () => {
  const finalAnswer = 'RLinf runs distributed reinforcement learning jobs through Worker, WorkerGroup, and Channel primitives while the scheduler maps logical roles onto runtime processes.';
  const requests = [];
  const encoder = new TextEncoder();
  const context = loadServiceContext(
    async () => {
      throw new Error('SDK streaming should not be used when browser fetch streaming is available');
    },
    {
      TextDecoder,
      fetch: async (url, init) => {
        requests.push({
          url,
          body: JSON.parse(init.body)
        });

        const events = [
          'data: {"choices":[{"delta":{"content":null,"reasoning_content":"private reasoning"},"finish_reason":null}]}\r\n\r\n',
          'data: [DONE]\n\n',
          `data: ${JSON.stringify({
            conversation: {
              conversation_id: 'conv-sse',
              answer: finalAnswer
            },
            results: [{
              conversation: {
                conversation_id: 'conv-sse',
                answer: finalAnswer
              },
              hits: [{
                document: {
                  'hierarchy.lvl1': 'Execution Flow',
                  content: 'Worker, WorkerGroup, and Channel define RLinf execution.',
                  url: 'https://rlinf.readthedocs.io/en/latest/rst_source/concepts/execution_flow.html'
                }
              }]
            }]
          })}\n\n`
        ];

        let index = 0;
        return {
          ok: true,
          status: 200,
          headers: {
            get: (name) => name.toLowerCase() === 'content-type'
              ? 'text/event-stream;charset=utf-8'
              : ''
          },
          body: {
            getReader() {
              return {
                async read() {
                  if (index >= events.length) {
                    return { done: true };
                  }
                  const value = encoder.encode(events[index]);
                  index += 1;
                  return { done: false, value };
                }
              };
            }
          }
        };
      }
    }
  );

  const service = new context.window.SphinxAIChatService(
    { isReady: () => true },
    {
      debug: false,
      typesense: {
        host: 'typesense.example.com',
        port: 443,
        protocol: 'https',
        apiKey: 'search-key',
        collectionName: 'infini-RL',
        enableVectorSearch: false
      },
      chat: {
        enableTypesenseStreaming: true,
        defaultMode: 'quick',
        modes: {
          quick: {
            model: 'rlinf-quick-response-model'
          }
        }
      }
    }
  );

  const chunks = [];
  let completed;
  const result = await service.sendMessageStreaming(
    'Explain RLinf execution flow.',
    undefined,
    {
      onChunk: (chunk) => chunks.push(chunk),
      onComplete: (sources, response) => {
        completed = { sources, response };
      }
    },
    { mode: 'quick' }
  );

  assert.equal(requests.length, 1);
  assert.match(requests[0].url, /conversation_stream=true/);
  assert.equal(requests[0].body.searches[0].collection, 'infini-RL');
  assert.equal(chunks.join(''), finalAnswer);
  assert.doesNotMatch(chunks.join(''), /private reasoning/);
  assert.ok(chunks.length > 1);
  assert.equal(completed.response.conversationId, 'conv-sse');
  assert.equal(completed.sources[0].title, 'Execution Flow');
  assert.equal(result.conversationId, 'conv-sse');
});
