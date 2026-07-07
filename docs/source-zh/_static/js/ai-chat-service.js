// AI chat service for the RLinf Sphinx assistant.
class SphinxAIChatService {
  constructor(typesenseClient, config) {
    this.typesenseClient = typesenseClient;
    this.config = config;
    this.isOnline = navigator.onLine;
    this.searchClientCache = new Map();

    this.enableTypesenseStreaming = config.chat?.enableTypesenseStreaming || false;
    this.setupNetworkListeners();
  }

  setupNetworkListeners() {
    window.addEventListener('online', () => {
      this.isOnline = true;
      if (this.config.debug) {
        console.log('Network connection restored');
      }
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      if (this.config.debug) {
        console.log('Network connection lost');
      }
    });
  }

  getTimeoutSeconds(modelId) {
    if (modelId && modelId.includes('deep')) return 120;
    return 60;
  }

  getSearchClient(modelId) {
    const cacheKey = modelId || 'default';
    if (this.searchClientCache.has(cacheKey)) {
      return this.searchClientCache.get(cacheKey);
    }

    const ClientCtor = Typesense.SearchClient || Typesense.Client;
    const client = new ClientCtor({
      nodes: [{
        host: this.config.typesense.host,
        port: this.config.typesense.port,
        protocol: this.config.typesense.protocol
      }],
      apiKey: this.config.typesense.apiKey,
      connectionTimeoutSeconds: this.getTimeoutSeconds(modelId),
      sendApiKeyAsQueryParam: false
    });

    this.searchClientCache.set(cacheKey, client);
    return client;
  }

  async sendMessage(message, conversationId, options = {}) {
    const mode = options.mode || this.config.chat.defaultMode;
    const modeConfig = this.config.chat.modes[mode];

    try {
      const response = await this.callTypesenseAIService(message, conversationId, modeConfig, options);
      return {
        message: response.message,
        conversationId: response.conversationId,
        sources: response.sources || [],
        queryPlan: response.queryPlan
      };
    } catch (error) {
      console.error('AI chat service error:', error);

      const mockResponse = await this.generateMockResponse(message, mode);
      return {
        message: mockResponse,
        conversationId: conversationId || this.generateFallbackConversationId(),
        sources: [],
        queryPlan: options.queryPlan || this.createQueryPlan(message),
        error: error.message
      };
    }
  }

  async sendMessageStreaming(message, conversationId, callbacks, options = {}) {
    const mode = options.mode || this.config.chat.defaultMode;
    const modeConfig = this.config.chat.modes[mode];

    try {
      if (this.enableTypesenseStreaming) {
        return await this.streamTypesenseResponse(message, conversationId, modeConfig, callbacks, options);
      }

      const response = await this.callTypesenseAIService(message, conversationId, modeConfig, options);

      if (callbacks.onChunk) {
        callbacks.onChunk(response.message);
      }
      if (callbacks.onComplete) {
        callbacks.onComplete(response.sources, response);
      }

      return {
        conversationId: response.conversationId,
        sources: response.sources,
        queryPlan: response.queryPlan
      };
    } catch (error) {
      console.error('Streaming AI chat error:', error);
      await this.streamMockResponse(message, mode, callbacks);

      return {
        conversationId: conversationId || this.generateFallbackConversationId(),
        sources: [],
        queryPlan: options.queryPlan || this.createQueryPlan(message),
        error: error.message
      };
    }
  }

  createQueryPlan(message) {
    return window.RLinfAssistantUtils?.createQueryPlan
      ? window.RLinfAssistantUtils.createQueryPlan({ message })
      : { originalQuery: message, searchQuery: message, planner: 'fallback' };
  }

  buildMultiSearchParameters(message, conversationId, modeConfig, options = {}, includeVectorSearch = true) {
    const queryPlan = options.queryPlan || this.createQueryPlan(message);
    const modelId = modeConfig.model;
    const utils = window.RLinfAssistantUtils;

    if (!utils?.buildAIChatMultiSearchParameters) {
      throw new Error('RLinf assistant utilities are not loaded');
    }

    const built = utils.buildAIChatMultiSearchParameters({
      message: queryPlan.searchQuery || message,
      modelId,
      conversationId,
      isStreaming: Boolean(options.isStreaming),
      queryPlan,
      config: this.config.typesense,
      includeVectorSearch
    });

    return {
      ...built,
      modelId
    };
  }

  async performMultiSearch(searchClient, requestBody, commonParams) {
    if (!searchClient.multiSearch?.perform) {
      throw new Error('Typesense multiSearch.perform is not available in the loaded client');
    }
    return await searchClient.multiSearch.perform(requestBody, commonParams);
  }

  shouldRetryWithoutVector(error) {
    const message = String(error?.message || error || '');
    return /embedding|vector|vector_query|field/i.test(message);
  }

  async callTypesenseAIService(message, conversationId, modeConfig, options = {}) {
    if (!this.typesenseClient.isReady()) {
      throw new Error('Typesense not connected');
    }

    const searchClient = this.getSearchClient(modeConfig.model);
    let built = this.buildMultiSearchParameters(message, conversationId, modeConfig, options, this.config.typesense.enableVectorSearch !== false);

    if (this.config.debug) {
      console.log('Typesense AI multiSearch parameters:', {
        commonParams: built.commonParams,
        requestBody: built.requestBody
      });
    }

    let response;
    try {
      response = await this.performMultiSearch(searchClient, built.requestBody, built.commonParams);
    } catch (error) {
      if (!built.requestBody.searches[0]?.vector_query || !this.shouldRetryWithoutVector(error)) {
        throw error;
      }

      if (this.config.debug) {
        console.warn('Vector AI chat search failed; retrying without vector search:', error);
      }

      built = this.buildMultiSearchParameters(message, conversationId, modeConfig, options, false);
      response = await this.performMultiSearch(searchClient, built.requestBody, built.commonParams);
    }

    const primary = response?.results?.[0] || response;
    const answer = response?.conversation?.answer || primary?.conversation?.answer;
    if (!answer) {
      throw new Error('No conversation answer in Typesense response');
    }

    const sources = this.processSources(primary.hits || []);
    const conversationIdFromResponse = window.RLinfAssistantUtils.extractConversationId(
      response,
      conversationId || this.generateFallbackConversationId()
    );

    return {
      message: answer,
      conversationId: conversationIdFromResponse,
      sources,
      queryPlan: built.queryPlan
    };
  }

  async streamTypesenseResponse(message, conversationId, modeConfig, callbacks, options = {}) {
    if (!this.typesenseClient.isReady()) {
      throw new Error('Typesense not connected');
    }

    const searchClient = this.getSearchClient(modeConfig.model);
    const built = this.buildMultiSearchParameters(
      message,
      conversationId,
      modeConfig,
      { ...options, isStreaming: true },
      this.config.typesense.enableVectorSearch !== false
    );

    let lastActivity = Date.now();
    let completed = false;
    let watchdogTriggered = false;
    let latestConversationId = conversationId;
    let deliveredContent = '';

    const markActivity = () => {
      lastActivity = Date.now();
    };

    const appendChunk = async (text, chunked = false) => {
      if (!text || !callbacks.onChunk) return;

      const chunks = chunked ? this.splitReadableChunks(text) : [text];
      for (const chunk of chunks) {
        if (!chunk) continue;
        deliveredContent += chunk;
        markActivity();
        callbacks.onChunk(chunk);
        if (chunked && chunks.length > 1) {
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }
    };

    const completeFromRawResponse = async (raw) => {
      if (completed) return;

      const primary = raw?.results?.[0] || raw;
      const answer = raw?.conversation?.answer || primary?.conversation?.answer || '';
      latestConversationId = window.RLinfAssistantUtils.extractConversationId(raw, latestConversationId);

      if (answer) {
        const remaining = answer.startsWith(deliveredContent)
          ? answer.slice(deliveredContent.length)
          : (deliveredContent ? '' : answer);
        await appendChunk(remaining, deliveredContent.length === 0);
      }

      completed = true;
      clearInterval(watchdog);
      callbacks.onComplete?.(this.processSources(primary?.hits || []), {
        conversationId: latestConversationId,
        queryPlan: built.queryPlan
      });
    };

    const watchdog = setInterval(async () => {
      if (completed || watchdogTriggered) return;
      if (Date.now() - lastActivity < 8000) return;

      watchdogTriggered = true;
      clearInterval(watchdog);

      try {
        const rescueParams = { ...built.commonParams };
        delete rescueParams.conversation_stream;
        const rescue = await this.performMultiSearch(searchClient, built.requestBody, rescueParams);
        await completeFromRawResponse(rescue);
      } catch (error) {
        if (callbacks.onError) {
          callbacks.onError('Streaming stalled and rescue failed');
        }
      }
    }, 1500);

    try {
      if (this.canUseFetchStreaming()) {
        const raw = await this.performFetchStreamingMultiSearch(
          built,
          {
            onChunk: (chunk) => appendChunk(chunk),
            onActivity: markActivity,
            onConversationId: (nextConversationId) => {
              latestConversationId = nextConversationId;
            }
          }
        );
        await completeFromRawResponse(raw);

        return {
          conversationId: latestConversationId || conversationId || this.generateFallbackConversationId(),
          sources: [],
          queryPlan: built.queryPlan
        };
      }
    } catch (error) {
      if (this.config.debug) {
        console.warn('Fetch streaming failed; falling back to Typesense SDK streaming:', error);
      }
    }

    const streamConfig = {
      onChunk: (chunk) => {
        markActivity();
        if (!chunk) return;

        if (typeof chunk === 'string') {
          appendChunk(chunk);
          return;
        }

        if (chunk.conversation_id) {
          latestConversationId = chunk.conversation_id;
        }
        if (typeof chunk.message === 'string') {
          appendChunk(chunk.message);
        }
      },
      onError: (error) => {
        completed = true;
        clearInterval(watchdog);
        callbacks.onError?.(error);
      },
      onComplete: async (raw) => {
        await completeFromRawResponse(raw);
      }
    };

    try {
      await this.performMultiSearch(
        searchClient,
        built.requestBody,
        { ...built.commonParams, streamConfig }
      );
    } catch (error) {
      clearInterval(watchdog);
      if (this.config.debug) {
        console.warn('Streaming multiSearch failed; falling back to non-streaming response:', error);
      }
      const fallback = await this.callTypesenseAIService(message, conversationId, modeConfig, options);
      await appendChunk(fallback.message, deliveredContent.length === 0);
      callbacks.onComplete?.(fallback.sources, fallback);
      latestConversationId = fallback.conversationId;
    }

    return {
      conversationId: latestConversationId || conversationId || this.generateFallbackConversationId(),
      sources: [],
      queryPlan: built.queryPlan
    };
  }

  canUseFetchStreaming() {
    return typeof fetch === 'function' &&
      typeof TextDecoder === 'function' &&
      typeof URL === 'function';
  }

  buildMultiSearchUrl(commonParams) {
    const port = this.config.typesense.port ? `:${this.config.typesense.port}` : '';
    const url = new URL(`${this.config.typesense.protocol}://${this.config.typesense.host}${port}/multi_search`);

    for (const [key, value] of Object.entries(commonParams)) {
      if (value === undefined || value === null) continue;
      url.searchParams.set(key, String(value));
    }

    return url;
  }

  splitReadableChunks(text, targetSize = 48) {
    if (!text || text.length <= targetSize) return [text];

    const chunks = [];
    let index = 0;
    while (index < text.length) {
      let end = Math.min(index + targetSize, text.length);
      if (end < text.length) {
        const breakAt = text.lastIndexOf(' ', end);
        if (breakAt > index + Math.floor(targetSize / 2)) {
          end = breakAt + 1;
        }
      }
      chunks.push(text.slice(index, end));
      index = end;
    }

    return chunks;
  }

  async performFetchStreamingMultiSearch(built, handlers) {
    const url = this.buildMultiSearchUrl(built.commonParams);
    const response = await fetch(url.toString(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'X-TYPESENSE-API-KEY': this.config.typesense.apiKey
      },
      body: JSON.stringify(built.requestBody)
    });

    if (!response.ok) {
      throw new Error(`Typesense streaming request failed with HTTP ${response.status}: ${await response.text()}`);
    }

    if (!response.body?.getReader) {
      const json = await response.json();
      return json;
    }

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('text/event-stream')) {
      return await response.json();
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalPayload = null;

    const processEvent = async (rawEvent) => {
      const dataLines = rawEvent
        .split(/\r?\n/)
        .map(line => line.trim())
        .filter(line => line.startsWith('data:'))
        .map(line => line.slice(5).trim());

      if (dataLines.length === 0) return;

      const data = dataLines.join('\n');
      if (!data || data === '[DONE]') {
        handlers.onActivity?.();
        return;
      }

      let payload;
      try {
        payload = JSON.parse(data);
      } catch (error) {
        await handlers.onChunk?.(data);
        return;
      }

      handlers.onActivity?.();

      const delta = payload.choices?.[0]?.delta;
      if (typeof delta?.content === 'string' && delta.content) {
        await handlers.onChunk?.(delta.content);
      }

      if (typeof payload.message === 'string' && payload.message) {
        await handlers.onChunk?.(payload.message);
      }

      const conversationId = payload.conversation_id ||
        payload.conversation?.conversation_id ||
        payload.results?.[0]?.conversation?.conversation_id;
      if (conversationId) {
        handlers.onConversationId?.(conversationId);
      }

      if (payload.conversation || payload.results) {
        finalPayload = payload;
      }
    };

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');
      let separatorIndex;
      while ((separatorIndex = buffer.indexOf('\n\n')) >= 0) {
        const rawEvent = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        await processEvent(rawEvent);
      }
    }

    buffer += decoder.decode();
    buffer = buffer.replace(/\r\n/g, '\n');
    if (buffer.trim()) {
      await processEvent(buffer);
    }

    if (!finalPayload) {
      throw new Error('Typesense stream ended without a final conversation payload');
    }

    return finalPayload;
  }

  processSources(hits) {
    if (!hits || !Array.isArray(hits)) {
      return [];
    }

    return hits
      .slice(0, 5)
      .map((hit) => window.RLinfAssistantUtils.normalizeSource(hit))
      .filter((source) => source.url !== '#');
  }

  canUseTypesense() {
    return this.typesenseClient.isReady() && this.isOnline;
  }

  toggleTypesenseStreaming() {
    this.enableTypesenseStreaming = !this.enableTypesenseStreaming;

    if (this.config.debug) {
      console.log(`Typesense streaming toggled: ${this.enableTypesenseStreaming}`);
    }

    return this.enableTypesenseStreaming;
  }

  enableTypesenseStreamingMode() {
    this.enableTypesenseStreaming = true;

    if (this.config.debug) {
      console.log('Typesense streaming enabled');
    }

    return true;
  }

  disableTypesenseStreamingMode() {
    this.enableTypesenseStreaming = false;

    if (this.config.debug) {
      console.log('Typesense streaming disabled');
    }

    return false;
  }

  isTypesenseStreamingEnabled() {
    return this.enableTypesenseStreaming;
  }

  async generateMockResponse(message, mode = 'quick') {
    const modeResponses = {
      quick: [
        'Typesense is not available right now, so this is a local fallback response.',
        'I received your question. Configure the RLinf Typesense search key to enable document-grounded answers.'
      ],
      balance: [
        'Balanced Mode fallback: the assistant could not reach Typesense, so no document-grounded answer is available.'
      ],
      deep: [
        'Deep Research fallback: the assistant could not reach Typesense. Please retry when the search service is available.'
      ]
    };

    const responses = modeResponses[mode] || modeResponses.quick;
    const response = responses[Math.floor(Math.random() * responses.length)];
    const delays = { quick: 200, balance: 300, deep: 400 };
    await new Promise(resolve => setTimeout(resolve, delays[mode] || 200));

    return response;
  }

  async streamMockResponse(message, mode, callbacks) {
    const response = await this.generateMockResponse(message, mode);

    if (callbacks.onChunk) {
      callbacks.onChunk(response);
    }
    if (callbacks.onComplete) {
      callbacks.onComplete([]);
    }
  }

  generateFallbackConversationId() {
    return `mock-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  getServiceStatus() {
    return {
      typesenseReady: this.canUseTypesense(),
      online: this.isOnline,
      mockMode: !this.canUseTypesense(),
      streamingEnabled: this.enableTypesenseStreaming,
      streamingMode: this.enableTypesenseStreaming ? 'typesense-streaming' : 'non-streaming'
    };
  }
}

window.SphinxAIChatService = SphinxAIChatService;
