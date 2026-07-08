// Shared utilities for the RLinf Sphinx AI assistant.
(function () {
  'use strict';

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function escapeAttribute(value) {
    return escapeHtml(value).replace(/`/g, '&#96;');
  }

  function sanitizeUrl(value) {
    const raw = String(value == null ? '' : value).trim();
    if (!raw) return '#';

    try {
      const url = new URL(raw, window.location.href);
      if (!['http:', 'https:', 'mailto:'].includes(url.protocol)) {
        return '#';
      }
      return url.href;
    } catch (error) {
      return '#';
    }
  }

  function renderInlineMarkdown(value) {
    return value
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_match, label, href) {
        const safeHref = sanitizeUrl(href);
        return '<a href="' + escapeAttribute(safeHref) + '" target="_blank" rel="noopener noreferrer">' + label + '</a>';
      })
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/__([^_]+)__/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  }

  function renderMarkdown(markdown) {
    if (!markdown) return '';

    let text = String(markdown);
    const codeBlocks = [];
    const inlineCodes = [];

    text = text.replace(/```([\s\S]*?)```/g, function (_match, content) {
      const index = codeBlocks.length;
      codeBlocks.push('<pre><code>' + escapeHtml(String(content).trim()) + '</code></pre>');
      return '\u0000CODE_BLOCK_' + index + '\u0000';
    });

    text = text.replace(/`([^`]+)`/g, function (_match, content) {
      const index = inlineCodes.length;
      inlineCodes.push('<code>' + escapeHtml(content) + '</code>');
      return '\u0000INLINE_CODE_' + index + '\u0000';
    });

    const paragraphs = [];
    const lines = text.split(/\n/);
    let currentParagraph = [];
    let currentList = [];

    function flushParagraph() {
      if (!currentParagraph.length) return;
      const body = renderInlineMarkdown(escapeHtml(currentParagraph.join(' ')));
      paragraphs.push('<p>' + body + '</p>');
      currentParagraph = [];
    }

    function flushList() {
      if (!currentList.length) return;
      paragraphs.push('<ul>' + currentList.map(function (item) {
        return '<li>' + renderInlineMarkdown(escapeHtml(item)) + '</li>';
      }).join('') + '</ul>');
      currentList = [];
    }

    lines.forEach(function (line) {
      const trimmed = line.trim();
      if (!trimmed) {
        flushParagraph();
        flushList();
        return;
      }

      const heading = /^(#{1,3})\s+(.+)$/.exec(trimmed);
      if (heading) {
        flushParagraph();
        flushList();
        const level = heading[1].length;
        paragraphs.push('<h' + level + '>' + renderInlineMarkdown(escapeHtml(heading[2])) + '</h' + level + '>');
        return;
      }

      const bullet = /^[-*]\s+(.+)$/.exec(trimmed);
      if (bullet) {
        flushParagraph();
        currentList.push(bullet[1]);
        return;
      }

      flushList();
      currentParagraph.push(trimmed);
    });

    flushParagraph();
    flushList();

    let html = paragraphs.join('');

    codeBlocks.forEach(function (block, index) {
      html = html.replace('\u0000CODE_BLOCK_' + index + '\u0000', block);
    });
    inlineCodes.forEach(function (code, index) {
      html = html.replace('\u0000INLINE_CODE_' + index + '\u0000', code);
    });

    return html;
  }

  function titleFromDocument(doc) {
    return String(
      doc.title ||
      doc['hierarchy.lvl2'] ||
      (doc.hierarchy && doc.hierarchy.lvl2) ||
      doc['hierarchy.lvl1'] ||
      (doc.hierarchy && doc.hierarchy.lvl1) ||
      doc['hierarchy.lvl0'] ||
      (doc.hierarchy && doc.hierarchy.lvl0) ||
      'Untitled Source'
    ).trim();
  }

  function urlWithAnchor(url, anchor) {
    const rawUrl = String(url || '').trim();
    const rawAnchor = String(anchor || '').trim().replace(/^#/, '');
    if (!rawUrl) return '#';
    if (!rawAnchor || rawUrl.includes('#')) return rawUrl;
    return rawUrl.split('#')[0] + '#' + encodeURIComponent(rawAnchor);
  }

  function normalizeSource(source) {
    if (!source) {
      return { title: 'Invalid Source Data', excerpt: '', url: '#' };
    }

    const doc = source.document || source;
    const title = source.title || titleFromDocument(doc);
    const content = source.excerpt || source.highlightedContent || doc.content || '';
    const excerpt = String(content).trim().slice(0, 180);
    const url = sanitizeUrl(urlWithAnchor(source.url || doc.url, source.anchor || doc.anchor));

    return {
      title: title || 'Untitled Source',
      excerpt: excerpt + (String(content).trim().length > 180 ? '...' : ''),
      url
    };
  }

  function createSourcesHTML(sources) {
    const normalized = (sources || [])
      .map(normalizeSource)
      .filter(function (source) { return source.url && source.url !== '#'; })
      .slice(0, 5);

    if (!normalized.length) return '';

    const items = normalized.map(function (source) {
      return [
        '<div class="sphinx-source">',
        '<a href="' + escapeAttribute(source.url) + '" target="_blank" rel="noopener noreferrer">',
        '<span class="sphinx-source-title">' + escapeHtml(source.title) + '</span>',
        source.excerpt ? '<span class="sphinx-source-excerpt">' + escapeHtml(source.excerpt) + '</span>' : '',
        '</a>',
        '</div>'
      ].join('');
    }).join('');

    return '<div class="sphinx-sources"><h4>References</h4>' + items + '</div>';
  }

  function normalizeQueryText(value) {
    return String(value || '').trim().toLowerCase();
  }

  function analyzeQuery(query) {
    const text = String(query || '').trim();
    const normalized = normalizeQueryText(text);
    const hasChinese = /[\u3400-\u9fff]/.test(text);
    const hasLatin = /[a-z]/i.test(text);
    const technicalPattern = /(?:[A-Z]{2,}|[a-z]+[-_/][a-z0-9_.-]+|[a-z0-9_.-]+\([^)]*\)|\/[a-z0-9_.\/-]+|bge-m3|grpo|ppo|dapo|fsdp|vllm|sglang|cuda|api|yaml)/i;
    const isTechnical = technicalPattern.test(text);
    const tokenCount = normalized.split(/\s+/).filter(Boolean).length;
    const complexity = text.length > 80 || tokenCount > 9 ? 'complex' : 'simple';

    let queryType = 'english_only';
    if (hasChinese && hasLatin) {
      queryType = isTechnical ? 'technical_mixed' : 'mixed_multilingual';
    } else if (hasChinese) {
      queryType = 'chinese_only';
    } else if (isTechnical) {
      queryType = 'technical_mixed';
    }

    return {
      queryType,
      complexity,
      isTechnical,
      semanticWeight: complexity === 'complex' ? 0.45 : 0.35,
      exactMatchWeight: complexity === 'complex' ? 0.55 : 0.65
    };
  }

  function strategyForAnalysis(analysis) {
    switch (analysis.queryType) {
      case 'chinese_only':
        return {
          query_by: 'embedding,content,hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3',
          query_by_weights: analysis.complexity === 'simple' ? '0,4,3,2,1,1' : '0,4,2,2,1,1',
          num_typos: 1,
          drop_tokens_threshold: 1,
          per_page: analysis.complexity === 'simple' ? 8 : 12
        };
      case 'technical_mixed':
        return {
          query_by: 'embedding,hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3,content',
          query_by_weights: '0,5,4,3,2,2',
          num_typos: 1,
          drop_tokens_threshold: 1,
          per_page: 10
        };
      case 'mixed_multilingual':
        return {
          query_by: 'embedding,content,hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3',
          query_by_weights: '0,3,3,2,2,1',
          num_typos: 2,
          drop_tokens_threshold: 2,
          per_page: 10
        };
      case 'english_only':
      default:
        return {
          query_by: 'embedding,hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3,content',
          query_by_weights: analysis.complexity === 'simple' ? '0,5,4,3,2,2' : '0,3,3,2,2,3',
          num_typos: 2,
          drop_tokens_threshold: 2,
          per_page: analysis.complexity === 'simple' ? 8 : 12
        };
    }
  }

  function createQueryPlan(options) {
    const originalQuery = String((options && options.message) || '').trim();
    const overviewPattern = /(?:what\s+is|what\s+does|about|overview|introduce|introduction|简介|是什么).*(?:rlinf)|(?:rlinf).*(?:what\s+is|what\s+does|about|overview|introduce|introduction|简介|是什么)/i;
    const isProductOverview = overviewPattern.test(originalQuery);
    const searchQuery = isProductOverview ? 'RLinf' : originalQuery;
    const analysis = analyzeQuery(searchQuery);

    return {
      originalQuery,
      searchQuery,
      sourceScope: 'docs',
      analysis,
      retrievalMode: isProductOverview ? 'text' : 'hybrid',
      resetConversation: isProductOverview,
      display: {
        searchingFor: searchQuery,
        sourceScope: 'Documentation',
        notes: isProductOverview
          ? ['Overview query strategy']
          : (analysis.isTechnical ? ['Technical query strategy'] : [])
      },
      planner: 'deterministic'
    };
  }

  function withoutEmbedding(strategy) {
    const fields = strategy.query_by.split(',');
    const weights = strategy.query_by_weights.split(',');
    const nextFields = [];
    const nextWeights = [];

    fields.forEach(function (field, index) {
      if (field === 'embedding') return;
      nextFields.push(field);
      if (weights[index]) nextWeights.push(weights[index]);
    });

    return Object.assign({}, strategy, {
      query_by: nextFields.join(','),
      query_by_weights: nextWeights.join(',')
    });
  }

  function buildAIChatMultiSearchParameters(options) {
    const config = options.config || {};
    const queryPlan = options.queryPlan || createQueryPlan({ message: options.message });
    const analysis = queryPlan.analysis || analyzeQuery(options.message);
    const strategy = strategyForAnalysis(analysis);
    const collectionName = config.collectionName;
    const useVector = queryPlan.retrievalMode !== 'text' &&
      config.enableVectorSearch !== false &&
      options.includeVectorSearch !== false;
    const activeStrategy = useVector ? strategy : withoutEmbedding(strategy);
    const alpha = analysis.complexity === 'complex' ? 0.35 : 0.25;
    const distanceThreshold = analysis.complexity === 'complex' ? 0.65 : 0.58;

    const search = Object.assign({
      collection: collectionName,
      prefix: false,
      text_match_type: analysis.complexity === 'simple' || analysis.isTechnical ? 'max_weight' : 'sum_score',
      highlight_full_fields: 'content,hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3'
    }, activeStrategy);

    if (useVector) {
      search.vector_query = 'embedding:([], k: 20, distance_threshold: ' + distanceThreshold + ', alpha: ' + alpha + ')';
      search.rerank_hybrid_matches = analysis.complexity !== 'simple';
      search.exclude_fields = 'embedding';
    }

    if (config.stopwordsSet) {
      search.stopwords = config.stopwordsSet;
    }

    const commonParams = {
      q: options.message,
      conversation_model_id: options.modelId,
      conversation: true
    };

    if (options.conversationId && !queryPlan.resetConversation) {
      commonParams.conversation_id = options.conversationId;
    }
    if (options.isStreaming) {
      commonParams.conversation_stream = true;
    }

    return {
      commonParams,
      requestBody: {
        searches: [search]
      },
      queryPlan
    };
  }

  function extractConversationId(response, fallback) {
    return response &&
      ((response.conversation && response.conversation.conversation_id) ||
      (response.results && response.results[0] && response.results[0].conversation && response.results[0].conversation.conversation_id) ||
      fallback);
  }

  function createSearchRequestControl() {
    let latestRequestId = 0;
    let pendingTimer = null;

    function clearPendingTimer() {
      if (pendingTimer) {
        clearTimeout(pendingTimer);
        pendingTimer = null;
      }
    }

    function startNextRequest() {
      clearPendingTimer();
      latestRequestId += 1;
      return latestRequestId;
    }

    return {
      runNow: function (runner) {
        const requestId = startNextRequest();
        return Promise.resolve(runner({ requestId: requestId }));
      },
      runDebounced: function (runner, delayMs) {
        const requestId = startNextRequest();
        pendingTimer = setTimeout(function () {
          pendingTimer = null;
          runner({ requestId: requestId });
        }, delayMs);
        return requestId;
      },
      isLatest: function (requestId) {
        return requestId === latestRequestId;
      },
      dispose: function () {
        startNextRequest();
      }
    };
  }

  window.RLinfAssistantUtils = {
    escapeHtml: escapeHtml,
    sanitizeUrl: sanitizeUrl,
    renderMarkdown: renderMarkdown,
    normalizeSource: normalizeSource,
    createSourcesHTML: createSourcesHTML,
    analyzeQuery: analyzeQuery,
    createQueryPlan: createQueryPlan,
    buildAIChatMultiSearchParameters: buildAIChatMultiSearchParameters,
    extractConversationId: extractConversationId,
    createSearchRequestControl: createSearchRequestControl
  };
})();
