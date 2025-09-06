// Typesense client for Sphinx AI Widget
class SphinxTypesenseClient {
  constructor(config) {
    this.config = config;
    this.client = null;
    this.isConnected = false;
    this.connectionAttempts = 0;
    this.maxRetries = 3;
    this.retryDelay = 1000;
    this.init();
  }
  
  async init() {
    try {
      // Check if Typesense library is available
      if (typeof Typesense === 'undefined') {
        throw new Error('Typesense library not loaded. Make sure typesense.min.js is included.');
      }
      
      // Initialize Typesense client
      this.client = new Typesense.Client({
        nodes: [{
          host: this.config.host,
          port: this.config.port,
          protocol: this.config.protocol
        }],
        apiKey: this.config.apiKey,
        connectionTimeoutSeconds: this.config.connectionTimeoutSeconds,
        retryIntervalSeconds: 2
      });
      
      // Check connection
      await this.checkConnection();
    } catch (error) {
      console.error('Typesense client initialization failed:', error);
      this.isConnected = false;
      this.handleConnectionError(error);
    }
  }
  

  
  async checkConnection() {
    try {
      if (!this.config.apiKey) {
        throw new Error('Typesense API key not configured');
      }
      
      await this.client.health.retrieve();
      this.isConnected = true;
      this.connectionAttempts = 0;
      
      if (SphinxAIConfig.debug) {
        console.log('Typesense connection established successfully');
      }
    } catch (error) {
      this.isConnected = false;
      this.connectionAttempts++;
      
      if (SphinxAIConfig.debug) {
        console.warn(`Typesense connection failed (attempt ${this.connectionAttempts}):`, error);
      }
      
      // Retry connection if under max attempts
      if (this.connectionAttempts < this.maxRetries) {
        setTimeout(() => this.checkConnection(), this.retryDelay * this.connectionAttempts);
      }
      
      throw error;
    }
  }
  
  handleConnectionError(error) {
    if (error.message.includes('API key')) {
      console.warn('Typesense API key not configured or invalid - using fallback search');
    } else if (error.message.includes('Network')) {
      console.warn('Network error connecting to Typesense - check connectivity');
    } else {
      console.warn('Typesense connection error:', error.message);
    }
  }
  
  async searchDocuments(query, options = {}) {
    // Fallback to mock search if not connected
    if (!this.isConnected || !this.config.apiKey) {
      return this.fallbackSearch(query, options);
    }
    
    const {
      page = 1,
      perPage = 10,
      filters = '',
      sortBy = '_text_match:desc'
    } = options;
    
    const searchParameters = {
      q: query,
      query_by: 'content,hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3',
      sort_by: sortBy,
      per_page: perPage,
      page: page,
      highlight_full_fields: 'content,hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3',
      highlight_affix_num_tokens: 2,
      num_typos: 2,
      typo_tokens_threshold: 1
    };
    
    if (filters) {
      searchParameters.filter_by = filters;
    }
    
    try {
      const response = await this.client
        .collections(this.config.collectionName)
        .documents()
        .search(searchParameters);
      
      return this.processSearchResults(response, query);
    } catch (error) {
      console.error('Typesense search failed:', error);
      
      // Fallback to mock search on error
      return this.fallbackSearch(query, options);
    }
  }
  
  processSearchResults(response, originalQuery) {
    if (!response.hits || !Array.isArray(response.hits)) {
      return { hits: [], found: 0, query: originalQuery };
    }
    
    const results = response.hits.map(hit => {
      const document = hit.document;
      const highlights = hit.highlights || [];
      
      // Extract highlighted content
      let highlightedContent = document.content || '';
      const contentHighlight = highlights.find(h => h.field === 'content');
      if (contentHighlight && contentHighlight.snippets) {
        highlightedContent = contentHighlight.snippets.join('...');
      }
      
      return {
        document: {
          id: document.id || '',
          title: document['hierarchy.lvl0'] || 'Untitled',
          hierarchy: {
            lvl0: document['hierarchy.lvl0'] || '',
            lvl1: document['hierarchy.lvl1'] || '',
            lvl2: document['hierarchy.lvl2'] || '',
            lvl3: document['hierarchy.lvl3'] || ''
          },
          content: document.content || '',
          url: document.url || '#',
          anchor: document.anchor || ''
        },
        highlights: highlights,
        highlightedContent: highlightedContent,
        textMatch: hit.text_match || 0
      };
    });
    
    return {
      hits: results,
      found: response.found || 0,
      query: originalQuery,
      page: response.page || 1,
      searchTimeMs: response.search_time_ms || 0
    };
  }
  
  // Fallback search for when Typesense is not available
  fallbackSearch(query, options = {}) {
    if (SphinxAIConfig.debug) {
      console.log('Using fallback search for query:', query);
    }
    
    // Simple text-based search in current page content
    const content = document.querySelector('[role="main"]')?.textContent || '';
    const title = document.title;
    const url = window.location.href;
    
    // Check if query matches current page
    const queryLower = query.toLowerCase();
    const contentLower = content.toLowerCase();
    const titleLower = title.toLowerCase();
    
    const titleMatch = titleLower.includes(queryLower);
    const contentMatch = contentLower.includes(queryLower);
    
    if (titleMatch || contentMatch) {
      // Extract relevant content snippet
      let snippet = '';
      if (contentMatch) {
        const index = contentLower.indexOf(queryLower);
        const start = Math.max(0, index - 100);
        const end = Math.min(content.length, index + 200);
        snippet = content.slice(start, end).trim();
        if (start > 0) snippet = '...' + snippet;
        if (end < content.length) snippet = snippet + '...';
      }
      
      return {
        hits: [{
          document: {
            id: 'current-page',
            title: title,
            hierarchy: {
              lvl0: title,
              lvl1: '',
              lvl2: ''
            },
            content: snippet || content.slice(0, 200) + '...',
            url: url,
            anchor: ''
          },
          highlights: [],
          highlightedContent: snippet,
          textMatch: titleMatch ? 1.0 : 0.5
        }],
        found: 1,
        query: query,
        page: 1,
        searchTimeMs: 0,
        isFallback: true
      };
    }
    
    return {
      hits: [],
      found: 0,
      query: query,
      page: 1,
      searchTimeMs: 0,
      isFallback: true
    };
  }
  
  // Get collection info
  async getCollectionInfo() {
    if (!this.isConnected) {
      return null;
    }
    
    try {
      return await this.client.collections(this.config.collectionName).retrieve();
    } catch (error) {
      console.error('Failed to get collection info:', error);
      return null;
    }
  }
  
  // Check if service is ready
  isReady() {
    return this.isConnected && !!this.config.apiKey;
  }
  
  // Get connection status
  getConnectionStatus() {
    return {
      connected: this.isConnected,
      hasApiKey: !!this.config.apiKey,
      attempts: this.connectionAttempts,
      ready: this.isReady()
    };
  }
}

// Export Typesense client class
window.SphinxTypesenseClient = SphinxTypesenseClient; 