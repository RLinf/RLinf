// AI chat service for Sphinx AI Widget - Using Typesense for AI conversations
class SphinxAIChatService {
  constructor(typesenseClient, config) {
    this.typesenseClient = typesenseClient;
    this.config = config;
    this.isOnline = navigator.onLine;
    
    // Streaming toggle - currently disabled, can be enabled in the future
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
  
  // Main method for sending messages - uses Typesense for AI conversation
  async sendMessage(message, conversationId, options = {}) {
    const mode = options.mode || this.config.chat.defaultMode;
    const modeConfig = this.config.chat.modes[mode];
    
    try {
      // Use Typesense for AI conversation instead of direct API calls
      const response = await this.callTypesenseAIService(message, conversationId, modeConfig);
      
      return {
        message: response.message,
        conversationId: response.conversationId,
        sources: response.sources || []
      };
    } catch (error) {
      console.error('AI chat service error:', error);
      
      // Fallback to mock response on any error
      const mockResponse = await this.generateMockResponse(message, mode);
      return {
        message: mockResponse,
        conversationId: conversationId || this.generateFallbackConversationId(),
        sources: [],
        error: error.message
      };
    }
  }
  
  // Streaming message support with Typesense - toggleable implementation
  async sendMessageStreaming(message, conversationId, callbacks, options = {}) {
    const mode = options.mode || this.config.chat.defaultMode;
    const modeConfig = this.config.chat.modes[mode];
    
    try {
      if (this.enableTypesenseStreaming) {
        // Use actual Typesense streaming (future functionality)
        if (this.config.debug) {
          console.log('Using Typesense streaming');
        }
        await this.streamTypesenseResponse(message, conversationId, modeConfig, callbacks);
        
        return {
          conversationId: conversationId,
          sources: []
        };
      } else {
        // Use non-streaming Typesense call (current default)
        if (this.config.debug) {
          console.log('Using non-streaming Typesense call with simulated streaming UI');
        }
        const response = await this.callTypesenseAIService(message, conversationId, modeConfig);
        
        // Simulate streaming by sending the complete response
        if (callbacks.onChunk) {
          callbacks.onChunk(response.message);
        }
        
        if (callbacks.onComplete) {
          callbacks.onComplete(response.sources);
        }
        
        return {
          conversationId: response.conversationId,
          sources: response.sources
        };
      }
    } catch (error) {
      console.error('Streaming AI chat error:', error);
      
      // Fallback to mock streaming
      await this.streamMockResponse(message, mode, callbacks);
      
      return {
        conversationId: conversationId || this.generateFallbackConversationId(),
        sources: [],
        error: error.message
      };
    }
  }
  
  // Call Typesense AI service (like VitePress implementation)
  async callTypesenseAIService(message, conversationId, modeConfig) {
    if (!this.typesenseClient.isReady()) {
      throw new Error('Typesense not connected');
    }
    
    // Build search parameters for Typesense AI conversation
    const searchParameters = {
      q: message,
      prefix: false,
      query_by: 'embedding',
      exclude_fields: 'embedding',
      conversation_model_id: modeConfig.model,
      conversation: true,
      conversation_id: conversationId,
      per_page: 3,
    };
    
    if (this.config.debug) {
      console.log('Typesense AI search parameters:', searchParameters);
    }
    
    try {
      const response = await this.typesenseClient.client
        .collections(this.config.typesense.collectionName)
        .documents()
        .search(searchParameters);
      
      if (!response.conversation?.answer) {
        throw new Error('No conversation answer in response');
      }
      
      // Extract sources from search results
      const sources = this.processSources(response.hits || []);
      
      return {
        message: response.conversation.answer,
        conversationId: response.conversation.conversation_id,
        sources: sources
      };
    } catch (error) {
      console.error('Typesense AI service error:', error);
      throw error;
    }
  }
  
  // Stream Typesense response - FUTURE FUNCTIONALITY
  // Currently disabled, using non-streaming approach by default
  async streamTypesenseResponse(message, conversationId, modeConfig, callbacks) {
    if (!this.typesenseClient.isReady()) {
      throw new Error('Typesense not connected');
    }
    
    const searchParameters = {
      q: message,
      prefix: false,
      query_by: 'embedding',
      exclude_fields: 'embedding',
      conversation_model_id: modeConfig.model,
      conversation: true,
      conversation_id: conversationId,
      conversation_stream: true,
      per_page: 3,
      streamConfig: {
        onChunk: (chunk) => {
          if (callbacks.onChunk) {
            // Process chunk similar to VitePress implementation
            if (typeof chunk === 'string') {
              callbacks.onChunk(chunk);
            } else if (chunk && chunk.message) {
              callbacks.onChunk(chunk.message);
            }
          }
        },
        onError: (error) => {
          if (callbacks.onError) {
            callbacks.onError(error);
          }
        },
        onComplete: (response) => {
          if (callbacks.onComplete) {
            const sources = this.processSources(response.hits || []);
            callbacks.onComplete(sources);
          }
        }
      }
    };
    
    if (this.config.debug) {
      console.log('Typesense streaming parameters:', searchParameters);
    }
    
    await this.typesenseClient.client
      .collections(this.config.typesense.collectionName)
      .documents()
      .search(searchParameters);
  }
  
  // Process search results to sources (similar to VitePress)
  processSources(hits) {
    if (!hits || !Array.isArray(hits)) {
      return [];
    }
    
    return hits.slice(0, 3).map(hit => {
      const doc = hit.document;
      
      if (!doc || !doc.url) {
        return {
          title: 'Invalid Source Data',
          excerpt: '',
          url: '#',
        };
      }
      
      // Use hierarchy information for title
      const title = doc['hierarchy.lvl2']?.trim() 
                 || doc['hierarchy.lvl1']?.trim() 
                 || doc['hierarchy.lvl0']?.trim() 
                 || 'Untitled Source';
      
      // Content excerpt
      let excerpt = 'No preview available';
      const maxExcerptLength = 150;
      
      if (doc.content) {
        excerpt = doc.content.substring(0, maxExcerptLength);
        if (doc.content.length > maxExcerptLength) {
          excerpt += '...';
        }
      }
      
      return {
        title,
        excerpt: excerpt.trim(),
        url: doc.url,
      };
    }).filter(source => source.url !== '#');
  }
  
  // Check if Typesense is available
  canUseTypesense() {
    return this.typesenseClient.isReady() && this.isOnline;
  }
  
  // Toggle Typesense streaming mode (for future use)
  toggleTypesenseStreaming() {
    this.enableTypesenseStreaming = !this.enableTypesenseStreaming;
    
    if (this.config.debug) {
      console.log(`Typesense streaming toggled: ${this.enableTypesenseStreaming}`);
    }
    
    return this.enableTypesenseStreaming;
  }
  
  // Enable Typesense streaming (for future use)
  enableTypesenseStreamingMode() {
    this.enableTypesenseStreaming = true;
    
    if (this.config.debug) {
      console.log('Typesense streaming enabled');
    }
    
    return true;
  }
  
  // Disable Typesense streaming
  disableTypesenseStreamingMode() {
    this.enableTypesenseStreaming = false;
    
    if (this.config.debug) {
      console.log('Typesense streaming disabled');
    }
    
    return false;
  }
  
  // Get current streaming mode
  isTypesenseStreamingEnabled() {
    return this.enableTypesenseStreaming;
  }
  
  // Mock response generation for fallback
  async generateMockResponse(message, mode = 'quick') {
    const modeResponses = {
      quick: [
        'This is a quick simulated response. In a real environment, this will display intelligent replies from the Typesense AI service.',
        'Quick Mode: For demonstration purposes, this is a simulated AI reply. In actual deployment, it will connect to the Typesense server.',
        'Simulated response: Your question has been received. Once Typesense is configured, you will receive intelligent, document-based answers.'
      ],
      balance: [
        'Balanced Mode simulated response: This is a more detailed simulated reply, showing the characteristics of the Balanced Mode. In real use, Typesense AI will provide accurate answers based on your document content.',
        'This is a simulated response for Balanced Mode. In a real environment, the system will search related documents via Typesense and generate a comprehensive reply.'
      ],
      deep: [
        'Deep Research Mode simulated response: This is a detailed, thoroughly analyzed simulated reply. In actual deployment, Deep Research Mode will leverage Typesense’s advanced AI features to fully analyze complex questions, providing deep insights and detailed answers. The system synthesizes multiple document sources, applies logical reasoning, and delivers a structured response.',
        'Deep Research Mode demo: This showcases a simulated analysis process for a complex question. In real use, Typesense AI will perform multi-layered document analysis and reasoning.'
      ]
    };
    
    const responses = modeResponses[mode] || modeResponses.quick;
    const response = responses[Math.floor(Math.random() * responses.length)];
    
    // Simulate processing delay based on mode
    const delays = { quick: 1000, balance: 2000, deep: 3000 };
    await new Promise(resolve => setTimeout(resolve, delays[mode] || 1000));
    
    return response;
  }
  
  // Mock streaming response
  async streamMockResponse(message, mode, callbacks) {
    const response = await this.generateMockResponse(message, mode);
    const chunks = response.split('。').filter(chunk => chunk.trim());
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i] + (i < chunks.length - 1 ? '。' : '');
      
      if (callbacks.onChunk) {
        callbacks.onChunk(chunk);
      }
      
      // Simulate streaming delay
      await new Promise(resolve => setTimeout(resolve, 300));
    }
    
    if (callbacks.onComplete) {
      callbacks.onComplete([]);
    }
  }
  
  // Generate fallback conversation ID
  generateFallbackConversationId() {
    return `mock-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  // Get service status
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

// Export AI chat service class
window.SphinxAIChatService = SphinxAIChatService; 