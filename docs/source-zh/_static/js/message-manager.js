// Message management for AI chat
class SphinxMessageManager {
  constructor() {
    this.messages = [];
    this.conversationId = null;
    this.maxMessages = SphinxAIConfig.chat.maxMessages || 50;
    this.storageKey = 'sphinx-ai-messages';
    this.storageVersion = '1.0';
    this.maxStorageAge = 24 * 60 * 60 * 1000; // 24 hours
  }
  
  // Add a new message to the conversation
  addMessage(sender, content, options = {}) {
    const message = {
      id: this.generateMessageId(),
      sender, // 'user' | 'ai'
      content: content || '',
      timestamp: Date.now(),
      sources: options.sources || [],
      isStreaming: options.isStreaming || false,
      isLoading: options.isLoading || false,
      error: options.error || null,
      metadata: options.metadata || {}
    };
    
    this.messages.push(message);
    
    // Limit message history
    if (this.messages.length > this.maxMessages) {
      const removed = this.messages.splice(0, this.messages.length - this.maxMessages);
      if (SphinxAIConfig.debug) {
        console.log(`Removed ${removed.length} old messages to maintain limit`);
      }
    }
    
    this.notifyMessageAdded(message);
    return message;
  }
  
  // Update an existing message
  updateMessage(messageId, updates) {
    const messageIndex = this.messages.findIndex(m => m.id === messageId);
    if (messageIndex === -1) {
      console.warn(`Message ${messageId} not found for update`);
      return null;
    }
    
    const message = this.messages[messageIndex];
    const originalContent = message.content;
    
    // Update message properties
    Object.assign(message, updates);
    message.lastUpdated = Date.now();
    
    // Handle streaming content accumulation
    if (updates.content && message.isStreaming) {
      if (updates.appendContent) {
        message.content = originalContent + updates.content;
      }
    }
    
    this.notifyMessageUpdated(message, updates);
    return message;
  }
  
  // Get a specific message
  getMessage(messageId) {
    return this.messages.find(m => m.id === messageId);
  }
  
  // Get all messages
  getAllMessages() {
    return [...this.messages];
  }
  
  // Get messages by sender
  getMessagesBySender(sender) {
    return this.messages.filter(m => m.sender === sender);
  }
  
  // Get the last message
  getLastMessage() {
    return this.messages[this.messages.length - 1];
  }
  
  // Get the last user message
  getLastUserMessage() {
    for (let i = this.messages.length - 1; i >= 0; i--) {
      if (this.messages[i].sender === 'user') {
        return this.messages[i];
      }
    }
    return null;
  }
  
  // Clear all messages
  clearMessages() {
    const oldCount = this.messages.length;
    this.messages = [];
    this.conversationId = null;
    
    this.notifyMessagesCleared(oldCount);
    
    if (SphinxAIConfig.debug) {
      console.log(`Cleared ${oldCount} messages`);
    }
  }
  
  // Delete a specific message
  deleteMessage(messageId) {
    const messageIndex = this.messages.findIndex(m => m.id === messageId);
    if (messageIndex === -1) {
      return false;
    }
    
    const message = this.messages.splice(messageIndex, 1)[0];
    this.notifyMessageDeleted(message);
    
    if (SphinxAIConfig.debug) {
      console.log(`Deleted message ${messageId}`);
    }
    
    return true;
  }
  
  // Conversation ID management
  setConversationId(id) {
    this.conversationId = id;
    if (SphinxAIConfig.debug) {
      console.log(`Set conversation ID: ${id}`);
    }
  }
  
  getConversationId() {
    return this.conversationId;
  }
  
  // Generate a new conversation ID
  generateConversationId() {
    const id = `conv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.setConversationId(id);
    return id;
  }
  
  // Message formatting and export
  formatMessagesForAI() {
    const relevantMessages = this.messages
      .filter(m => !m.error && m.content.trim())
      .slice(-10); // Last 10 messages for context
    
    return relevantMessages.map(message => ({
      role: message.sender === 'user' ? 'user' : 'assistant',
      content: message.content
    }));
  }
  
  // Export conversation
  exportConversation(format = 'json') {
    const conversation = {
      id: this.conversationId,
      timestamp: Date.now(),
      messageCount: this.messages.length,
      messages: this.messages,
      version: this.storageVersion
    };
    
    switch (format) {
      case 'json':
        return JSON.stringify(conversation, null, 2);
      case 'markdown':
        return this.exportAsMarkdown(conversation);
      case 'text':
        return this.exportAsText(conversation);
      default:
        return conversation;
    }
  }
  
  exportAsMarkdown(conversation) {
    let markdown = `# Sphinx AI Chat Conversation\n\n`;
    markdown += `**Conversation ID:** ${conversation.id}\n`;
    markdown += `**Date:** ${new Date(conversation.timestamp).toLocaleString()}\n`;
    markdown += `**Messages:** ${conversation.messageCount}\n\n`;
    
    conversation.messages.forEach((message, index) => {
      const time = new Date(message.timestamp).toLocaleTimeString();
      const sender = message.sender === 'user' ? '👤 User' : '🤖 AI Assistant';
      
      markdown += `## ${sender} (${time})\n\n`;
      markdown += `${message.content}\n\n`;
      
      if (message.sources && message.sources.length > 0) {
        markdown += `**Sources:**\n`;
        message.sources.forEach(source => {
          markdown += `- [${source.document.title}](${source.document.url})\n`;
        });
        markdown += `\n`;
      }
    });
    
    return markdown;
  }
  
  exportAsText(conversation) {
    let text = `Sphinx AI Chat Conversation\n`;
    text += `========================\n\n`;
    text += `Conversation ID: ${conversation.id}\n`;
    text += `Date: ${new Date(conversation.timestamp).toLocaleString()}\n`;
    text += `Messages: ${conversation.messageCount}\n\n`;
    
    conversation.messages.forEach((message, index) => {
      const time = new Date(message.timestamp).toLocaleTimeString();
      const sender = message.sender === 'user' ? 'User' : 'AI Assistant';
      
      text += `[${time}] ${sender}:\n`;
      text += `${message.content}\n\n`;
    });
    
    return text;
  }
  
  // Local storage persistence
  saveToStorage() {
    try {
      const data = {
        messages: this.messages,
        conversationId: this.conversationId,
        timestamp: Date.now(),
        version: this.storageVersion
      };
      
      localStorage.setItem(this.storageKey, JSON.stringify(data));
      
      if (SphinxAIConfig.debug) {
        console.log('Messages saved to storage');
      }
    } catch (error) {
      console.warn('Failed to save messages to storage:', error);
      
      // Try to clear old data if quota exceeded
      if (error.name === 'QuotaExceededError') {
        this.clearOldStorage();
        try {
          localStorage.setItem(this.storageKey, JSON.stringify(data));
        } catch (retryError) {
          console.error('Failed to save after clearing storage:', retryError);
        }
      }
    }
  }
  
  // Load from local storage
  loadFromStorage() {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (!stored) return;
      
      const data = JSON.parse(stored);
      
      // Check if data is too old
      if (Date.now() - data.timestamp > this.maxStorageAge) {
        if (SphinxAIConfig.debug) {
          console.log('Stored messages are too old, ignoring');
        }
        localStorage.removeItem(this.storageKey);
        return;
      }
      
      // Check version compatibility
      if (data.version !== this.storageVersion) {
        if (SphinxAIConfig.debug) {
          console.log('Storage version mismatch, ignoring stored data');
        }
        localStorage.removeItem(this.storageKey);
        return;
      }
      
      this.messages = data.messages || [];
      this.conversationId = data.conversationId || null;
      
      if (SphinxAIConfig.debug) {
        console.log(`Loaded ${this.messages.length} messages from storage`);
      }
    } catch (error) {
      console.warn('Failed to load messages from storage:', error);
      localStorage.removeItem(this.storageKey);
    }
  }
  
  // Clear old storage data
  clearOldStorage() {
    try {
      // Remove our storage
      localStorage.removeItem(this.storageKey);
      
      // Clean up other potentially old items
      const keys = Object.keys(localStorage);
      keys.forEach(key => {
        if (key.startsWith('sphinx-ai') && key !== this.storageKey) {
          try {
            const item = localStorage.getItem(key);
            const data = JSON.parse(item);
            if (data.timestamp && Date.now() - data.timestamp > this.maxStorageAge) {
              localStorage.removeItem(key);
            }
          } catch (e) {
            // Invalid JSON, remove it
            localStorage.removeItem(key);
          }
        }
      });
    } catch (error) {
      console.warn('Failed to clear old storage:', error);
    }
  }
  
  // Event notification system
  notifyMessageAdded(message) {
    this.dispatchEvent('messageAdded', { message });
  }
  
  notifyMessageUpdated(message, updates) {
    this.dispatchEvent('messageUpdated', { message, updates });
  }
  
  notifyMessageDeleted(message) {
    this.dispatchEvent('messageDeleted', { message });
  }
  
  notifyMessagesCleared(count) {
    this.dispatchEvent('messagesCleared', { count });
  }
  
  dispatchEvent(eventType, data) {
    const event = new CustomEvent(`sphinxAI:${eventType}`, { detail: data });
    window.dispatchEvent(event);
  }
  
  // Utility methods
  generateMessageId() {
    return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  getMessageCount() {
    return this.messages.length;
  }
  
  getUserMessageCount() {
    return this.messages.filter(m => m.sender === 'user').length;
  }
  
  getAIMessageCount() {
    return this.messages.filter(m => m.sender === 'ai').length;
  }
  
  hasMessages() {
    return this.messages.length > 0;
  }
  
  // Statistics
  getConversationStats() {
    const userMessages = this.getUserMessageCount();
    const aiMessages = this.getAIMessageCount();
    const totalMessages = this.getMessageCount();
    
    let firstMessageTime = null;
    let lastMessageTime = null;
    
    if (totalMessages > 0) {
      firstMessageTime = this.messages[0].timestamp;
      lastMessageTime = this.messages[totalMessages - 1].timestamp;
    }
    
    return {
      totalMessages,
      userMessages,
      aiMessages,
      conversationId: this.conversationId,
      firstMessageTime,
      lastMessageTime,
      duration: lastMessageTime && firstMessageTime ? lastMessageTime - firstMessageTime : 0
    };
  }
}

// Export message manager class
window.SphinxMessageManager = SphinxMessageManager; 