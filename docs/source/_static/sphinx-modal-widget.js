// Enhanced Sphinx AI Modal Widget with advanced chat functionality
(function() {
  'use strict';
  
  // Initialize services
  let typesenseClient, messageManager, aiChatService;
  let modeBadge, modePanel;
  let currentMode = 'quick';
  let isInitialized = false;

  // Initialize services when dependencies are loaded
  function initializeServices() {
    if (typeof SphinxAIConfig === 'undefined' ||
        typeof SphinxTypesenseClient === 'undefined' ||
        typeof SphinxMessageManager === 'undefined' ||
        typeof SphinxAIChatService === 'undefined' ||
        typeof SphinxModeBadge === 'undefined' ||
        typeof SphinxModePanel === 'undefined') {
      // Retry after a short delay
      setTimeout(initializeServices, 100);
      return;
    }
    
    try {
      typesenseClient = new SphinxTypesenseClient(SphinxAIConfig.typesense);
      messageManager = new SphinxMessageManager();
      aiChatService = new SphinxAIChatService(typesenseClient, SphinxAIConfig);
      
      // Load previous messages
      messageManager.loadFromStorage();
      
      isInitialized = true;
      
      if (SphinxAIConfig.debug) {
        console.log('Sphinx AI services initialized successfully');
      }
    } catch (error) {
      console.error('Failed to initialize Sphinx AI services:', error);
    }
  }

  function getPageContext() {
    const title = document.title;
    const url = window.location.href;
    // const currentSection = document.querySelector('.current')?.textContent || ''; // Not used - removed section detection
    const content = document.querySelector('[role="main"]')?.textContent?.slice(0, 500) || '';
    
    return {
      title,
      url,
      // section: currentSection, // Removed - not needed
      contentPreview: content
    };
  }

  // Add basic modal styles (detailed styles are in external CSS file)
  function injectModalStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .sphinx-modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5) ;
        z-index: 1001;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s ease, visibility 0.3s ease;
        backdrop-filter: blur(3px);
      }
      
      .sphinx-modal-overlay.show {
        opacity: 1;
        visibility: visible;
      }
      
      .sphinx-modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) scale(0.9);
        background: white;
        border-radius: 12px;
        width: 800px;
        max-width: 95vw;
        max-height: 85vh;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        z-index: 1002;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      
      .sphinx-modal.show {
        opacity: 1;
        visibility: visible;
        transform: translate(-50%, -50%) scale(1);
      }
      
      .sphinx-modal-header {
        background: rgba(171, 71, 188, 0.1);
        color: #673ab7;
        padding: 24px 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(171, 71, 188, 0.2);
      }
      
      .sphinx-modal-header-content {
        display: flex;
        align-items: center;
        gap: 12px;
        flex: 1;
        min-width: 0;
      }
      
      .sphinx-modal-logo {
        width: 24px;
        height: 20px;
        flex-shrink: 0;
      }
      
      .sphinx-modal-logo path {
        fill: #673ab7;
      }
      
      .sphinx-modal-title {
        margin: 0;
        font-size: 20px;
        font-weight: 600;
        color: #000000ff;
      }
      
      .sphinx-modal-close {
        background: transparent;
        border: none;
        color: #673ab7;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        cursor: pointer;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
      }
      
      .sphinx-modal-close:hover {
        background: rgba(171, 71, 188, 0.1);
        color: #673ab7;
        transform: rotate(90deg);
      }
      
      .sphinx-modal-body {
        padding: 0;
        flex: 1;
        overflow: hidden; /* Prevent modal body from scrolling */
        background-color: white;
        display: flex;
        flex-direction: column;
      }
      

      .sphinx-modal-footer {
        padding: 8px 30px;
        background: rgba(171, 71, 188, 0.05);
        border-top: 1px solid #f0e2ebff;
        text-align: center;
        font-size: 13px;
        color: #673ab7;
        line-height: 1.4;
        position: relative;
      }
      
      .sphinx-input-container {
        display: flex;
        gap: 12px;
        align-items: center;
        position: relative;
      }
      
      .sphinx-input-left-controls {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
      }
      
      .sphinx-mode-selector-wrapper {
        position: relative;
        display: flex;
        align-items: center;
      }
      
      .sphinx-input-container textarea {
        flex: 1;
        min-height: 44px;
        max-height: 120px;
        padding: 12px 16px;
        border: 2px solid #673ab7;
        border-radius: 8px;
        resize: none;
        font-family: inherit;
        font-size: 14px;
        line-height: 1.5;
        background: white;
        color: #4c1a52ff;
        transition: all 0.2s ease;
        overflow-y: auto;
      }
      
      .sphinx-input-container textarea:focus {
        outline: none;
        outline-offset: -2px;
        border-color: #673ab7;
        box-shadow: 0 0 0 3px rgba(103, 58, 183, 0.1);
      }
      
      .sphinx-submit-btn {
        padding: 12px 20px;
        background: #673ab7;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 600;
        transition: all 0.2s ease;
        white-space: nowrap;
        min-width: 80px;
        height: 44px;
      }
      
      .sphinx-submit-btn:hover:not(:disabled) {
        background: #512da8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(103, 58, 183, 0.2);
      }
    `;
    document.head.appendChild(style);
  }

  // Enhanced modal creation with AI chat focus
  function createModal() {
    // const base   = window.location.origin + window.location.pathname;
    // const logoUrl = new URL("_static/svg/logo.svg", base).href;
    const urlRoot = (window.DOCUMENTATION_OPTIONS && DOCUMENTATION_OPTIONS.URL_ROOT) || '/';
    const logoUrl = urlRoot + '_static/svg/logo.svg';
    const homeUrl = urlRoot; 

    const modal = document.createElement('div');
    
    modal.innerHTML = `
      <div class="sphinx-modal-overlay">
        <div class="sphinx-modal">
          <div class="sphinx-modal-header">
            <div class="sphinx-modal-header-content">
              <a href="${homeUrl}" class="navbar-brand">
                <img src="${logoUrl}" class="sphinx-modal-logo" alt="logo">
              </a>
              <h3 class="sphinx-modal-title">Your AI Assistant</h3>
            </div>
            <button class="sphinx-modal-close">×</button>
          </div>
          
          <div class="sphinx-modal-body">
            <div class="sphinx-ai-chat">
              <div class="sphinx-chat-messages"></div>
              <div class="sphinx-chat-input">
                <div class="sphinx-input-container">
                  <div class="sphinx-input-left-controls">
                    <div class="sphinx-mode-selector-wrapper">
                      <!-- Mode badge and panel will be inserted here -->
                    </div>
                  </div>
                  <textarea 
                    placeholder="Ask me any question about the RLinf..."
                    rows="1"
                  ></textarea>
                  <button class="sphinx-submit-btn">Ask AI</button>
                </div>
              </div>
            </div>
          </div>
          
        <div class="sphinx-modal-footer">
          💡 Powered by Infini AI Assistant | ⚠️ Content generated by AI — please verify carefully.
        </div>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    return modal;
  }

  function bindNavTrigger() {
    // 模板里按钮的 ID
    const trigger = document.getElementById('ask-ai-trigger');
    if (!trigger) return;
    
    trigger.addEventListener('click', () => {
      const overlay = document.querySelector('.sphinx-modal-overlay');
      const modal   = document.querySelector('.sphinx-modal');
      overlay.classList.add('show');
      modal.classList.add('show');
      // 聚焦输入框
      setTimeout(() => modal.querySelector('textarea')?.focus(), 100);
    });
  }

  // // creater trigger button
  // function createTrigger() {
  //   const trigger = document.createElement('button');
  //   trigger.className = 'sphinx-modal-trigger';
  //   trigger.innerHTML = 'Ask AI';
  //   trigger.style.cssText = `
  //     position: fixed;
  //     bottom: 100px;
  //     right: 100px;
  //     background-color: #673ab7;
  //     color: white;
  //     border: none;
  //     border-radius: 50px;
  //     padding: 12px 20px;
  //     cursor: pointer;
  //     font-family: var(--sphinx-font-sans, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif);
  //     font-size: 14px;
  //     font-weight: 500;
  //     box-shadow: 0 4px 15px rgba(31, 13, 45, 0.7);
  //     z-index: 1000;
  //     transition: all 0.2s ease;
  //     display: flex;
  //     align-items: center;
  //     gap: 8px;
  //   `;
    
  //   // Hover effects
  //   trigger.addEventListener('mouseenter', () => {
  //     trigger.style.transform = 'translateY(-2px)';
  //     trigger.style.boxShadow = '0 6px 20px rgba(50, 0, 65, 0.4)';
  //   });
    
  //   trigger.addEventListener('mouseleave', () => {
  //     trigger.style.transform = 'translateY(0)';
  //     trigger.style.boxShadow = '0 4px 15px rgba(77, 0, 61, 0.3)';
  //   });
    
  //   document.body.appendChild(trigger);
  //   return trigger;
  // }

  // Enhanced chat functionality
  function initChat(modal) {
    const messagesContainer = modal.querySelector('.sphinx-chat-messages');
    const textarea = modal.querySelector('textarea');
    const submitBtn = modal.querySelector('.sphinx-submit-btn');
    const modeSelectorWrapper = modal.querySelector('.sphinx-mode-selector-wrapper');
    
    // Initialize mode selection components
    initModeSelection(modeSelectorWrapper);
    
    // Load existing messages
    loadMessages(messagesContainer);
    
    // Submit message
    submitBtn.addEventListener('click', () => {
      sendMessage(textarea.value.trim(), currentMode, messagesContainer, textarea);
    });
    
    // Enter key handling
    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(textarea.value.trim(), currentMode, messagesContainer, textarea);
      }
    });
    
    // Auto-resize textarea
    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      const minHeight = 44; // Match CSS height
      const maxHeight = 120;
      const newHeight = Math.max(minHeight, Math.min(textarea.scrollHeight, maxHeight));
      textarea.style.height = newHeight + 'px';
    });
    
    // Show empty state if no messages
    if (!messageManager.hasMessages()) {
      showEmptyState(messagesContainer);
    }
  }
  
  // Initialize mode selection components
  function initModeSelection(container) {
    if (!container || !isInitialized) return;
    
    // Create mode panel
    modePanel = new SphinxModePanel(SphinxAIConfig, (selectedMode) => {
      currentMode = selectedMode;
      modeBadge.setMode(selectedMode);
      
      if (SphinxAIConfig.debug) {
        console.log('Chat mode changed to:', selectedMode);
      }
    }, () => {
      // Callback when panel closes - sync badge state
      modeBadge.closePanel();
    });
    
    // Create mode badge
    modeBadge = new SphinxModeBadge(SphinxAIConfig, (isOpen) => {
      if (isOpen) {
        modePanel.show();
      } else {
        modePanel.hide();
      }
    });
    
    // Add components to container
    const badgeElement = modeBadge.create();
    const panelElement = modePanel.create();
    
    container.appendChild(badgeElement);
    container.appendChild(panelElement);
    
    // Set initial mode
    currentMode = SphinxAIConfig.chat.defaultMode || 'quick';
    modeBadge.setMode(currentMode);
    modePanel.setCurrentMode(currentMode);
    
    // Click outside to close panel
    document.addEventListener('click', (e) => {
      if (modePanel && modePanel.isOpen()) {
        const clickedInside = e.target.closest('.sphinx-mode-selector-wrapper');
        if (!clickedInside) {
          modePanel.hide();
          // No need to call modeBadge.closePanel() - it will be called automatically via callback
        }
      }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modePanel && modePanel.isOpen()) {
        modePanel.hide();
        // No need to call modeBadge.closePanel() - it will be called automatically via callback
      }
    });
  }
  
  function showEmptyState(container) {
    container.innerHTML = `
    <div class="sphinx-chat-empty">
      <h3>👋 Hello! I'm your AI Documentation Assistant</h3>
      <p>I can help you understand the content of this document and answer your questions.</p>
      <div class="sphinx-suggested-prompts">
        <div class="sphinx-suggested-prompt" data-prompt="What is RLinf about?">What is RLinf about?</div>
        <div class="sphinx-suggested-prompt" data-prompt="How do I get started quickly?">How do I get started quickly?</div>
        <div class="sphinx-suggested-prompt" data-prompt="Where can I find examples?">Where can I find examples?</div>
        <div class="sphinx-suggested-prompt" data-prompt="Are there any important notes?">Are there any important notes?</div>
      </div>
    </div>
    `;
    
    // Add click handlers for suggested prompts
    container.querySelectorAll('.sphinx-suggested-prompt').forEach(prompt => {
      prompt.addEventListener('click', () => {
        const text = prompt.dataset.prompt;
        const textarea = document.querySelector('.sphinx-ai-chat textarea');
        if (textarea) {
          textarea.value = text;
          textarea.focus();
        }
      });
    });
  }
  
  function loadMessages(container) {
    const messages = messageManager.getAllMessages();
    container.innerHTML = '';
    
    messages.forEach(message => {
      addMessageToUI(container, message);
    });
    
    if (messages.length > 0) {
      scrollToBottom(container);
    }
  }
  
  async function sendMessage(content, mode, container, textarea) {
    if (!content || !isInitialized) {
      if (!content) {
        textarea.focus();
      }
      return;
    }
    
    // Clear textarea
    textarea.value = '';
    textarea.style.height = '44px'; // Reset to minimum height
    
    // Remove empty state if present
    const emptyState = container.querySelector('.sphinx-chat-empty');
    if (emptyState) {
      emptyState.remove();
    }
    
    // Add user message
    const userMessage = messageManager.addMessage('user', content);
    addMessageToUI(container, userMessage);
    
    // Add AI message placeholder
    const aiMessage = messageManager.addMessage('ai', '', { isLoading: true });
    addMessageToUI(container, aiMessage);
    
    scrollToBottom(container);
    
    try {
      // Send to AI service with streaming
      await aiChatService.sendMessageStreaming(
        content,
        messageManager.getConversationId(),
        {
          onChunk: (chunk) => {
            updateAIMessage(aiMessage.id, chunk, container, true);
          },
          onComplete: () => {
            finalizeAIMessage(aiMessage.id, container);
            messageManager.saveToStorage();
          }
        },
        { mode }
      );
      
    } catch (error) {
      const errorMessage = 'Sorry, an error occurred. Please try again.';
      updateAIMessage(aiMessage.id, errorMessage, container, false);
      messageManager.updateMessage(aiMessage.id, { 
        isLoading: false,
        error: error.message 
      });
      
      console.error('Message sending failed:', error);
    }
  }
  
  function addMessageToUI(container, message) {
    const messageEl = document.createElement('div');
    messageEl.className = `sphinx-message ${message.sender}-message`;
    messageEl.dataset.messageId = message.id;
    
    if (message.isLoading) {
      messageEl.classList.add('loading');
    }
    
    const avatar = message.sender === 'user' ? '👤' : '🤖';
    let content;
    
    if (message.isLoading) {
      content = 'thinking...';
    } else if (message.sender === 'ai') {
      // Render markdown for AI messages
      content = renderMarkdown(message.content);
    } else {
      // Escape HTML for user messages to prevent XSS
      content = escapeHtml(message.content);
    }
    
    messageEl.innerHTML = `
      <div class="sphinx-message-avatar">${avatar}</div>
      <div class="sphinx-message-content">
        <div class="sphinx-message-text">${content}</div>
        ${message.sources && message.sources.length > 0 ? createSourcesHTML(message.sources) : ''}
      </div>
    `;
    
    container.appendChild(messageEl);
  }
  
  function updateAIMessage(messageId, content, container, isStreaming = false) {
    const messageEl = container.querySelector(`[data-message-id="${messageId}"]`);
    if (messageEl) {
      const textEl = messageEl.querySelector('.sphinx-message-text');
      
      if (isStreaming) {
        // Clear loading state on first streaming chunk
        if (messageEl.classList.contains('loading')) {
          messageEl.classList.remove('loading');
        }
        
        // Append content for streaming
        const currentContent = messageManager.getMessage(messageId)?.content || '';
        const newContent = currentContent + content;
        // During streaming, render markdown progressively but handle incomplete syntax gracefully
        textEl.innerHTML = renderMarkdown(newContent);
        messageManager.updateMessage(messageId, { content: newContent, isLoading: false });
      } else {
        // Replace content and render markdown for final content
        textEl.innerHTML = renderMarkdown(content);
        messageManager.updateMessage(messageId, { content, isLoading: false });
      }
      
      scrollToBottom(container);
    }
  }
  
  function finalizeAIMessage(messageId, container) {
    const messageEl = container.querySelector(`[data-message-id="${messageId}"]`);
    if (messageEl) {
      messageEl.classList.remove('loading');
      
      // Ensure final content is properly rendered as markdown
      const message = messageManager.getMessage(messageId);
      if (message && message.content) {
        const textEl = messageEl.querySelector('.sphinx-message-text');
        if (textEl) {
          textEl.innerHTML = renderMarkdown(message.content);
        }
      }
    }
    
    messageManager.updateMessage(messageId, { 
      isLoading: false,
      isStreaming: false 
    });
  }
  
  function createSourcesHTML(sources) {
    if (!sources.length) return '';
    
    const sourcesHTML = sources.map(source => `
      <div class="sphinx-source">
        <a href="${source.document.url}" target="_blank">
          ${source.document.title || source.document.hierarchy.lvl0}
        </a>
      </div>
    `).join('');
    
    return `
      <div class="sphinx-sources">
        <h4>reference</h4>
        ${sourcesHTML}
      </div>
    `;
  }
  
  function scrollToBottom(container) {
    container.scrollTop = container.scrollHeight;
  }
  
  // Clear chat history function
  function clearChatHistory(container) {
    if (messageManager) {
      // Clear messages from manager
      messageManager.clearMessages();
      
      // Clear from storage
      messageManager.saveToStorage();
      
      // Clear UI
      if (container) {
        container.innerHTML = '';
        
        // Show empty state
        showEmptyState(container);
      }
      
      if (SphinxAIConfig.debug) {
        console.log('Chat history cleared on modal close');
      }
    }
  }
  
  // Enhanced markdown renderer based on VitePress implementation
  function renderMarkdown(markdown) {
    if (!markdown) return '';
    
    // First, protect code content with placeholders to prevent interference
    let result = markdown;
    const codeBlocks = [];
    const inlineCodes = [];
    
    // Extract and protect code blocks first
    result = result.replace(/```([\s\S]*?)```/g, (match, content) => {
      const index = codeBlocks.length;
      codeBlocks.push(`<pre><code>${escapeHtml(content.trim())}</code></pre>`);
      return `XCODEBLOCKREPLACEMENTX${index}XCODEBLOCKREPLACEMENTX`;
    });
    
    // Extract and protect inline code
    result = result.replace(/`([^`]+)`/g, (match, content) => {
      const index = inlineCodes.length;
      inlineCodes.push(`<code>${escapeHtml(content)}</code>`);
      return `XINLINECODEREPLACEMENTX${index}XINLINECODEREPLACEMENTX`;
    });
    
    // Now apply other formatting (safe from code interference)
    result = result
      // Headers with proper hierarchy
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      
      // Bold
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/__(.*?)__/g, '<strong>$1</strong>')
      
      // Italic
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/_(.*?)_/g, '<em>$1</em>')
      
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
      
      // Normalize multiple line breaks
      .replace(/\n\n\n+/g, '\n\n')
      // Handle paragraphs and line breaks
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>');
    
    // Wrap content in paragraphs
    if (result && !result.includes('<h1>') && !result.includes('<h2>') && !result.includes('<h3>') && 
        !result.includes('<pre>')) {
      result = '<p>' + result + '</p>';
    }
    
    // Restore protected code blocks
    codeBlocks.forEach((codeBlock, index) => {
      result = result.replace(`XCODEBLOCKREPLACEMENTX${index}XCODEBLOCKREPLACEMENTX`, codeBlock);
    });
    
    // Restore protected inline code
    inlineCodes.forEach((inlineCode, index) => {
      result = result.replace(`XINLINECODEREPLACEMENTX${index}XINLINECODEREPLACEMENTX`, inlineCode);
    });
    
    return result;
  }
  
  // Safely escape HTML for user content
  function escapeHtml(text) {
    if (!text) return '';
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Initialize when DOM is ready
  function init() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }
    
    // Initialize services first
    initializeServices();
    
    // Inject basic modal styles
    injectModalStyles();
    
    // Create UI components
    // const trigger = createTrigger();
    const modal = createModal();

    bindNavTrigger();
    
    // Initialize chat functionality
    initChat(modal);
    
    // 3) 关闭逻辑
    const overlay = modal.querySelector('.sphinx-modal-overlay');
    const modalBox = modal.querySelector('.sphinx-modal');
    const closeBtn = modal.querySelector('.sphinx-modal-close');

    function handleModalClose() {
      overlay.classList.remove('show');
      modalBox.classList.remove('show');
      const messagesContainer = modal.querySelector('.sphinx-chat-messages');
      clearChatHistory(messagesContainer);
    }

    closeBtn.addEventListener('click', handleModalClose);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) handleModalClose();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && overlay.classList.contains('show')) {
        handleModalClose();
      }
    });
    
    console.log('Enhanced Sphinx AI Modal Widget loaded');
  }

  init();
})();
