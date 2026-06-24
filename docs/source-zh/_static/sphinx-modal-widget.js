// Enhanced Sphinx AI Modal Widget with advanced chat functionality
(function() {
  'use strict';
  
  // Initialize services
  let typesenseClient, messageManager, aiChatService;
  let modeBadge, modePanel;
  let currentMode = 'quick';
  let isInitialized = false;
  let requestControl = null;
  let isChatBusy = false;
  let lastFocusedTrigger = null;
  
  // IME state variables
  let isComposing = false;
  let justFinishedComposition = false;
  const JUST_FINISHED_MS = 80;

  // Initialize services when dependencies are loaded
  function initializeServices() {
    if (typeof SphinxAIConfig === 'undefined' ||
        typeof SphinxTypesenseClient === 'undefined' ||
        typeof SphinxMessageManager === 'undefined' ||
        typeof SphinxAIChatService === 'undefined' ||
        typeof SphinxModeBadge === 'undefined' ||
        typeof SphinxModePanel === 'undefined' ||
        typeof RLinfAssistantUtils === 'undefined') {
      // Retry after a short delay
      setTimeout(initializeServices, 100);
      return;
    }
    
    try {
      typesenseClient = new SphinxTypesenseClient(SphinxAIConfig.typesense);
      messageManager = new SphinxMessageManager();
      aiChatService = new SphinxAIChatService(typesenseClient, SphinxAIConfig);
      requestControl = RLinfAssistantUtils.createSearchRequestControl();
      
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

  function createModal() {
    const modal = document.createElement('div');

    modal.innerHTML = `
      <div class="sphinx-modal-overlay">
        <div class="sphinx-modal" role="dialog" aria-modal="true" aria-labelledby="sphinx-modal-title" tabindex="-1">
          <div class="sphinx-modal-header">
            <div class="sphinx-modal-header-content">
              <h3 class="sphinx-modal-title" id="sphinx-modal-title">RLinf AI Assistant</h3>
            </div>
            <div class="sphinx-modal-actions">
              <button class="sphinx-modal-reset" type="button" aria-label="Start a new chat">New chat</button>
              <button class="sphinx-modal-close" type="button" aria-label="Close">&times;</button>
            </div>
          </div>
          
          <div class="sphinx-modal-body">
            <div class="sphinx-ai-chat">
              <div class="sphinx-chat-messages" aria-live="polite" aria-relevant="additions text"></div>
              <div class="sphinx-chat-input">
                <div class="sphinx-input-container">
                  <div class="sphinx-input-left-controls">
                    <div class="sphinx-mode-selector-wrapper">
                      <!-- Mode badge and panel will be inserted here -->
                    </div>
                  </div>
                  <textarea 
                    placeholder="Ask RLinf docs..."
                    rows="1"
                  ></textarea>
                  <button class="sphinx-submit-btn" type="button">Ask AI</button>
                </div>
              </div>
            </div>
          </div>
          
          <div class="sphinx-modal-footer">AI-generated. Verify with cited sources.</div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    return modal;
  }

  function getModalElements(modalRoot) {
    return {
      overlay: modalRoot?.querySelector('.sphinx-modal-overlay'),
      modalBox: modalRoot?.querySelector('.sphinx-modal')
    };
  }

  function openModal(modalRoot, trigger = null) {
    const { overlay, modalBox } = getModalElements(modalRoot);
    if (!overlay || !modalBox) return;

    lastFocusedTrigger = trigger || document.activeElement;
    document.body.classList.add('rlinf-ai-modal-open');
    overlay.classList.add('show');
    modalBox.classList.add('show');
    setTimeout(() => modalBox.querySelector('textarea')?.focus(), 80);
  }

  function closeModal(modalRoot) {
    const { overlay, modalBox } = getModalElements(modalRoot);
    if (!overlay || !modalBox) return;

    overlay.classList.remove('show');
    modalBox.classList.remove('show');
    document.body.classList.remove('rlinf-ai-modal-open');

    if (
      lastFocusedTrigger &&
      typeof lastFocusedTrigger.focus === 'function' &&
      document.contains(lastFocusedTrigger)
    ) {
      lastFocusedTrigger.focus({ preventScroll: true });
    }
    lastFocusedTrigger = null;
  }

  function trapModalFocus(event, modalBox) {
    if (event.key !== 'Tab' || !modalBox?.classList.contains('show')) return;

    const focusable = Array.from(modalBox.querySelectorAll([
      'a[href]',
      'button:not([disabled])',
      'textarea:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      '[tabindex]:not([tabindex="-1"])'
    ].join(','))).filter(element => element.offsetParent !== null || element === document.activeElement);

    if (!focusable.length) {
      event.preventDefault();
      modalBox.focus({ preventScroll: true });
      return;
    }

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function bindNavTrigger(modalRoot) {
    // 模板里按钮的 ID
    const trigger = document.getElementById('ask-ai-trigger');
    if (!trigger) return;
    
    trigger.addEventListener('click', () => {
      openModal(modalRoot, trigger);
    });
  }

  // Enhanced chat functionality
  function initChat(modal) {
    const messagesContainer = modal.querySelector('.sphinx-chat-messages');
    const textarea = modal.querySelector('textarea');
    const submitBtn = modal.querySelector('.sphinx-submit-btn');
    const resetBtn = modal.querySelector('.sphinx-modal-reset');
    const modeSelectorWrapper = modal.querySelector('.sphinx-mode-selector-wrapper');
    
    // Initialize mode selection components
    initModeSelection(modeSelectorWrapper);
    
    // Load existing messages
    loadMessages(messagesContainer);
    
    // Add IME composition listeners
    textarea.addEventListener('compositionstart', () => {
      isComposing = true;
    });
    textarea.addEventListener('compositionend', () => {
      isComposing = false;
      justFinishedComposition = true;
      setTimeout(() => { justFinishedComposition = false; }, JUST_FINISHED_MS);
    });
    
    // Optional: beforeinput for enhanced IME detection
    textarea.addEventListener('beforeinput', (e) => {
      const t = e.inputType || '';
      if (t === 'insertCompositionText') {
        isComposing = true;
      } else if (t === 'insertFromComposition' || t === 'deleteCompositionText') {
        justFinishedComposition = true;
        setTimeout(() => { justFinishedComposition = false; }, JUST_FINISHED_MS);
      }
    });
    
    // Submit message
    submitBtn.addEventListener('click', () => {
      if (isComposing || justFinishedComposition) {
        textarea.focus();
        return;
      }
      const content = textarea.value.trim();
      if (!content) {
        textarea.focus();
        return;
      }
      sendMessage(content, currentMode, messagesContainer, textarea);
    });

    resetBtn?.addEventListener('click', () => {
      requestControl?.dispose();
      setChatBusy(false, modal.querySelector('.sphinx-modal'));
      clearChatHistory(messagesContainer);
      textarea.value = '';
      textarea.style.height = '';
      textarea.focus();
    });
    
    // Enter key handling
    textarea.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      if (e.shiftKey) return; // Shift+Enter for newline
      if (isComposing || e.isComposing || justFinishedComposition) {
        return; // Let IME handle Enter
      }
      e.preventDefault();
      const content = textarea.value.trim();
      if (!content) return;
      sendMessage(content, currentMode, messagesContainer, textarea);
    });
    
    // Auto-resize textarea
    const autoResize = () => {
      requestAnimationFrame(() => {
        const minHeight = Number.parseFloat(window.getComputedStyle(textarea).minHeight) || 40;
        const maxHeight = 120;
        const prevOverflow = textarea.style.overflowY;
        textarea.style.overflowY = 'hidden';
        textarea.style.height = 'auto';
        const newHeight = Math.max(minHeight, Math.min(textarea.scrollHeight, maxHeight));
        textarea.style.height = newHeight + 'px';
        textarea.style.overflowY = prevOverflow || 'auto';
      });
    };
    textarea.addEventListener('input', autoResize);
    
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
      <h3>Ask RLinf docs</h3>
      <p>Answers are grounded in indexed documentation and cited sources.</p>
      <div class="sphinx-suggested-prompts">
        <button class="sphinx-suggested-prompt" type="button" data-prompt="What is RLinf about?">What is RLinf about?</button>
        <button class="sphinx-suggested-prompt" type="button" data-prompt="How do I get started quickly?">How do I get started quickly?</button>
        <button class="sphinx-suggested-prompt" type="button" data-prompt="Where can I find examples?">Where can I find examples?</button>
        <button class="sphinx-suggested-prompt" type="button" data-prompt="How does RLinf run distributed training?">How does distributed training work?</button>
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
          textarea.dispatchEvent(new Event('input', { bubbles: true }));
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
  
  function setChatBusy(isBusy, modalRoot) {
    isChatBusy = isBusy;
    const submitBtn = modalRoot?.querySelector('.sphinx-submit-btn') || document.querySelector('.sphinx-submit-btn');
    const input = modalRoot?.querySelector('textarea') || document.querySelector('.sphinx-ai-chat textarea');
    if (submitBtn) {
      submitBtn.disabled = isBusy;
      submitBtn.textContent = isBusy ? 'Asking' : 'Ask AI';
    }
    if (input) {
      input.disabled = isBusy;
    }
  }

  async function sendMessage(content, mode, container, textarea) {
    if (!content || !isInitialized || isChatBusy) {
      if (!content) {
        textarea.focus();
      }
      return;
    }

    const queryPlan = RLinfAssistantUtils.createQueryPlan({ message: content });
    const conversationId = queryPlan.resetConversation ? null : messageManager.getConversationId();
    if (queryPlan.resetConversation) {
      messageManager.setConversationId(null);
    }
    
    // Clear textarea
    textarea.value = '';
    textarea.style.height = '';
    
    // Remove empty state if present
    const emptyState = container.querySelector('.sphinx-chat-empty');
    if (emptyState) {
      emptyState.remove();
    }
    
    // Add user message
    const userMessage = messageManager.addMessage('user', content);
    addMessageToUI(container, userMessage);
    
    // Add AI message placeholder
    const aiMessage = messageManager.addMessage('ai', '', {
      isLoading: true,
      queryPlan
    });
    addMessageToUI(container, aiMessage);
    
    scrollToBottom(container);
    setChatBusy(true, container.closest('.sphinx-modal'));
    
    try {
      let activeRequestId = null;
      const result = await requestControl.runNow(async ({ requestId }) => {
        activeRequestId = requestId;
        return await aiChatService.sendMessageStreaming(
          content,
          conversationId,
          {
            onChunk: (chunk) => {
              if (!requestControl.isLatest(requestId)) return;
              updateAIMessage(aiMessage.id, chunk, container, true);
            },
            onComplete: (sources, response = {}) => {
              if (!requestControl.isLatest(requestId)) return;
              finalizeAIMessage(aiMessage.id, container, sources, response.queryPlan || queryPlan);
              if (response.conversationId) {
                messageManager.setConversationId(response.conversationId);
              }
              messageManager.saveToStorage();
            },
            onError: (error) => {
              if (!requestControl.isLatest(requestId)) return;
              updateAIMessage(aiMessage.id, String(error || 'Request failed'), container, false);
            }
          },
          { mode, queryPlan }
        );
      });

      if (activeRequestId && requestControl.isLatest(activeRequestId) && result?.conversationId) {
        messageManager.setConversationId(result.conversationId);
      }
      
    } catch (error) {
      const errorMessage = 'Sorry, an error occurred. Please try again.';
      updateAIMessage(aiMessage.id, errorMessage, container, false);
      messageManager.updateMessage(aiMessage.id, { 
        isLoading: false,
        error: error.message 
      });
      
      console.error('Message sending failed:', error);
    } finally {
      setChatBusy(false, container.closest('.sphinx-modal'));
      textarea.focus();
    }
  }
  
  function addMessageToUI(container, message) {
    const messageEl = document.createElement('div');
    messageEl.className = `sphinx-message ${message.sender}-message`;
    messageEl.dataset.messageId = message.id;
    
    if (message.isLoading) {
      messageEl.classList.add('loading');
    }
    
    const messageHeader = message.sender === 'ai'
      ? `
        <div class="sphinx-message-header">
          <span class="sphinx-message-label">Answer</span>
          ${message.queryPlan ? createQueryPlanHTML(message.queryPlan) : ''}
        </div>
      `
      : '';
    let content;
    
    if (message.isLoading) {
      content = 'Searching docs...';
    } else if (message.sender === 'ai') {
      // Render markdown for AI messages
      content = renderMarkdown(message.content);
    } else {
      // Escape HTML for user messages to prevent XSS
      content = escapeHtml(message.content);
    }
    
    messageEl.innerHTML = `
      <div class="sphinx-message-content">
        ${messageHeader}
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
  
  function finalizeAIMessage(messageId, container, sources = [], queryPlan = null) {
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

      const oldSources = messageEl.querySelector('.sphinx-sources');
      if (oldSources) oldSources.remove();
      if (sources && sources.length) {
        messageEl.querySelector('.sphinx-message-content')?.insertAdjacentHTML('beforeend', createSourcesHTML(sources));
      }
    }
    
    messageManager.updateMessage(messageId, { 
      isLoading: false,
      isStreaming: false,
      sources,
      queryPlan: queryPlan || undefined
    });
  }
  
  function createSourcesHTML(sources) {
    return RLinfAssistantUtils.createSourcesHTML(sources);
  }

  function createQueryPlanHTML(plan) {
    const display = plan.display || {};
    const label = [
      display.sourceScope || 'Documentation',
      plan.analysis?.queryType || ''
    ].filter(Boolean).join(' · ');

    return `
      <div class="sphinx-query-plan" title="Search plan">
        ${RLinfAssistantUtils.escapeHtml(label)}
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
    return RLinfAssistantUtils.renderMarkdown(markdown);
  }
  
  // Safely escape HTML for user content
  function escapeHtml(text) {
    return RLinfAssistantUtils.escapeHtml(text);
  }

  // Initialize when DOM is ready
  function init() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }
    
    // Initialize services first
    initializeServices();
    
    // Create UI components
    // const trigger = createTrigger();
    const modal = createModal();

    bindNavTrigger(modal);
    
    // Initialize chat functionality
    initChat(modal);
    
    // 3) 关闭逻辑
    const overlay = modal.querySelector('.sphinx-modal-overlay');
    const modalBox = modal.querySelector('.sphinx-modal');
    const closeBtn = modal.querySelector('.sphinx-modal-close');

    function handleModalClose() {
      closeModal(modal);
    }

    closeBtn.addEventListener('click', handleModalClose);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        if (isComposing || justFinishedComposition) {
          const textarea = modal.querySelector('textarea');
          textarea.focus();
          return;
        }
        handleModalClose();
      }
    });
    document.addEventListener('keydown', (e) => {
      trapModalFocus(e, modalBox);

      if (e.key === 'Escape' && overlay.classList.contains('show')) {
        if (isComposing || justFinishedComposition || document.activeElement === modal.querySelector('textarea')) {
          return;
        }
        handleModalClose();
      }
    });
    
    console.log('Enhanced Sphinx AI Modal Widget loaded');
  }

  init();
})();
