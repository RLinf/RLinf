// Mode Panel Component for Sphinx AI Widget
class SphinxModePanel {
  constructor(config, onModeSelect, onPanelClose) {
    this.config = config;
    this.onModeSelect = onModeSelect;
    this.onPanelClose = onPanelClose;
    this.currentMode = config.chat.defaultMode || 'quick';
    this.element = null;
    this.isVisible = false;
  }
  
  create() {
    const panel = document.createElement('div');
    panel.className = 'sphinx-mode-panel';
    panel.innerHTML = this.renderPanelContent();
    panel.style.display = 'none';
    
    // Add event listeners for mode selection
    this.addEventListeners(panel);
    
    this.element = panel;
    return panel;
  }
  
  renderPanelContent() {
    const modes = this.config.chat.modes;
    return `
      <div class="sphinx-mode-panel-content">
        ${Object.entries(modes).map(([key, mode]) => this.renderModeOption(key, mode)).join('')}
      </div>
    `;
  }
  
  renderModeOption(key, mode) {
    const isSelected = key === this.currentMode;
    return `
      <button class="sphinx-mode-option ${isSelected ? 'sphinx-mode-option-active' : ''}" data-mode="${key}">
        <div class="sphinx-mode-option-content">
          <div class="sphinx-mode-icon">
            ${this.renderIcon(mode)}
          </div>
          <div class="sphinx-mode-info">
            <div class="sphinx-mode-header">
              <div class="sphinx-mode-label">${mode.label}</div>
              <div class="sphinx-mode-time">${mode.time}</div>
            </div>
            <div class="sphinx-mode-description">${mode.description}</div>
          </div>
        </div>
        <div class="sphinx-mode-radio ${isSelected ? 'sphinx-mode-radio-selected' : ''}">
          ${isSelected ? '<div class="sphinx-mode-radio-dot"></div>' : ''}
        </div>
      </button>
    `;
  }
  
  renderIcon(mode) {
    return `
      <svg viewBox="0 0 20 20" fill="none" class="sphinx-mode-icon-svg">
        <path d="${mode.iconPath}" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    `;
  }
  
  addEventListeners(panel) {
    // Mode option click handlers
    panel.addEventListener('click', (e) => {
      const modeOption = e.target.closest('.sphinx-mode-option');
      if (modeOption) {
        const selectedMode = modeOption.dataset.mode;
        this.selectMode(selectedMode);
      }
    });
    
    // Prevent panel from closing when clicking inside
    panel.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  }
  
  selectMode(mode) {
    if (this.config.chat.modes[mode] && mode !== this.currentMode) {
      this.currentMode = mode;
      this.updateSelection();
      
      if (this.onModeSelect) {
        this.onModeSelect(mode);
      }
      
      // Hide panel after selection
      this.hide();
    }
  }
  
  updateSelection() {
    if (!this.element) return;
    
    // Remove previous selection
    this.element.querySelectorAll('.sphinx-mode-option').forEach(option => {
      option.classList.remove('sphinx-mode-option-active');
      const radio = option.querySelector('.sphinx-mode-radio');
      radio.classList.remove('sphinx-mode-radio-selected');
      radio.innerHTML = '';
    });
    
    // Add current selection
    const selectedOption = this.element.querySelector(`[data-mode="${this.currentMode}"]`);
    if (selectedOption) {
      selectedOption.classList.add('sphinx-mode-option-active');
      const radio = selectedOption.querySelector('.sphinx-mode-radio');
      radio.classList.add('sphinx-mode-radio-selected');
      radio.innerHTML = '<div class="sphinx-mode-radio-dot"></div>';
    }
  }
  
  show() {
    if (this.element && !this.isVisible) {
      this.element.style.display = 'block';
      this.isVisible = true;
      
      // Trigger animation
      requestAnimationFrame(() => {
        this.element.classList.add('sphinx-mode-panel-visible');
      });
      
      // Update selection state
      this.updateSelection();
    }
  }
  
  hide() {
    if (this.element && this.isVisible) {
      this.element.classList.remove('sphinx-mode-panel-visible');
      this.isVisible = false;
      
      // Notify badge that panel is closing
      if (this.onPanelClose) {
        this.onPanelClose();
      }
      
      // Hide after animation
      setTimeout(() => {
        if (this.element && !this.isVisible) {
          this.element.style.display = 'none';
        }
      }, 200);
    }
  }
  
  toggle() {
    if (this.isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }
  
  setCurrentMode(mode) {
    this.currentMode = mode;
    this.updateSelection();
  }
  
  getCurrentMode() {
    return this.currentMode;
  }
  
  isOpen() {
    return this.isVisible;
  }
  
  destroy() {
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
    this.element = null;
    this.isVisible = false;
  }
}

// Export for use in other modules
window.SphinxModePanel = SphinxModePanel; 