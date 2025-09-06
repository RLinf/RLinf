// Mode Badge Component for Sphinx AI Widget
class SphinxModeBadge {
  constructor(config, onTogglePanel) {
    this.config = config;
    this.currentMode = config.chat.defaultMode || 'quick';
    this.onTogglePanel = onTogglePanel;
    this.isOpen = false;
    this.element = null;
  }
  
  create() {
    const badge = document.createElement('button');
    badge.className = 'sphinx-mode-badge';
    badge.type = 'button';
    badge.title = this.getCurrentModeDescription();
    badge.innerHTML = this.renderBadgeContent();
    
    badge.addEventListener('click', () => this.togglePanel());
    this.element = badge;
    return badge;
  }
  
  renderBadgeContent() {
    const mode = this.config.chat.modes[this.currentMode];
    return `
      <div class="sphinx-mode-badge-icon">
        ${this.renderIcon(mode)}
      </div>
      <span class="sphinx-mode-badge-text">${mode.label}</span>
      <svg viewBox="0 0 24 24" fill="none" class="sphinx-mode-chevron">
        <path d="M8.46967 4.21967C8.17678 4.51256 8.17678 4.98744 8.46967 5.28033L15.1893 12L8.46967 18.7197C8.17678 19.0126 8.17678 19.4874 8.46967 19.7803C8.76256 20.0732 9.23744 20.0732 9.53033 19.7803L16.7803 12.5303C17.0732 12.2374 17.0732 11.7626 16.7803 11.4697L9.53033 4.21967C9.23744 3.92678 8.76256 3.92678 8.46967 4.21967Z" fill="currentColor"/>
      </svg>
    `;
  }
  
  renderIcon(mode) {
    return `
      <svg viewBox="0 0 20 20" fill="none" class="sphinx-mode-icon-svg">
        <path d="${mode.iconPath}" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    `;
  }
  
  togglePanel() {
    this.isOpen = !this.isOpen;
    this.element.classList.toggle('sphinx-mode-badge-open', this.isOpen);
    
    if (this.onTogglePanel) {
      this.onTogglePanel(this.isOpen);
    }
  }
  
  setMode(mode) {
    if (this.config.chat.modes[mode]) {
      this.currentMode = mode;
      this.updateDisplay();
    }
  }
  
  getCurrentMode() {
    return this.currentMode;
  }
  
  getCurrentModeDescription() {
    const mode = this.config.chat.modes[this.currentMode];
    return `${mode.label} - ${mode.description} (${mode.time})`;
  }
  
  updateDisplay() {
    if (this.element) {
      this.element.innerHTML = this.renderBadgeContent();
      this.element.title = this.getCurrentModeDescription();
    }
  }
  
  closePanel() {
    this.isOpen = false;
    if (this.element) {
      this.element.classList.remove('sphinx-mode-badge-open');
    }
  }
  
  destroy() {
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
    this.element = null;
  }
}

// Export for use in other modules
window.SphinxModeBadge = SphinxModeBadge; 