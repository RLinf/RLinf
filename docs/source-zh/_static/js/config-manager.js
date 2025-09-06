// Configuration management for Sphinx AI Widget
let SphinxAIConfig = null;

// Function to initialize and get configuration
function getSphinxAIConfig() {
  if (SphinxAIConfig === null) {
    // Initialize configuration from window.SPHINX_AI_CONFIG
    SphinxAIConfig = {
      // Typesense configuration
      typesense: {
        host: window.SPHINX_AI_CONFIG?.typesense?.host || 'localhost',
        port: window.SPHINX_AI_CONFIG?.typesense?.port || 8108,
        protocol: window.SPHINX_AI_CONFIG?.typesense?.protocol || 'http',
        apiKey: window.SPHINX_AI_CONFIG?.typesense?.apiKey || '',
        collectionName: window.SPHINX_AI_CONFIG?.typesense?.collectionName || 'sphinx_docs',
        connectionTimeoutSeconds: 60
      },
      
      // Chat configuration
      chat: {
        maxMessages: 50,
        enableTypesenseStreaming: false, // Controls Typesense streaming behavior - disabled by default
        defaultMode: 'quick',
        modes: {
          quick: {
            label: 'Quick Answer',
            description: 'Everyday Q&A',
            time: '2-3s',
            model: 'rlinf-quick-response-model',
            timeout: 30000,
            icon: 'lightning',
            iconPath: 'M11.0015 1.00015V8.00015H17.0015L9.00146 19.0002V12.0002H3.00146L11.0015 1.00015Z'
          },
          balance: {
            label: 'Balanced',
            description: 'Balance speed and depth',
            time: '~30s',
            model: 'rlinf-balanced-model',
            timeout: 60000,
            icon: 'balance',
            iconPath: 'M10 2v16M10 6l-6 2v2l6-2 6 2V8l-6-2z'
          },
          deep: {
            label: 'Deep Research',
            description: 'For complex problem analysis',
            time: '~120s',
            model: 'rlinf-deep-analysis-model',
            timeout: 120000,
            icon: 'brain',
            iconPath: 'M14.158 18.8845C12.0209 19.8509 10.214 18.5948 10.214 18.5948C10.214 18.5948 8.71062 17.7382 8.67936 15.8744C8.65786 14.6039 9.5828 14.2352 10.214 14.2452C10.7859 14.254 11.591 14.4521 11.591 15.9434C11.591 15.9434 11.6763 17.6961 9.59713 18.7051C7.83323 19.5611 6.11754 19.1435 5.62967 18.7158C5.03041 18.2461 4.4038 17.4691 4.11263 16.3842C3.53031 14.2139 4.95615 13.1208 4.95615 13.1208C6.12079 12.1381 6.80603 12.5897 7.19034 12.9985C7.61372 13.4488 7.9355 14.3618 6.92914 15.1834C5.45249 16.3886 3.79281 15.865 3.79281 15.865C3.79281 15.865 1.93382 15.4543 1.13785 13.6633L1.12808 13.667C0.124324 11.6095 1.42901 9.86992 1.42901 9.86992C1.42901 9.86992 2.31877 8.42257 4.25463 8.39247C5.5743 8.37178 5.9573 9.26226 5.94688 9.86992C5.93776 10.4205 5.73193 11.1956 4.18298 11.1956C4.18298 11.1956 2.36242 11.2778 1.31437 9.27606C0.846688 8.38306 0.825844 7.50136 0.966539 6.79963C1.61986 4.61544 3.79281 4.13508 3.79281 4.13508C3.79281 4.13508 5.45249 3.61145 6.92914 4.81674C7.9355 5.63824 7.61372 6.5513 7.19034 7.00156C6.80603 7.4098 6.12079 7.86194 4.95615 6.87927C4.95615 6.87927 3.53097 5.78624 4.11263 3.61584C4.4038 2.53033 5.03041 1.75397 5.62967 1.28428C6.3846 0.749985 7.83323 0.438943 9.59713 1.29494C11.6763 2.30394 11.591 4.05669 11.591 4.05669C11.591 5.54794 10.7865 5.7461 10.214 5.75488C9.5828 5.76491 8.65786 5.39618 8.67936 4.12567C8.71062 2.26193 10.214 1.40531 10.214 1.40531C10.214 1.40531 12.0209 0.148595 14.158 1.11559C14.158 1.11559 15.621 1.78784 15.8971 3.61647C16.2319 5.83452 15.0536 6.8799 15.0536 6.8799C13.889 7.86257 13.2037 7.41106 12.8194 7.00219C12.3961 6.55193 12.0743 5.63887 13.0806 4.81736C14.5573 3.61208 16.217 4.13571 16.217 4.13571C16.217 4.13571 18.3893 4.61606 19.0432 6.80026L19.0361 6.80151C19.1768 7.50387 19.1462 8.38369 18.6785 9.27668C17.6304 11.2784 15.8099 11.1962 15.8099 11.1962C14.2609 11.1962 14.0551 10.4218 14.046 9.87055C14.0355 9.26289 14.4185 8.3724 15.7382 8.3931C17.6741 8.4232 18.5638 9.87055 18.5638 9.87055C18.5638 9.87055 19.8737 11.6126 18.87 13.6695C18.0727 15.4555 16.2176 15.865 16.2176 15.865C16.2176 15.865 14.5579 16.3886 13.0813 15.1834C12.0749 14.3618 12.3967 13.4488 12.8201 12.9985C13.2044 12.5903 13.8896 12.1381 15.0543 13.1208C15.0543 13.1208 16.4795 14.2139 15.8978 16.3842C15.8978 16.3842 15.6503 17.9727 14.1586 18.8851L14.158 18.8845Z'
          }
        }
      },
      
      // UI configuration
      ui: {
        theme: 'auto', // 'auto', 'light', 'dark'
        language: 'en', // 'zh', 'en'
        enableAnimations: true,
        enableKeyboardShortcuts: true
      },
      
      // Debug configuration
      debug: window.SPHINX_AI_CONFIG?.debug || false
    };
  }
  
  return SphinxAIConfig;
}

// Configuration validation
function validateConfig() {
  const config = getSphinxAIConfig();
  const errors = [];
  const warnings = [];
  
  // Check Typesense configuration
  if (!config.typesense.apiKey || config.typesense.apiKey.trim() === '') {
    warnings.push('Typesense API key is not configured - AI chat will use mock responses');
  }
  
  if (!config.typesense.host || config.typesense.host.trim() === '') {
    errors.push('Typesense host is required');
  }
  
  if (!config.typesense.collectionName || config.typesense.collectionName.trim() === '') {
    errors.push('Typesense collection name is required');
  }
  
  // Check port number
  if (isNaN(config.typesense.port) || config.typesense.port <= 0) {
    errors.push('Invalid Typesense port number');
  }
  
  // Check protocol
  if (!['http', 'https'].includes(config.typesense.protocol)) {
    errors.push('Typesense protocol must be http or https');
  }
  
  // Check chat mode timeouts
  Object.entries(config.chat.modes).forEach(([mode, modeConfig]) => {
    if (modeConfig.timeout < 5000) {
      warnings.push(`Chat mode "${mode}" timeout is very low, may cause failures`);
    }
  });
  
  if (errors.length > 0) {
    console.error('Sphinx AI Configuration Errors:', errors);
    return { valid: false, errors, warnings };
  }
  
  if (warnings.length > 0) {
    console.warn('Sphinx AI Configuration Warnings:', warnings);
  }
  
  if (config.debug) {
    console.log('Sphinx AI Configuration:', config);
  }
  
  return { valid: true, errors: [], warnings };
}

// Environment detection
function detectEnvironment() {
  const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const isFileProtocol = window.location.protocol === 'file:';
  const isDevelopment = isLocalhost || isFileProtocol;
  
  return {
    isDevelopment,
    isProduction: !isDevelopment,
    protocol: window.location.protocol,
    hostname: window.location.hostname,
    isFileProtocol
  };
}

// Configuration utilities
const SphinxAIConfigUtils = {
  // Get current chat mode configuration
  getCurrentMode() {
    const config = getSphinxAIConfig();
    const mode = config.chat.defaultMode;
    return config.chat.modes[mode] || config.chat.modes.quick;
  },
  
  // Get mode configuration by name
  getMode(modeName) {
    const config = getSphinxAIConfig();
    return config.chat.modes[modeName] || config.chat.modes.quick;
  },
  
  // Check if feature is enabled
  isFeatureEnabled(feature) {
    const config = getSphinxAIConfig();
    const featureMap = {
      streaming: config.chat.enableTypesenseStreaming, // Streaming is controlled by Typesense streaming setting
      typesenseStreaming: config.chat.enableTypesenseStreaming,
      animations: config.ui.enableAnimations,
      shortcuts: config.ui.enableKeyboardShortcuts,
      typesense: !!(config.typesense.apiKey && config.typesense.apiKey.trim())
    };
    
    return featureMap[feature] || false;
  },
  
  // Get theme preference
  getTheme() {
    const config = getSphinxAIConfig();
    if (config.ui.theme === 'auto') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return config.ui.theme;
  },
  
  // Toggle Typesense streaming (for future use)
  toggleTypesenseStreaming() {
    const config = getSphinxAIConfig();
    config.chat.enableTypesenseStreaming = !config.chat.enableTypesenseStreaming;
    
    if (config.debug) {
      console.log(`Typesense streaming toggled: ${config.chat.enableTypesenseStreaming}`);
    }
    
    return config.chat.enableTypesenseStreaming;
  },
  
  // Enable Typesense streaming (for future use)
  enableTypesenseStreaming() {
    const config = getSphinxAIConfig();
    config.chat.enableTypesenseStreaming = true;
    
    if (config.debug) {
      console.log('Typesense streaming enabled');
    }
    
    return true;
  },
  
  // Disable Typesense streaming
  disableTypesenseStreaming() {
    const config = getSphinxAIConfig();
    config.chat.enableTypesenseStreaming = false;
    
    if (config.debug) {
      console.log('Typesense streaming disabled');
    }
    
    return false;
  },
  
  // Update configuration at runtime
  updateConfig(path, value) {
    const config = getSphinxAIConfig();
    const keys = path.split('.');
    let current = config;
    
    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }
    
    current[keys[keys.length - 1]] = value;
    
    if (config.debug) {
      console.log(`Configuration updated: ${path} = ${value}`);
    }
  }
};

// Export configuration and utilities
Object.defineProperty(window, 'SphinxAIConfig', {
  get: getSphinxAIConfig
});
window.SphinxAIConfigUtils = SphinxAIConfigUtils;
window.validateSphinxAIConfig = validateConfig;
window.detectSphinxEnvironment = detectEnvironment;

// Auto-validate configuration on load
document.addEventListener('DOMContentLoaded', () => {
  const validation = validateConfig();
  const environment = detectEnvironment();
  const config = getSphinxAIConfig();
  
  if (config.debug) {
    console.log('Sphinx AI Environment:', environment);
    console.log('Sphinx AI Configuration Validation:', validation);
  }
}); 