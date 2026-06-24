import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

const syncedFiles = [
  '_static/js/assistant-utils.js',
  '_static/js/config-manager.js',
  '_static/js/typesense-client.js',
  '_static/js/message-manager.js',
  '_static/js/ai-chat-service.js',
  '_static/sphinx-modal-widget.js',
  '_static/css/sphinx-modal.css'
];

for (const relativePath of syncedFiles) {
  const source = path.join(root, 'docs/source-en', relativePath);
  const target = path.join(root, 'docs/source-zh', relativePath);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.copyFileSync(source, target);
  console.log(`synced ${relativePath}`);
}
