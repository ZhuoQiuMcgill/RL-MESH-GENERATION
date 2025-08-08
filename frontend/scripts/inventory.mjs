#!/usr/bin/env node

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const SRC_DIR = path.join(__dirname, '../src');
const OUTPUT_DIR = path.join(__dirname, '../data/docs/overview');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'code-inventory.md');
const JSON_OUTPUT_FILE = path.join(OUTPUT_DIR, 'code-inventory.json');

// File extensions to analyze
const VALID_EXTENSIONS = ['.js', '.jsx', '.ts', '.tsx', '.mjs'];
const ASSET_EXTENSIONS = ['.css', '.scss', '.sass', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.ico'];

// Import patterns
const IMPORT_PATTERNS = [
  // ES6 imports
  /import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)(?:\s*,\s*(?:\{[^}]*\}|\*\s+as\s+\w+|\w+))*\s+from\s+)?['"`]([^'"`]+)['"`]/g,
  // Dynamic imports
  /import\s*\(\s*['"`]([^'"`]+)['"`]\s*\)/g,
  // Require statements
  /require\s*\(\s*['"`]([^'"`]+)['"`]\s*\)/g
];

class CodeInventory {
  constructor() {
    this.files = [];
    this.importGraph = new Map();
    this.reverseDependencies = new Map();
    this.categorizedFiles = {
      components: [],
      pages: [],
      hooks: [],
      context: [],
      utils: [],
      assets: [],
      other: []
    };
    this.orphanedFiles = new Set();
  }

  async analyzeCodebase() {
    console.log('🔍 Starting codebase analysis...');
    
    // Ensure output directory exists
    await fs.mkdir(OUTPUT_DIR, { recursive: true });
    
    // Recursively find all files
    await this.findFiles(SRC_DIR);
    
    // Analyze each file
    for (const file of this.files) {
      await this.analyzeFile(file);
    }
    
    // Categorize files
    this.categorizeFiles();
    
    // Find orphaned files
    this.findOrphanedFiles();
    
    // Generate reports
    await this.generateReports();
    
    console.log('✅ Analysis complete!');
    console.log(`📄 Reports generated:`);
    console.log(`   - Markdown: ${OUTPUT_FILE}`);
    console.log(`   - JSON: ${JSON_OUTPUT_FILE}`);
  }

  async findFiles(dir, relativePath = '') {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      const relPath = path.join(relativePath, entry.name);
      
      if (entry.isDirectory()) {
        await this.findFiles(fullPath, relPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name);
        if (VALID_EXTENSIONS.includes(ext) || ASSET_EXTENSIONS.includes(ext)) {
          this.files.push({
            name: entry.name,
            path: fullPath,
            relativePath: relPath.replace(/\\/g, '/'), // Normalize path separators
            extension: ext,
            directory: path.dirname(relPath).replace(/\\/g, '/'),
            isAsset: ASSET_EXTENSIONS.includes(ext)
          });
        }
      }
    }
  }

  async analyzeFile(file) {
    if (file.isAsset) {
      this.importGraph.set(file.relativePath, []);
      return;
    }

    try {
      const content = await fs.readFile(file.path, 'utf-8');
      const imports = this.extractImports(content, file);
      
      this.importGraph.set(file.relativePath, imports);
      
      // Build reverse dependency graph
      for (const importPath of imports) {
        if (!this.reverseDependencies.has(importPath)) {
          this.reverseDependencies.set(importPath, []);
        }
        this.reverseDependencies.get(importPath).push(file.relativePath);
      }
      
    } catch (error) {
      console.warn(`⚠️  Could not analyze ${file.relativePath}: ${error.message}`);
      this.importGraph.set(file.relativePath, []);
    }
  }

  extractImports(content, file) {
    const imports = new Set();
    
    for (const pattern of IMPORT_PATTERNS) {
      pattern.lastIndex = 0; // Reset regex state
      let match;
      
      while ((match = pattern.exec(content)) !== null) {
        let importPath = match[1];
        
        // Skip external packages (node_modules)
        if (!importPath.startsWith('.') && !importPath.startsWith('/')) {
          continue;
        }
        
        // Resolve relative imports
        importPath = this.resolveImportPath(importPath, file);
        if (importPath) {
          imports.add(importPath);
        }
      }
    }
    
    return Array.from(imports);
  }

  resolveImportPath(importPath, currentFile) {
    try {
      // Handle relative imports
      if (importPath.startsWith('.')) {
        const currentDir = path.dirname(currentFile.relativePath);
        let resolvedPath = path.join(currentDir, importPath).replace(/\\/g, '/');
        
        // Try to resolve with different extensions if no extension provided
        if (!path.extname(resolvedPath)) {
          const possibleExtensions = ['.js', '.jsx', '.ts', '.tsx', '.mjs'];
          for (const ext of possibleExtensions) {
            const withExt = resolvedPath + ext;
            if (this.files.some(f => f.relativePath === withExt)) {
              return withExt;
            }
          }
          
          // Try index files
          const indexFiles = possibleExtensions.map(ext => `${resolvedPath}/index${ext}`);
          for (const indexFile of indexFiles) {
            if (this.files.some(f => f.relativePath === indexFile)) {
              return indexFile;
            }
          }
        }
        
        // Check if resolved path exists
        if (this.files.some(f => f.relativePath === resolvedPath)) {
          return resolvedPath;
        }
      }
      
      return null;
    } catch (error) {
      return null;
    }
  }

  categorizeFiles() {
    for (const file of this.files) {
      const dir = file.directory;
      const name = file.name.toLowerCase();
      
      if (file.isAsset) {
        this.categorizedFiles.assets.push(file);
      } else if (dir.includes('components') || dir === 'components') {
        this.categorizedFiles.components.push(file);
      } else if (dir.includes('pages') || dir === 'pages') {
        this.categorizedFiles.pages.push(file);
      } else if (dir.includes('hooks') || dir === 'hooks' || name.startsWith('use')) {
        this.categorizedFiles.hooks.push(file);
      } else if (dir.includes('context') || dir === 'context' || name.includes('context') || name.includes('provider')) {
        this.categorizedFiles.context.push(file);
      } else if (dir.includes('utils') || dir === 'utils' || dir.includes('helpers')) {
        this.categorizedFiles.utils.push(file);
      } else {
        this.categorizedFiles.other.push(file);
      }
    }
  }

  findOrphanedFiles() {
    // Files that are not imported by any other file
    const allImportedFiles = new Set();
    
    for (const imports of this.importGraph.values()) {
      for (const importPath of imports) {
        allImportedFiles.add(importPath);
      }
    }
    
    for (const file of this.files) {
      if (!allImportedFiles.has(file.relativePath)) {
        // Skip entry points (main.jsx, App.jsx, index files)
        const fileName = file.name.toLowerCase();
        const isEntryPoint = fileName === 'main.jsx' || fileName === 'app.jsx' || 
                            fileName.startsWith('index.') || file.isAsset;
        
        if (!isEntryPoint) {
          this.orphanedFiles.add(file.relativePath);
        }
      }
    }
  }

  async generateReports() {
    const data = this.generateReportData();
    
    // Generate JSON report
    await fs.writeFile(JSON_OUTPUT_FILE, JSON.stringify(data, null, 2));
    
    // Generate Markdown report
    const markdown = this.generateMarkdownReport(data);
    await fs.writeFile(OUTPUT_FILE, markdown);
  }

  generateReportData() {
    return {
      summary: {
        totalFiles: this.files.length,
        totalCodeFiles: this.files.filter(f => !f.isAsset).length,
        totalAssets: this.files.filter(f => f.isAsset).length,
        totalImports: Array.from(this.importGraph.values()).reduce((sum, imports) => sum + imports.length, 0),
        orphanedFiles: this.orphanedFiles.size
      },
      categories: Object.fromEntries(
        Object.entries(this.categorizedFiles).map(([category, files]) => [
          category,
          files.map(f => ({
            name: f.name,
            path: f.relativePath,
            imports: this.importGraph.get(f.relativePath) || [],
            importedBy: this.reverseDependencies.get(f.relativePath) || []
          }))
        ])
      ),
      importGraph: Object.fromEntries(this.importGraph),
      reverseDependencies: Object.fromEntries(this.reverseDependencies),
      orphanedFiles: Array.from(this.orphanedFiles),
      generatedAt: new Date().toISOString()
    };
  }

  generateMarkdownReport(data) {
    const md = [];
    
    // Header
    md.push('# Code Inventory Report');
    md.push('');
    md.push(`Generated on: ${new Date().toLocaleString()}`);
    md.push('');
    
    // Summary
    md.push('## Summary');
    md.push('');
    md.push('| Metric | Count |');
    md.push('|--------|-------|');
    md.push(`| Total Files | ${data.summary.totalFiles} |`);
    md.push(`| Code Files | ${data.summary.totalCodeFiles} |`);
    md.push(`| Asset Files | ${data.summary.totalAssets} |`);
    md.push(`| Total Imports | ${data.summary.totalImports} |`);
    md.push(`| Orphaned Files | ${data.summary.orphanedFiles} |`);
    md.push('');
    
    // Components
    md.push('## Components');
    md.push('');
    this.addCategorySection(md, data.categories.components, 'Components organized by folder');
    
    // Pages
    md.push('## Pages');
    md.push('');
    this.addCategorySection(md, data.categories.pages, 'Page components and their dependencies');
    
    // Hooks
    md.push('## Custom Hooks');
    md.push('');
    this.addCategorySection(md, data.categories.hooks, 'Custom React hooks');
    
    // Context
    md.push('## Context Providers');
    md.push('');
    this.addCategorySection(md, data.categories.context, 'React context providers and related files');
    
    // Utils
    md.push('## Utilities');
    md.push('');
    this.addCategorySection(md, data.categories.utils, 'Utility functions and helpers');
    
    // Assets
    md.push('## Assets');
    md.push('');
    this.addCategorySection(md, data.categories.assets, 'Static assets (CSS, images, etc.)');
    
    // Other
    if (data.categories.other.length > 0) {
      md.push('## Other Files');
      md.push('');
      this.addCategorySection(md, data.categories.other, 'Other source files');
    }
    
    // Orphaned Files
    if (data.orphanedFiles.length > 0) {
      md.push('## Orphaned Files');
      md.push('');
      md.push('Files that are not imported by any other file:');
      md.push('');
      for (const file of data.orphanedFiles) {
        md.push(`- \`${file}\``);
      }
      md.push('');
    }
    
    // Import Graph Summary
    md.push('## Import Graph Analysis');
    md.push('');
    md.push('### Most Imported Files');
    md.push('');
    const mostImported = Object.entries(data.reverseDependencies)
      .sort(([,a], [,b]) => b.length - a.length)
      .slice(0, 10);
    
    if (mostImported.length > 0) {
      md.push('| File | Imported By | Count |');
      md.push('|------|-------------|-------|');
      for (const [file, importedBy] of mostImported) {
        md.push(`| \`${file}\` | ${importedBy.length} files | ${importedBy.length} |`);
      }
      md.push('');
    }
    
    md.push('### Files with Most Dependencies');
    md.push('');
    const mostDependencies = Object.entries(data.importGraph)
      .sort(([,a], [,b]) => b.length - a.length)
      .slice(0, 10);
    
    if (mostDependencies.length > 0) {
      md.push('| File | Dependencies | Count |');
      md.push('|------|--------------|-------|');
      for (const [file, dependencies] of mostDependencies) {
        if (dependencies.length > 0) {
          md.push(`| \`${file}\` | ${dependencies.length} imports | ${dependencies.length} |`);
        }
      }
      md.push('');
    }
    
    return md.join('\n');
  }

  addCategorySection(md, files, description) {
    md.push(description);
    md.push('');
    
    if (files.length === 0) {
      md.push('*No files found in this category*');
      md.push('');
      return;
    }
    
    // Group by directory
    const byDirectory = {};
    for (const file of files) {
      const dir = path.dirname(file.path);
      if (!byDirectory[dir]) byDirectory[dir] = [];
      byDirectory[dir].push(file);
    }
    
    for (const [directory, dirFiles] of Object.entries(byDirectory)) {
      const cleanDir = directory.replace(/\\/g, '/');
      md.push(`### ${cleanDir}`);
      md.push('');
      
      for (const file of dirFiles) {
        md.push(`#### \`${file.name}\``);
        
        if (file.imports && file.imports.length > 0) {
          md.push('**Imports:**');
          for (const imp of file.imports) {
            md.push(`- \`${imp}\``);
          }
        }
        
        if (file.importedBy && file.importedBy.length > 0) {
          md.push('**Imported by:**');
          for (const dep of file.importedBy) {
            md.push(`- \`${dep}\``);
          }
        }
        
        if ((!file.imports || file.imports.length === 0) && (!file.importedBy || file.importedBy.length === 0)) {
          md.push('*No dependencies tracked*');
        }
        
        md.push('');
      }
    }
  }
}

// Main execution
async function main() {
  try {
    const inventory = new CodeInventory();
    await inventory.analyzeCodebase();
  } catch (error) {
    console.error('❌ Error during analysis:', error);
    process.exit(1);
  }
}

// Always run main when script is executed directly
if (import.meta.url.endsWith('inventory.mjs') && process.argv[1] && process.argv[1].endsWith('inventory.mjs')) {
  main();
} else {
  // Fallback - run main anyway for this script
  main();
}

export default CodeInventory;
