#!/usr/bin/env python3
"""
Endpoint Comparison Tool
========================

This script compares discovered API endpoints from endpoint_catalog.json 
against harvested documentation to identify:
- Undocumented endpoints (exist in code but not in docs)
- Extra documented endpoints (exist in docs but not in code)
- Renamed/changed endpoints (potential mismatches)
- Documentation completeness scores

Usage: python endpoint_comparison_report.py
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class EndpointComparator:
    def __init__(self):
        self.discovered_endpoints = {}
        self.documented_endpoints = {}
        self.comparison_results = {
            'undocumented': [],
            'extra_documented': [],
            'potential_renames': [],
            'matched': [],
            'blueprint_coverage': {},
            'overall_stats': {}
        }
    
    def load_discovered_endpoints(self, catalog_path: str):
        """Load endpoints from the endpoint catalog JSON"""
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
            
            self.discovered_endpoints = {}
            for blueprint_name, blueprint_data in catalog.get('blueprints', {}).items():
                url_prefix = blueprint_data.get('url_prefix', '')
                for endpoint in blueprint_data.get('endpoints', []):
                    path = endpoint['path']
                    methods = endpoint['methods']
                    for method in methods:
                        key = f"{method} {path}"
                        self.discovered_endpoints[key] = {
                            'blueprint': blueprint_name,
                            'path': path,
                            'method': method,
                            'description': endpoint.get('description', ''),
                            'parameters': endpoint.get('parameters', {}),
                            'responses': endpoint.get('responses', {})
                        }
            
            print(f"✅ Loaded {len(self.discovered_endpoints)} discovered endpoints from catalog")
            return True
        except Exception as e:
            print(f"❌ Error loading discovered endpoints: {e}")
            return False
    
    def extract_endpoints_from_docs(self, docs_dir: str):
        """Extract endpoints from documentation files"""
        self.documented_endpoints = {}
        
        # Patterns to match HTTP endpoints in documentation
        endpoint_patterns = [
            r'```http\s*\n([A-Z]+)\s+(/[^\s\n]+)',  # ```http\nGET /path
            r'([A-Z]+)\s+(/[^\s\n]+)',  # GET /path
            r'`([A-Z]+)\s+(/[^\s\n`]+)`',  # `GET /path`
        ]
        
        # Walk through documentation directory
        for root, dirs, files in os.walk(docs_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    self._extract_from_markdown(file_path, endpoint_patterns)
        
        # Also check main README for additional endpoints
        readme_path = os.path.join(docs_dir, 'README.md')
        if os.path.exists(readme_path):
            self._extract_from_markdown(readme_path, endpoint_patterns)
        
        print(f"✅ Extracted {len(self.documented_endpoints)} documented endpoints")
    
    def _extract_from_markdown(self, file_path: str, patterns: List[str]):
        """Extract endpoints from a markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_name = os.path.basename(file_path)
            
            # Find all HTTP method + path combinations
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    method = match.group(1).upper()
                    path = match.group(2)
                    
                    # Skip if it's not a valid HTTP method
                    if method not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS']:
                        continue
                    
                    # Normalize path (remove query parameters, fragments)
                    path = path.split('?')[0].split('#')[0]
                    
                    key = f"{method} {path}"
                    if key not in self.documented_endpoints:
                        self.documented_endpoints[key] = {
                            'method': method,
                            'path': path,
                            'source_file': doc_name,
                            'blueprint': self._infer_blueprint_from_path(path),
                            'context': self._extract_context(content, match.start(), match.end())
                        }
        
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
    
    def _infer_blueprint_from_path(self, path: str) -> str:
        """Infer blueprint name from path"""
        path_parts = path.strip('/').split('/')
        if path_parts and path_parts[0]:
            # Handle special cases
            if path.startswith('/training/history'):
                return 'training_history'
            return path_parts[0]
        return 'unknown'
    
    def _extract_context(self, content: str, start: int, end: int, window: int = 100) -> str:
        """Extract surrounding context for an endpoint match"""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        context = content[context_start:context_end].replace('\n', ' ').strip()
        return context[:200] + '...' if len(context) > 200 else context
    
    def normalize_path(self, path: str) -> str:
        """Normalize path for comparison by replacing parameters"""
        # Replace Flask-style parameters with normalized placeholders
        normalized = re.sub(r'<[^>]+>', '{param}', path)
        # Replace specific parameter patterns
        normalized = re.sub(r'/\{[^}]+\}', '/{param}', normalized)
        return normalized
    
    def compare_endpoints(self):
        """Compare discovered vs documented endpoints"""
        discovered_keys = set(self.discovered_endpoints.keys())
        documented_keys = set(self.documented_endpoints.keys())
        
        # Create normalized versions for fuzzy matching
        discovered_normalized = {}
        documented_normalized = {}
        
        for key in discovered_keys:
            endpoint = self.discovered_endpoints[key]
            normalized_key = f"{endpoint['method']} {self.normalize_path(endpoint['path'])}"
            discovered_normalized[normalized_key] = key
        
        for key in documented_keys:
            endpoint = self.documented_endpoints[key]
            normalized_key = f"{endpoint['method']} {self.normalize_path(endpoint['path'])}"
            documented_normalized[normalized_key] = key
        
        discovered_normalized_keys = set(discovered_normalized.keys())
        documented_normalized_keys = set(documented_normalized.keys())
        
        # Find matches (exact and normalized)
        exact_matches = discovered_keys & documented_keys
        normalized_matches = discovered_normalized_keys & documented_normalized_keys
        
        # Find undocumented (in discovered but not documented)
        undocumented = discovered_keys - documented_keys
        undocumented_normalized = discovered_normalized_keys - documented_normalized_keys
        
        # Find extra documented (in docs but not discovered)
        extra_documented = documented_keys - discovered_keys
        extra_documented_normalized = documented_normalized_keys - discovered_normalized_keys
        
        # Store results
        self.comparison_results['matched'] = list(exact_matches)
        
        for key in undocumented:
            endpoint = self.discovered_endpoints[key]
            self.comparison_results['undocumented'].append({
                'endpoint': key,
                'blueprint': endpoint['blueprint'],
                'path': endpoint['path'],
                'method': endpoint['method'],
                'description': endpoint['description']
            })
        
        for key in extra_documented:
            endpoint = self.documented_endpoints[key]
            self.comparison_results['extra_documented'].append({
                'endpoint': key,
                'source_file': endpoint['source_file'],
                'blueprint': endpoint['blueprint'],
                'path': endpoint['path'],
                'method': endpoint['method'],
                'context': endpoint['context']
            })
        
        # Find potential renames (similar but not exact)
        self._find_potential_renames()
        
        # Calculate blueprint coverage
        self._calculate_blueprint_coverage()
        
        # Calculate overall statistics
        total_discovered = len(discovered_keys)
        total_documented = len(documented_keys)
        matched = len(exact_matches)
        
        self.comparison_results['overall_stats'] = {
            'total_discovered': total_discovered,
            'total_documented': total_documented,
            'exact_matches': matched,
            'undocumented': len(self.comparison_results['undocumented']),
            'extra_documented': len(self.comparison_results['extra_documented']),
            'match_rate': (matched / total_discovered * 100) if total_discovered > 0 else 0,
            'documentation_coverage': (matched / total_discovered * 100) if total_discovered > 0 else 0
        }
    
    def _find_potential_renames(self):
        """Find potential renames by comparing similar paths"""
        undocumented = [item['endpoint'] for item in self.comparison_results['undocumented']]
        extra_documented = [item['endpoint'] for item in self.comparison_results['extra_documented']]
        
        for undoc in undocumented:
            undoc_endpoint = self.discovered_endpoints[undoc]
            undoc_path = undoc_endpoint['path']
            undoc_method = undoc_endpoint['method']
            
            for extra_doc in extra_documented:
                extra_endpoint = self.documented_endpoints[extra_doc]
                extra_path = extra_endpoint['path']
                extra_method = extra_endpoint['method']
                
                # Check if methods match and paths are similar
                if (undoc_method == extra_method and 
                    self._paths_similar(undoc_path, extra_path)):
                    
                    self.comparison_results['potential_renames'].append({
                        'discovered': undoc,
                        'documented': extra_doc,
                        'similarity_reason': self._get_similarity_reason(undoc_path, extra_path),
                        'discovered_blueprint': undoc_endpoint['blueprint'],
                        'documented_source': extra_endpoint['source_file']
                    })
    
    def _paths_similar(self, path1: str, path2: str) -> bool:
        """Check if two paths are similar enough to be potential renames"""
        # Normalize both paths
        norm1 = self.normalize_path(path1)
        norm2 = self.normalize_path(path2)
        
        # Same normalized path
        if norm1 == norm2:
            return True
        
        # Similar blueprint/prefix
        parts1 = norm1.strip('/').split('/')
        parts2 = norm2.strip('/').split('/')
        
        if len(parts1) >= 2 and len(parts2) >= 2:
            return parts1[0] == parts2[0] and parts1[1] == parts2[1]
        
        return False
    
    def _get_similarity_reason(self, path1: str, path2: str) -> str:
        """Get reason why paths are considered similar"""
        if self.normalize_path(path1) == self.normalize_path(path2):
            return "Same normalized path (different parameter formats)"
        
        parts1 = path1.strip('/').split('/')
        parts2 = path2.strip('/').split('/')
        
        if len(parts1) >= 2 and len(parts2) >= 2:
            if parts1[0] == parts2[0] and parts1[1] == parts2[1]:
                return f"Same prefix: /{parts1[0]}/{parts1[1]}"
        
        return "Similar structure"
    
    def _calculate_blueprint_coverage(self):
        """Calculate documentation coverage per blueprint"""
        blueprint_stats = defaultdict(lambda: {
            'discovered': 0, 'documented': 0, 'matched': 0
        })
        
        # Count discovered endpoints per blueprint
        for endpoint in self.discovered_endpoints.values():
            blueprint_stats[endpoint['blueprint']]['discovered'] += 1
        
        # Count matched endpoints per blueprint
        for key in self.comparison_results['matched']:
            endpoint = self.discovered_endpoints[key]
            blueprint_stats[endpoint['blueprint']]['matched'] += 1
        
        # Count documented endpoints per blueprint (approximate)
        for endpoint in self.documented_endpoints.values():
            blueprint_stats[endpoint['blueprint']]['documented'] += 1
        
        # Calculate coverage percentages
        for blueprint, stats in blueprint_stats.items():
            if stats['discovered'] > 0:
                stats['coverage'] = (stats['matched'] / stats['discovered']) * 100
            else:
                stats['coverage'] = 0
        
        self.comparison_results['blueprint_coverage'] = dict(blueprint_stats)
    
    def generate_report(self, output_file: str = 'endpoint_comparison_report.md'):
        """Generate a comprehensive comparison report"""
        report_lines = []
        
        # Header
        report_lines.extend([
            "# API Endpoint Comparison Report",
            "",
            f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Purpose**: Compare discovered endpoints against harvested documentation",
            "",
            "## Executive Summary",
            ""
        ])
        
        # Overall statistics
        stats = self.comparison_results['overall_stats']
        report_lines.extend([
            f"- **Total Discovered Endpoints**: {stats['total_discovered']}",
            f"- **Total Documented Endpoints**: {stats['total_documented']}",
            f"- **Exact Matches**: {stats['exact_matches']}",
            f"- **Undocumented Endpoints**: {stats['undocumented']}",
            f"- **Extra Documented Endpoints**: {stats['extra_documented']}",
            f"- **Documentation Coverage**: {stats['documentation_coverage']:.1f}%",
            ""
        ])
        
        # Coverage status
        coverage = stats['documentation_coverage']
        if coverage >= 90:
            status = "🟢 **Excellent** - Documentation is very comprehensive"
        elif coverage >= 75:
            status = "🟡 **Good** - Most endpoints are documented"
        elif coverage >= 50:
            status = "🟠 **Fair** - Significant documentation gaps exist"
        else:
            status = "🔴 **Poor** - Major documentation update needed"
        
        report_lines.extend([
            f"**Overall Status**: {status}",
            "",
            "---",
            ""
        ])
        
        # Blueprint coverage
        if self.comparison_results['blueprint_coverage']:
            report_lines.extend([
                "## Blueprint Coverage Analysis",
                "",
                "| Blueprint | Discovered | Matched | Coverage | Status |",
                "|-----------|------------|---------|----------|---------|"
            ])
            
            for blueprint, stats in self.comparison_results['blueprint_coverage'].items():
                coverage_pct = stats['coverage']
                if coverage_pct >= 90:
                    status_icon = "🟢"
                elif coverage_pct >= 75:
                    status_icon = "🟡"
                elif coverage_pct >= 50:
                    status_icon = "🟠"
                else:
                    status_icon = "🔴"
                
                report_lines.append(
                    f"| **{blueprint}** | {stats['discovered']} | {stats['matched']} | "
                    f"{coverage_pct:.1f}% | {status_icon} |"
                )
            
            report_lines.extend(["", "---", ""])
        
        # Undocumented endpoints
        if self.comparison_results['undocumented']:
            report_lines.extend([
                "## 🚨 Undocumented Endpoints",
                "",
                "These endpoints exist in the code but are missing from documentation:",
                ""
            ])
            
            # Group by blueprint
            by_blueprint = defaultdict(list)
            for item in self.comparison_results['undocumented']:
                by_blueprint[item['blueprint']].append(item)
            
            for blueprint, items in by_blueprint.items():
                report_lines.extend([
                    f"### Blueprint: `{blueprint}`",
                    ""
                ])
                
                for item in items:
                    report_lines.extend([
                        f"#### `{item['method']} {item['path']}`",
                        f"- **Description**: {item['description'] or 'No description'}",
                        f"- **Blueprint**: {item['blueprint']}",
                        f"- **Action**: Add to API documentation",
                        ""
                    ])
        
        # Extra documented endpoints
        if self.comparison_results['extra_documented']:
            report_lines.extend([
                "## ❓ Extra Documented Endpoints",
                "",
                "These endpoints exist in documentation but weren't found in the code:",
                ""
            ])
            
            for item in self.comparison_results['extra_documented']:
                report_lines.extend([
                    f"#### `{item['method']} {item['path']}`",
                    f"- **Source**: {item['source_file']}",
                    f"- **Blueprint**: {item['blueprint']}",
                    f"- **Context**: {item['context']}",
                    f"- **Action**: Verify if endpoint was removed or renamed",
                    ""
                ])
        
        # Potential renames
        if self.comparison_results['potential_renames']:
            report_lines.extend([
                "## 🔄 Potential Renames/Changes",
                "",
                "These appear to be potential renames or changes:",
                ""
            ])
            
            for item in self.comparison_results['potential_renames']:
                report_lines.extend([
                    f"#### Potential Match Found",
                    f"- **Discovered**: `{item['discovered']}`",
                    f"- **Documented**: `{item['documented']}`",
                    f"- **Reason**: {item['similarity_reason']}",
                    f"- **Blueprint**: {item['discovered_blueprint']}",
                    f"- **Doc Source**: {item['documented_source']}",
                    f"- **Action**: Verify if this is the same endpoint with different naming",
                    ""
                ])
        
        # Matched endpoints summary
        if self.comparison_results['matched']:
            report_lines.extend([
                "## ✅ Successfully Matched Endpoints",
                "",
                f"**Total Matched**: {len(self.comparison_results['matched'])}",
                "",
                "<details>",
                "<summary>Click to expand full list</summary>",
                ""
            ])
            
            for key in sorted(self.comparison_results['matched']):
                endpoint = self.discovered_endpoints[key]
                report_lines.append(f"- `{key}` - {endpoint.get('description', 'No description')}")
            
            report_lines.extend([
                "",
                "</details>",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "## 📋 Action Items & Recommendations",
            "",
            "### High Priority",
            ""
        ])
        
        if self.comparison_results['undocumented']:
            report_lines.append(
                f"1. **Document {len(self.comparison_results['undocumented'])} missing endpoints** - "
                "These are implemented but not documented"
            )
        
        if self.comparison_results['potential_renames']:
            report_lines.append(
                f"2. **Verify {len(self.comparison_results['potential_renames'])} potential renames** - "
                "Check if these are the same endpoints with different naming"
            )
        
        if self.comparison_results['extra_documented']:
            report_lines.append(
                f"3. **Review {len(self.comparison_results['extra_documented'])} extra documented endpoints** - "
                "Verify if these were removed or renamed"
            )
        
        report_lines.extend([
            "",
            "### Documentation Quality Improvements",
            "",
            "- Ensure all endpoint descriptions are clear and complete",
            "- Add request/response examples for complex endpoints",
            "- Include error handling documentation",
            "- Keep parameter specifications up to date",
            "- Add integration examples for frontend developers",
            "",
            "### Process Improvements",
            "",
            "- Set up automated endpoint discovery to catch changes early",
            "- Implement documentation review process for new endpoints",
            "- Consider generating OpenAPI/Swagger specs from code",
            "- Regular documentation audits (recommended monthly)",
            "",
            "---",
            "",
            f"*Report generated by endpoint comparison tool - {len(report_lines)} lines*"
        ])
        
        # Write report to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"✅ Report generated: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error writing report: {e}")
            return False

def main():
    """Main execution function"""
    print("🔍 API Endpoint Comparison Tool")
    print("=" * 40)
    
    comparator = EndpointComparator()
    
    # Load discovered endpoints
    if not comparator.load_discovered_endpoints('endpoint_catalog.json'):
        print("❌ Failed to load endpoint catalog")
        return
    
    # Extract documented endpoints
    print("📖 Extracting endpoints from documentation...")
    comparator.extract_endpoints_from_docs('data/docs')
    
    # Perform comparison
    print("⚖️  Comparing endpoints...")
    comparator.compare_endpoints()
    
    # Generate report
    print("📝 Generating comparison report...")
    if comparator.generate_report('endpoint_comparison_report.md'):
        print("\n✅ Analysis complete! Check 'endpoint_comparison_report.md' for results.")
        
        # Print summary to console
        stats = comparator.comparison_results['overall_stats']
        print(f"\n📊 Quick Summary:")
        print(f"   • Total endpoints discovered: {stats['total_discovered']}")
        print(f"   • Total endpoints documented: {stats['total_documented']}")
        print(f"   • Perfect matches: {stats['exact_matches']}")
        print(f"   • Undocumented: {stats['undocumented']}")
        print(f"   • Extra documented: {stats['extra_documented']}")
        print(f"   • Documentation coverage: {stats['documentation_coverage']:.1f}%")
        
        # Status indicator
        coverage = stats['documentation_coverage']
        if coverage >= 90:
            print("   🟢 Status: Excellent documentation coverage!")
        elif coverage >= 75:
            print("   🟡 Status: Good coverage with minor gaps")
        elif coverage >= 50:
            print("   🟠 Status: Fair coverage - significant gaps exist")
        else:
            print("   🔴 Status: Poor coverage - major update needed")
    else:
        print("❌ Failed to generate report")

if __name__ == "__main__":
    main()
