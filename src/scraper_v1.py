import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import re
import os
from urllib.parse import urljoin, urlparse
import logging
from typing import List, Dict, Any

class DMAWebscraper:
    def __init__(self, base_url: str = "https://mahadma.maharashtra.gov.in", output_dir: str = "scraped_output"):
        self.base_url = base_url
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store scraped data
        self.scraped_data = []
    
    def create_slug_from_url(self, url: str) -> str:
        """Create a clean filename slug from URL"""
        # Parse the URL and get the path
        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        
        # Replace path separators and special characters
        slug = path.replace('/', '_').replace('-', '_')
        
        # Clean up the slug
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '_', slug)
        slug = slug.strip('_').lower()
        
        # Handle empty slugs
        if not slug:
            domain = parsed_url.netloc.replace('.', '_')
            slug = f"{domain}_home"
        
        # Limit length to avoid filesystem issues
        if len(slug) > 50:
            slug = slug[:50].rstrip('_')
        
        return slug
    
    def save_individual_file(self, scraped_data: Dict, line_number: int, slug: str):
        """Save individual scraped data to separate file"""
        filename = f"{line_number:03d}_{slug}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Process data for vector DB format
        chunks = self.process_for_vector_db(scraped_data)
        
        # Create comprehensive output with both raw and processed data
        content_length = len(scraped_data.get('content', ''))
        output_data = {
            'file_info': {
                'line_number': line_number,
                'slug': slug,
                'filename': filename,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'raw_data': scraped_data,
            'vector_chunks': chunks,
            'summary': {
                'total_chunks': len(chunks),
                'content_chunks': len([c for c in chunks if c['metadata']['type'] == 'content']),
                'table_chunks': len([c for c in chunks if c['metadata']['type'] == 'table']),
                'link_chunks': len([c for c in chunks if c['metadata']['type'] == 'links']),
                'total_tables': len(scraped_data.get('tables', [])),
                'total_links': len(scraped_data.get('links', [])),
                'content_length': content_length,
                'deduplication_applied': True
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved: {filename} ({len(chunks)} chunks)")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving {filename}: {str(e)}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for vector embeddings"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might interfere with embeddings
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        return text
    
    def deduplicate_content(self, content: str) -> str:
        """Remove duplicate sentences and paragraphs from content"""
        if not content:
            return ""
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        seen_sentences = set()
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Normalize sentence for comparison
            normalized = re.sub(r'\s+', ' ', sentence.lower())
            
            # Check if we've seen this sentence before
            if normalized not in seen_sentences:
                seen_sentences.add(normalized)
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences) + '.' if unique_sentences else ""
    
    def extract_text_with_links(self, element) -> str:
        """Extract text while preserving link context and avoiding duplicates"""
        if not element:
            return ""
        
        content_parts = []
        seen_text = set()  # Track seen text to avoid duplicates
        
        # Get all text elements, but avoid nested duplicates
        for item in element.find_all(['p', 'div', 'span', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text_content = item.get_text(strip=True)
            if not text_content or text_content in seen_text:
                continue
            
            # Check if this text is a substring of already seen text
            is_duplicate = False
            for seen in seen_text:
                if text_content in seen or seen in text_content:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            seen_text.add(text_content)
            
            # Find links within this element
            links = item.find_all('a', href=True)
            
            if links:
                # Add main text
                content_parts.append(text_content)
                
                # Add associated links
                for link in links:
                    href = link.get('href')
                    link_text = link.get_text(strip=True)
                    
                    # Make absolute URLs
                    if href:
                        if href.startswith('/'):
                            href = urljoin(self.base_url, href)
                        elif not href.startswith('http'):
                            href = urljoin(self.base_url, href)
                    
                    if link_text and href:
                        content_parts.append(f"Link: {link_text} - {href}")
            else:
                content_parts.append(text_content)
        
        return '\n'.join(content_parts)
    
    def extract_table_data(self, soup) -> List[Dict]:
        """Extract structured table data"""
        tables = soup.find_all('table')
        table_data = []
        
        for idx, table in enumerate(tables):
            rows = table.find_all('tr')
            if not rows:
                continue
            
            # Extract headers
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(['th', 'td']):
                headers.append(self.clean_text(th.get_text()))
            
            if not headers:
                continue
            
            # Extract data rows
            table_rows = []
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) == len(headers):
                    row_data = {}
                    for i, cell in enumerate(cells):
                        cell_text = self.clean_text(cell.get_text())
                        
                        # Check for links in cells
                        links = cell.find_all('a', href=True)
                        if links:
                            link_info = []
                            for link in links:
                                href = link.get('href')
                                if href:
                                    if href.startswith('/'):
                                        href = urljoin(self.base_url, href)
                                    link_info.append(f"Link: {href}")
                            
                            if link_info:
                                cell_text += " | " + " | ".join(link_info)
                        
                        row_data[headers[i]] = cell_text
                    
                    table_rows.append(row_data)
            
            if table_rows:
                table_data.append({
                    'table_index': idx,
                    'headers': headers,
                    'data': table_rows
                })
        
        return table_data
    
    def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single page and return structured data"""
        try:
            self.logger.info(f"Scraping: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract basic page info
            title = soup.find('title')
            title_text = self.clean_text(title.get_text()) if title else ""
            
            # Extract main content
            main_content = ""
            content_selectors = [
                'main', '.content', '#content', '.post-content', 
                '.entry-content', 'article', '.main-content'
            ]
            
            content_element = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    break
            
            if not content_element:
                content_element = soup.find('body')
            
            if content_element:
                main_content = self.extract_text_with_links(content_element)
            
            # Extract tables
            tables = self.extract_table_data(soup)
            
            # Extract all links on the page
            all_links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                link_text = self.clean_text(link.get_text())
                
                if href and href.startswith('/'):
                    href = urljoin(self.base_url, href)
                
                if href and link_text and href.startswith('http'):
                    all_links.append({
                        'text': link_text,
                        'url': href
                    })
            
            # Clean and deduplicate content
            cleaned_content = self.clean_text(main_content)
            original_length = len(cleaned_content)
            deduplicated_content = self.deduplicate_content(cleaned_content)
            final_length = len(deduplicated_content)
            
            if original_length > final_length:
                reduction_percent = ((original_length - final_length) / original_length) * 100
                self.logger.info(f"Content deduplication: {original_length} -> {final_length} chars ({reduction_percent:.1f}% reduction)")
            
            return {
                'url': url,
                'title': title_text,
                'content': deduplicated_content,
                'tables': tables,
                'links': all_links,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'title': "",
                'content': f"Error scraping page: {str(e)}",
                'tables': [],
                'links': [],
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def process_for_vector_db(self, scraped_data: Dict) -> List[Dict]:
        """Process scraped data into chunks suitable for vector embeddings"""
        chunks = []
        url = scraped_data['url']
        title = scraped_data['title']
        
        # Main content chunk
        if scraped_data['content']:
            content_text = f"Title: {title}\n\nContent: {scraped_data['content']}"
            chunks.append({
                'id': f"{urlparse(url).path}_content",
                'text': content_text,
                'metadata': {
                    'source_url': url,
                    'type': 'content',
                    'title': title,
                    'scraped_at': scraped_data['scraped_at']
                }
            })
        
        # Table chunks
        for i, table in enumerate(scraped_data['tables']):
            # Create a readable table representation
            table_text = f"Title: {title}\n\nTable {i+1} Data:\n"
            table_text += f"Headers: {', '.join(table['headers'])}\n\n"
            
            for row in table['data']:
                row_parts = []
                for header in table['headers']:
                    value = row.get(header, '')
                    if value:
                        row_parts.append(f"{header}: {value}")
                table_text += " | ".join(row_parts) + "\n"
            
            chunks.append({
                'id': f"{urlparse(url).path}_table_{i}",
                'text': table_text,
                'metadata': {
                    'source_url': url,
                    'type': 'table',
                    'title': title,
                    'table_headers': table['headers'],
                    'scraped_at': scraped_data['scraped_at']
                }
            })
        
        # Links chunk (if significant number of links)
        if len(scraped_data['links']) > 5:
            links_text = f"Title: {title}\n\nRelevant Links:\n"
            for link in scraped_data['links']:
                links_text += f"• {link['text']}: {link['url']}\n"
            
            chunks.append({
                'id': f"{urlparse(url).path}_links",
                'text': links_text,
                'metadata': {
                    'source_url': url,
                    'type': 'links',
                    'title': title,
                    'scraped_at': scraped_data['scraped_at']
                }
            })
        
        return chunks
    
    def scrape_from_file(self, file_path: str, delay: float = 1.0):
        """Scrape all URLs from a text file and save individual files"""
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            self.logger.info(f"Found {len(urls)} URLs to scrape")
            self.logger.info(f"Output directory: {self.output_dir}")
            
            all_chunks = []
            successful_scrapes = []
            failed_scrapes = []
            
            for line_number, url in enumerate(urls, 1):
                if not url.startswith('http'):
                    self.logger.warning(f"Skipping invalid URL at line {line_number}: {url}")
                    continue
                
                # Create slug from URL
                slug = self.create_slug_from_url(url)
                
                self.logger.info(f"Processing {line_number}/{len(urls)}: {slug}")
                
                # Scrape the page
                scraped_data = self.scrape_page(url)
                
                if scraped_data and not scraped_data['content'].startswith('Error scraping'):
                    # Save individual file
                    filepath = self.save_individual_file(scraped_data, line_number, slug)
                    
                    if filepath:
                        successful_scrapes.append({
                            'line_number': line_number,
                            'url': url,
                            'slug': slug,
                            'filepath': filepath
                        })
                        
                        # Add to master collection
                        chunks = self.process_for_vector_db(scraped_data)
                        all_chunks.extend(chunks)
                        self.scraped_data.append(scraped_data)
                    else:
                        failed_scrapes.append({'line_number': line_number, 'url': url, 'reason': 'File save failed'})
                else:
                    failed_scrapes.append({'line_number': line_number, 'url': url, 'reason': 'Scraping failed'})
                
                # Progress update
                if line_number % 10 == 0:
                    self.logger.info(f"Progress: {line_number}/{len(urls)} URLs processed")
                
                # Be respectful with delays
                time.sleep(delay)
            
            # Save summary file
            self.save_scraping_summary(successful_scrapes, failed_scrapes, len(urls))
            
            return all_chunks
            
        except FileNotFoundError:
            self.logger.error(f"File {file_path} not found")
            return []
    
    def save_scraping_summary(self, successful_scrapes: List[Dict], failed_scrapes: List[Dict], total_urls: int):
        """Save a summary of the scraping session"""
        summary = {
            'scraping_session': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_urls': total_urls,
                'successful_scrapes': len(successful_scrapes),
                'failed_scrapes': len(failed_scrapes),
                'success_rate': f"{(len(successful_scrapes)/total_urls*100):.1f}%" if total_urls > 0 else "0%"
            },
            'successful_files': successful_scrapes,
            'failed_scrapes': failed_scrapes,
            'output_directory': self.output_dir
        }
        
        summary_file = os.path.join(self.output_dir, "scraping_summary.json")
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Scraping Summary:")
            self.logger.info(f"  - Total URLs: {total_urls}")
            self.logger.info(f"  - Successful: {len(successful_scrapes)}")
            self.logger.info(f"  - Failed: {len(failed_scrapes)}")
            self.logger.info(f"  - Success Rate: {summary['scraping_session']['success_rate']}")
            self.logger.info(f"  - Summary saved to: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving summary: {str(e)}")
    
    def create_master_files(self):
        """Create master files combining all individual files"""
        try:
            # Collect all individual files
            all_chunks = []
            file_list = []
            
            for filename in sorted(os.listdir(self.output_dir)):
                if filename.endswith('.json') and not filename.startswith('scraping_summary') and not filename.startswith('master_'):
                    filepath = os.path.join(self.output_dir, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        file_list.append({
                            'filename': filename,
                            'line_number': data['file_info']['line_number'],
                            'slug': data['file_info']['slug'],
                            'url': data['raw_data']['url'],
                            'chunks_count': len(data['vector_chunks'])
                        })
                        
                        all_chunks.extend(data['vector_chunks'])
                        
                    except Exception as e:
                        self.logger.error(f"Error reading {filename}: {str(e)}")
                        continue
            
            # Save master vector chunks file (for Pinecone)
            master_chunks_file = os.path.join(self.output_dir, "master_vector_chunks.json")
            with open(master_chunks_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            
            # Save file index
            file_index = {
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_files': len(file_list),
                'total_chunks': len(all_chunks),
                'files': file_list
            }
            
            index_file = os.path.join(self.output_dir, "master_file_index.json")
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(file_index, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Created master files:")
            self.logger.info(f"  - {master_chunks_file} ({len(all_chunks)} chunks)")
            self.logger.info(f"  - {index_file} ({len(file_list)} files)")
            
            return len(all_chunks)
            
        except Exception as e:
            self.logger.error(f"Error creating master files: {str(e)}")
            return 0
    
    def save_data(self, chunks: List[Dict], output_file: str = "master_vector_chunks.json"):
        """Save processed data to JSON file"""
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    
    def save_raw_data(self, output_file: str = "master_raw_data.json"):
        """Save raw scraped data"""
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved raw data to {output_path}")


# Usage example
if __name__ == "__main__":
    scraper = DMAWebscraper(output_dir="DMA_scraped_output")
    
    # Scrape all URLs from your file (creates individual files)
    chunks = scraper.scrape_from_file('DMA_Links.txt')
    
    # Create master files for easy access
    total_chunks = scraper.create_master_files()
    
    print("\n" + "="*60)
    print("🎉 SCRAPING COMPLETE!")
    print("="*60)
    print(f"📁 Output directory: DMA_scraped_output/")
    print(f"📄 Individual files: {len(scraper.scraped_data)} files created")
    print(f"🔗 Total chunks: {total_chunks} chunks for vector embedding")
    print("\n📋 Files created:")
    print("   • Individual JSON files: [LINE_NUMBER]_[SLUG].json")
    print("   • scraping_summary.json - Session overview")
    print("   • master_vector_chunks.json - All chunks for Pinecone")
    print("   • master_file_index.json - File directory")
    print("\n🚀 Ready for Pinecone upload!")
    print("   Use: master_vector_chunks.json")