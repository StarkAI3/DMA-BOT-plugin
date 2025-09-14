#!/usr/bin/env python3
"""
Marathi Tabular Data Scraper for DMA
This script scrapes tabular data from Marathi DMA webpages
Uses the same robust logic as tabular_scraper.py but for Marathi URLs
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

class MarathiTabularScraper:
    """Marathi tabular data scraper based on the original tabular_scraper.py logic"""
    
    def __init__(self, output_dir="final_data/marathi_tabular_data"):
        self.output_dir = output_dir
        self.base_url = "https://mahadma.maharashtra.gov.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.scraped_data = []
        
    def read_marathi_tabular_links(self):
        """Read URLs from the Marathi tabular links file"""
        marathi_links_dir = os.path.join(os.path.dirname(__file__), "Marathi links")
        
        # Try tabular_links_MR.txt first
        tabular_links_file = os.path.join(marathi_links_dir, "tabular_links_MR.txt")
        if not os.path.exists(tabular_links_file):
            # Fallback to tabular_links_MR (without .txt)
            tabular_links_file = os.path.join(marathi_links_dir, "tabular_links_MR")
        
        urls = []
        if os.path.exists(tabular_links_file):
            with open(tabular_links_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    url = line.strip()
                    if url and url.startswith('http'):
                        urls.append(url)
                    elif url:  # Non-empty line that doesn't start with http
                        print(f"Warning: Line {line_num} doesn't contain a valid URL: {url}")
        else:
            print(f"Error: File {tabular_links_file} not found!")
            
        print(f"✅ Loaded {len(urls)} Marathi tabular URLs")
        return urls
    
    def get_filename_from_url(self, url):
        """Generate a clean filename from URL"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        filename = re.sub(r'[^\w\-_.]', '_', path)
        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_')
        return filename if filename else 'marathi_scraped_data'
    
    def extract_table_data(self, soup):
        """Extract table data using multiple strategies with proper header detection"""
        headers = []
        data = []
        links_data = []
        
        # Strategy 1: Look for specific table classes
        table = soup.find('table', class_='ulb-table')
        if table:
            # First, try to get headers from thead
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # If no thead, try to get headers from first row
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
            
            # Extract data rows
            tbody = table.find('tbody', id='ulb-table-body')
            if tbody:
                rows = tbody.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        row_data = []
                        row_links = []
                        for cell in cells:
                            # Extract text content
                            text = cell.get_text(strip=True)
                            row_data.append(text)
                            
                            # Extract links from this cell
                            links = []
                            for link in cell.find_all('a', href=True):
                                href = link.get('href')
                                if href:
                                    # Convert relative URLs to absolute
                                    if href.startswith('/'):
                                        href = f"{self.base_url}{href}"
                                    elif not href.startswith('http'):
                                        href = f"{self.base_url}/{href}"
                                    links.append(href)
                            row_links.append('; '.join(links) if links else '')
                        data.append(row_data)
                        links_data.append(row_links)
            else:
                # If no tbody, get all rows except header
                rows = table.find_all('tr')[1:] if headers else table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        row_data = []
                        row_links = []
                        for cell in cells:
                            text = cell.get_text(strip=True)
                            row_data.append(text)
                            
                            # Extract links
                            links = []
                            for link in cell.find_all('a', href=True):
                                href = link.get('href')
                                if href:
                                    if href.startswith('/'):
                                        href = f"{self.base_url}{href}"
                                    elif not href.startswith('http'):
                                        href = f"{self.base_url}/{href}"
                                    links.append(href)
                            row_links.append('; '.join(links) if links else '')
                        data.append(row_data)
                        links_data.append(row_links)
        
        # Strategy 2: Look for any table if first strategy failed
        if not data:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) > 1:  # Has header and data
                    # Get headers from first row
                    first_row = rows[0]
                    headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
                    
                    # Get data from remaining rows
                    for row in rows[1:]:
                        cells = row.find_all(['td', 'th'])
                        if cells:
                            row_data = []
                            row_links = []
                            for cell in cells:
                                text = cell.get_text(strip=True)
                                row_data.append(text)
                                
                                # Extract links
                                links = []
                                for link in cell.find_all('a', href=True):
                                    href = link.get('href')
                                    if href:
                                        if href.startswith('/'):
                                            href = f"{self.base_url}{href}"
                                        elif not href.startswith('http'):
                                            href = f"{self.base_url}/{href}"
                                        links.append(href)
                                row_links.append('; '.join(links) if links else '')
                            
                            if any(row_data):  # Only add non-empty rows
                                data.append(row_data)
                                links_data.append(row_links)
                    if data:  # If we found data, break
                        break
        
        return headers, data, links_data
    
    def has_next_page(self, soup, current_page=1):
        """Check if there's a next page available"""
        # Look for specific pagination elements
        pagination = soup.find('div', class_='ulb-pagination')
        if pagination:
            next_link = pagination.find('a', class_='next-page')
            if next_link and 'disabled' not in next_link.get('class', []):
                return True
        
        # Look for "Next" or ">" links that are not disabled
        next_links = soup.find_all('a', string=re.compile(r'next|>|→', re.I))
        for link in next_links:
            if 'disabled' not in link.get('class', []):
                return True
        
        # Look for numbered pagination
        page_links = soup.find_all('a', href=re.compile(r'paged=\d+'))
        if page_links:
            for link in page_links:
                if 'disabled' not in link.get('class', []):
                    return True
        
        # Look for WordPress style pagination (/page/N/)
        wordpress_links = soup.find_all('a', href=re.compile(r'/page/\d+/'))
        if wordpress_links:
            for link in wordpress_links:
                if 'disabled' not in link.get('class', []):
                    return True
        
        return False
    
    def scrape_tabular_data(self, url, max_pages=None):
        """Scrape tabular data from a Marathi URL with pagination support"""
        try:
            print(f"🔍 Scraping tabular data from: {url}")
            
            all_data = []
            all_links = []
            all_headers = []
            page = 1
            
            while True:
                # Construct URL for current page
                if page == 1:
                    current_url = url
                else:
                    # Try different pagination patterns
                    if '/services-type/' in url:
                        base_url = url.rstrip('/')
                        current_url = f"{base_url}/page/{page}/"
                    else:
                        if '?' in url:
                            current_url = f"{url}&paged={page}"
                        else:
                            current_url = f"{url}?paged={page}"
                
                print(f"  📄 Scraping page {page}: {current_url}")
                
                # Make the request
                response = self.session.get(current_url, timeout=30)
                
                # Handle 404 errors gracefully
                if response.status_code == 404:
                    print(f"  ⚠️ Page {page} not found (404), stopping pagination...")
                    break
                
                response.raise_for_status()
                
                # Parse the HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract table data
                headers, table_data, links_data = self.extract_table_data(soup)
                
                # If no data found on this page, we've reached the end
                if not table_data:
                    print(f"  ❌ No data found on page {page}, stopping...")
                    break
                
                # Store headers from first page
                if page == 1 and headers:
                    all_headers = headers
                    print(f"  ✅ Found headers: {headers}")
                
                all_data.extend(table_data)
                all_links.extend(links_data)
                print(f"  📊 Extracted {len(table_data)} records from page {page}")
                
                # Check for duplicates (same as previous page)
                if page > 3 and len(all_data) >= 2 * len(table_data) and len(table_data) > 0:
                    prev_start = len(all_data) - 2 * len(table_data)
                    prev_end = len(all_data) - len(table_data)
                    current_batch = all_data[-len(table_data):]
                    previous_batch = all_data[prev_start:prev_end] if prev_start >= 0 else []
                    
                    if previous_batch and current_batch == previous_batch:
                        print(f"  ⚠️ Duplicate data detected on page {page}, stopping...")
                        all_data = all_data[:-len(table_data)]
                        all_links = all_links[:-len(links_data)]
                        break
                
                # Check max pages limit
                if max_pages and page >= max_pages:
                    print(f"  📝 Reached maximum pages limit ({max_pages})")
                    break
                
                # Check for next page
                if not self.has_next_page(soup, page):
                    print(f"  ✅ No next page found, stopping...")
                    break
                
                # Safety check
                if page > 1000:
                    print("  ⚠️ Safety limit reached (1000 pages). Stopping.")
                    break
                
                page += 1
                time.sleep(1)  # Be respectful to the server
            
            print(f"  📈 Total extracted {len(all_data)} records from {page-1} pages")
            
            # Convert to DataFrame if we have data
            if all_data:
                # Use extracted headers if available
                if all_headers:
                    num_data_cols = len(all_data[0]) if all_data else 0
                    num_header_cols = len(all_headers)
                    
                    if num_data_cols == num_header_cols:
                        columns = all_headers
                    elif num_data_cols > num_header_cols:
                        columns = all_headers + [f'Column_{i+1}' for i in range(num_header_cols, num_data_cols)]
                    else:
                        columns = all_headers[:num_data_cols]
                else:
                    num_cols = len(all_data[0]) if all_data else 0
                    columns = [f'Column_{i+1}' for i in range(num_cols)]
                
                df = pd.DataFrame(all_data, columns=columns)
                
                # Add links column if we have links data
                if all_links and len(all_links) == len(all_data):
                    combined_links = []
                    for row_links in all_links:
                        all_row_links = []
                        for cell_links in row_links:
                            if cell_links.strip():
                                all_row_links.append(cell_links)
                        combined_links.append('; '.join(all_row_links) if all_row_links else '')
                    
                    df['Links'] = combined_links
                
                return df
            else:
                print("  ❌ No data found. The page structure might have changed.")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ❌ Error scraping {url}: {e}")
            return pd.DataFrame()
    
    def save_dataframe(self, df, url, format_type='json'):
        """Save DataFrame to JSON file in the organized format"""
        if df.empty:
            return None
        
        # Generate filename from URL
        url_filename = self.get_filename_from_url(url)
        
        # Convert DataFrame to the format expected by the preprocessing pipeline
        tabular_data = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in df.columns:
                row_dict[col] = str(row[col]) if pd.notna(row[col]) else ""
            tabular_data.append(row_dict)
        
        # Create the data structure
        data_structure = {
            'url': url,
            'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_type': 'marathi_tabular',
            'total_records': len(tabular_data),
            'columns': list(df.columns),
            'data': tabular_data
        }
        
        # Save as JSON file
        filename = f"{url_filename}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_structure, f, indent=2, ensure_ascii=False)
            
            print(f"  💾 Saved: {filename} ({len(tabular_data)} records)")
            return filepath
            
        except Exception as e:
            print(f"  ❌ Error saving {filename}: {e}")
            return None
    
    def scrape_all_marathi_tabular_urls(self):
        """Scrape all URLs from the Marathi tabular links file"""
        print("🚀 Starting Marathi Tabular Data Scraping")
        print("=" * 60)
        
        # Read URLs
        urls = self.read_marathi_tabular_links()
        if not urls:
            print("❌ No Marathi tabular URLs found to scrape!")
            return
        
        print(f"📋 Found {len(urls)} Marathi tabular URLs to scrape")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⏰ Estimated time: {len(urls) * 2:.0f} seconds (with 2s delay)")
        print()
        
        # Confirm before starting
        response = input("Do you want to proceed with Marathi tabular scraping? (y/N): ")
        if response.lower() != 'y':
            print("Scraping cancelled.")
            return
        
        start_time = time.time()
        successful_scrapes = 0
        failed_scrapes = 0
        total_records = 0
        
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*60}")
            print(f"🔗 Processing URL {i}/{len(urls)}: {url}")
            print(f"{'='*60}")
            
            try:
                # Scrape data from this URL
                df = self.scrape_tabular_data(url, max_pages=None)
                
                if not df.empty:
                    # Save data
                    filepath = self.save_dataframe(df, url, 'json')
                    
                    if filepath:
                        successful_scrapes += 1
                        total_records += len(df)
                        print(f"  ✅ Successfully scraped {len(df)} records")
                        
                        # Store for summary
                        self.scraped_data.append({
                            'url': url,
                            'records': len(df),
                            'columns': len(df.columns),
                            'filepath': filepath
                        })
                    else:
                        failed_scrapes += 1
                        print(f"  ❌ Failed to save data")
                else:
                    failed_scrapes += 1
                    print(f"  ❌ No data found")
                    
            except Exception as e:
                failed_scrapes += 1
                print(f"  ❌ Error processing URL: {e}")
            
            # Add delay between URLs
            if i < len(urls):
                print("  ⏳ Waiting 2 seconds...")
                time.sleep(2)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Generate summary
        self.generate_summary(successful_scrapes, failed_scrapes, total_records, elapsed_time)
        
        return self.scraped_data
    
    def generate_summary(self, successful_scrapes, failed_scrapes, total_records, elapsed_time):
        """Generate and save scraping summary"""
        print(f"\n{'='*60}")
        print("🎉 MARATHI TABULAR SCRAPING COMPLETE!")
        print(f"{'='*60}")
        print(f"⏰ Total time: {elapsed_time/60:.1f} minutes")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"✅ Successful scrapes: {successful_scrapes}")
        print(f"❌ Failed scrapes: {failed_scrapes}")
        print(f"📊 Total records: {total_records}")
        print()
        
        # Save summary file
        summary = {
            'scraping_session': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_urls': successful_scrapes + failed_scrapes,
                'successful_scrapes': successful_scrapes,
                'failed_scrapes': failed_scrapes,
                'total_records': total_records,
                'elapsed_time_minutes': round(elapsed_time/60, 2),
                'success_rate': f"{(successful_scrapes/(successful_scrapes + failed_scrapes)*100):.1f}%" if (successful_scrapes + failed_scrapes) > 0 else "0%"
            },
            'scraped_files': self.scraped_data,
            'output_directory': self.output_dir
        }
        
        summary_file = os.path.join(self.output_dir, "marathi_tabular_scraping_summary.json")
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"❌ Error saving summary: {e}")
        
        print()
        print("🚀 Next steps:")
        print("   1. Run: python3 src/advanced_preprocess.py (to process all data including Marathi tabular)")
        print("   2. Run: python3 src/advanced_embed_upsert.py (to embed with multilingual model)")
        print("   3. Test with Marathi queries!")

def main():
    """Main function"""
    scraper = MarathiTabularScraper()
    scraper.scrape_all_marathi_tabular_urls()

if __name__ == "__main__":
    main()
