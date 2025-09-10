import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import csv
import os
import re
from urllib.parse import urljoin, urlparse

def read_urls_from_file(filename='tabular_links.txt'):
    """
    Read URLs from a text file, one URL per line
    Returns a list of clean URLs
    """
    urls = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append(url)
                elif url:  # Non-empty line that doesn't start with http
                    print(f"Warning: Line {line_num} doesn't contain a valid URL: {url}")
        print(f"Loaded {len(urls)} URLs from {filename}")
        return urls
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        return []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

def get_filename_from_url(url):
    """
    Generate a clean filename from URL
    """
    parsed = urlparse(url)
    # Extract path and clean it
    path = parsed.path.strip('/')
    # Replace slashes and special characters with underscores
    filename = re.sub(r'[^\w\-_.]', '_', path)
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    return filename if filename else 'scraped_data'

def scrape_generic_table(url, max_pages=None):
    """
    Generic table scraper that can handle different page structures
    Returns a pandas DataFrame with scraped data
    """
    
    # HTTP headers to mimic a real browser request
    http_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        print(f"Fetching data from: {url}")
        
        all_data = []
        all_links = []
        all_headers = []
        page = 1
        
        while True:
            # Construct URL for current page
            if page == 1:
                current_url = url
            else:
                # Try different pagination patterns based on URL structure
                if '/services-type/' in url:
                    # For service-type URLs, use WordPress style pagination
                    # Fix double slash issue - ensure URL ends with single slash
                    base_url = url.rstrip('/')
                    current_url = f"{base_url}/page/{page}/"
                else:
                    # For other URLs, use paged parameter
                    if '?' in url:
                        current_url = f"{url}&paged={page}"
                    else:
                        current_url = f"{url}?paged={page}"
            
            print(f"Scraping page {page}: {current_url}")
            
            # Make the request
            response = requests.get(current_url, headers=http_headers, timeout=30)
            
            # Handle 404 errors gracefully - just stop pagination
            if response.status_code == 404:
                print(f"Page {page} not found (404), stopping pagination...")
                break
            
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try multiple table detection strategies
            headers, table_data, links_data = extract_table_data(soup)
            
            # If no data found on this page, we've reached the end
            if not table_data:
                print(f"No data found on page {page}, stopping...")
                break
            
            # Store headers from first page
            if page == 1 and headers:
                all_headers = headers
                print(f"Found headers: {headers}")
            
            all_data.extend(table_data)
            all_links.extend(links_data)
            print(f"Extracted {len(table_data)} records from page {page}")
            
            # Check if we're getting duplicate data (same as previous page)
            # Only check for duplicates if we have enough data to compare and it's not the first few pages
            if page > 3 and len(all_data) >= 2 * len(table_data) and len(table_data) > 0:
                # Compare last batch with previous batch
                prev_start = len(all_data) - 2 * len(table_data)
                prev_end = len(all_data) - len(table_data)
                current_batch = all_data[-len(table_data):]
                previous_batch = all_data[prev_start:prev_end] if prev_start >= 0 else []
                
                # More intelligent duplicate detection - check if ALL rows are identical
                if previous_batch and current_batch == previous_batch:
                    print(f"⚠️  Duplicate data detected on page {page}, stopping...")
                    # Remove the duplicate data
                    all_data = all_data[:-len(table_data)]
                    all_links = all_links[:-len(links_data)]
                    break
                # Also check if we're getting empty or very small batches repeatedly
                elif len(table_data) <= 2 and page > 5:
                    print(f"⚠️  Very small data batch ({len(table_data)} records) on page {page}, stopping...")
                    break
            
            # Check if we should continue to next page
            if max_pages and page >= max_pages:
                print(f"Reached maximum pages limit ({max_pages})")
                break
            
            # Check for pagination - improved logic
            if not has_next_page(soup, page):
                print("No next page found, stopping...")
                # Debug: show what pagination elements were found
                pagination_divs = soup.find_all('div', class_=re.compile(r'pagination|page', re.I))
                if pagination_divs:
                    print(f"Found pagination divs: {[div.get('class') for div in pagination_divs]}")
                page_links = soup.find_all('a', href=re.compile(r'paged|page'))
                if page_links:
                    print(f"Found page links: {[link.get('href') for link in page_links[:5]]}")
                break
            
            # Additional safety check: if we've scraped more than 1000 pages, something is wrong
            if page > 1000:
                print("⚠️  Safety limit reached (1000 pages). Stopping to prevent infinite loop.")
                break
            
            # Check if we're getting the same data (potential infinite loop)
            if page > 10 and len(table_data) == 0:
                print("No data found for multiple pages, stopping...")
                break
            
            page += 1
            time.sleep(1)  # Be respectful to the server
        
        print(f"Total extracted {len(all_data)} records from {page-1} pages")
        
        # Convert to DataFrame if we have data
        if all_data:
            # Use extracted headers if available, otherwise use generic names
            if all_headers:
                # Ensure headers match data columns
                num_data_cols = len(all_data[0]) if all_data else 0
                num_header_cols = len(all_headers)
                
                if num_data_cols == num_header_cols:
                    columns = all_headers
                elif num_data_cols > num_header_cols:
                    # Add generic names for extra columns
                    columns = all_headers + [f'Column_{i+1}' for i in range(num_header_cols, num_data_cols)]
                else:
                    # Truncate headers if data has fewer columns
                    columns = all_headers[:num_data_cols]
            else:
                # Use generic column names
                num_cols = len(all_data[0]) if all_data else 0
                columns = [f'Column_{i+1}' for i in range(num_cols)]
            
            df = pd.DataFrame(all_data, columns=columns)
            
            # Add links column if we have links data
            if all_links and len(all_links) == len(all_data):
                # Create a combined links column with all links from each row
                combined_links = []
                for row_links in all_links:
                    # Combine all non-empty links from this row
                    all_row_links = []
                    for cell_links in row_links:
                        if cell_links.strip():
                            all_row_links.append(cell_links)
                    combined_links.append('; '.join(all_row_links) if all_row_links else '')
                
                df['Links'] = combined_links
            
            return df
        else:
            print("No data found. The page structure might have changed.")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return pd.DataFrame()
    
    except Exception as e:
        import traceback
        print(f"Error parsing the data: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return pd.DataFrame()

def extract_table_data(soup):
    """
    Extract table data using multiple strategies with proper header detection
    Also extracts embedded links from table cells
    Returns tuple: (headers, data_rows, links_data)
    """
    headers = []
    data = []
    links_data = []
    
    # Strategy 1: Look for specific table classes (original approach)
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
                                    href = f"https://mahadma.maharashtra.gov.in{href}"
                                elif not href.startswith('http'):
                                    href = f"https://mahadma.maharashtra.gov.in/{href}"
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
                                    href = f"https://mahadma.maharashtra.gov.in{href}"
                                elif not href.startswith('http'):
                                    href = f"https://mahadma.maharashtra.gov.in/{href}"
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
                                        href = f"https://mahadma.maharashtra.gov.in{href}"
                                    elif not href.startswith('http'):
                                        href = f"https://mahadma.maharashtra.gov.in/{href}"
                                    links.append(href)
                            row_links.append('; '.join(links) if links else '')
                        
                        if any(row_data):  # Only add non-empty rows
                            data.append(row_data)
                            links_data.append(row_links)
                if data:  # If we found data, break
                    break
    
    # Strategy 3: Look for structured lists or divs
    if not data:
        # Look for div-based tables or structured content
        structured_divs = soup.find_all('div', class_=re.compile(r'table|row|item|list'))
        if structured_divs:
            for div in structured_divs:
                # Try to extract structured data from divs
                text_content = div.get_text(strip=True)
                if text_content and len(text_content) > 10:  # Meaningful content
                    # Split by common separators
                    parts = re.split(r'\s{2,}|\t|\n', text_content)
                    if len(parts) > 1:
                        data.append(parts)
    
    return headers, data, links_data

def has_next_page(soup, current_page=1):
    """
    Check if there's a next page available - improved approach
    """
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
    
    # Look for numbered pagination - be more specific
    page_links = soup.find_all('a', href=re.compile(r'paged=\d+'))
    if page_links:
        # Only return True if we find actual clickable page numbers
        for link in page_links:
            if 'disabled' not in link.get('class', []):
                return True
    
    # Look for WordPress style pagination (/page/N/)
    wordpress_links = soup.find_all('a', href=re.compile(r'/page/\d+/'))
    if wordpress_links:
        for link in wordpress_links:
            if 'disabled' not in link.get('class', []):
                return True
    
    # Look for any pagination links with page numbers
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        href = link.get('href', '')
        if 'paged=' in href or 'page=' in href or '/page/' in href:
            # Check if it's a higher page number
            page_match = re.search(r'[?&](?:paged|page)=(\d+)|/page/(\d+)/', href)
            if page_match:
                page_num = int(page_match.group(1) or page_match.group(2))
                if page_num > current_page:  # If we find links to higher pages
                    return True
    
    # If no clear pagination found, assume no next page
    return False

def scrape_maharashtra_councils(max_pages=None):
    """
    Scrapes tabular data of municipal councils from Maharashtra government website
    Returns a pandas DataFrame with all council information
    
    Args:
        max_pages (int): Maximum number of pages to scrape. If None, scrapes all pages.
    """
    
    base_url = "https://mahadma.maharashtra.gov.in/en/list-of-councils-2/"
    
    # Headers to mimic a real browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        print("Fetching data from Maharashtra government website...")
        
        all_councils_data = []
        page = 1
        
        while True:
            # Construct URL for current page
            if page == 1:
                url = base_url
            else:
                url = f"{base_url}?paged={page}"
            
            print(f"Scraping page {page}...")
            
            # Make the request
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table - looking for the data structure we saw
            councils_data = []
            
            # The data appears to be in a specific format, let's extract it
            # Looking for table rows or similar structured data
            table = soup.find('table', class_='ulb-table')
            
            if table:
                # Find the tbody with the data
                tbody = table.find('tbody', id='ulb-table-body')
                if tbody:
                    rows = tbody.find_all('tr')
                    print(f"Found {len(rows)} data rows in tbody")
                    
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 8:  # Ensure we have all columns
                            # Extract text from each cell, handling nested elements
                            row_data = []
                            for cell in cells:
                                # Get text content, handling nested <strong> and <span> tags
                                cell_text = cell.get_text(strip=True)
                                row_data.append(cell_text)
                            
                            if len(row_data) >= 8:
                                councils_data.append(row_data[:8])  # Take first 8 columns
                else:
                    print("No tbody found, trying to extract from table directly...")
                    rows = table.find_all('tr')[1:]  # Skip header row
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 8:
                            row_data = [cell.get_text(strip=True) for cell in cells]
                            councils_data.append(row_data[:8])
            else:
                print("ulb-table not found, searching for alternative structures...")
                # Try to find any table
                table = soup.find('table')
                if table:
                    print("Found alternative table structure")
                    rows = table.find_all('tr')[1:]  # Skip header row
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 8:
                            row_data = [cell.get_text(strip=True) for cell in cells]
                            councils_data.append(row_data[:8])
            
            # If no data found on this page, we've reached the end
            if not councils_data:
                print(f"No data found on page {page}, stopping...")
                break
            
            all_councils_data.extend(councils_data)
            print(f"Extracted {len(councils_data)} council records from page {page}")
            
            # Check if we should continue to next page
            if max_pages and page >= max_pages:
                print(f"Reached maximum pages limit ({max_pages})")
                break
            
            # Check for pagination - look for next page link
            pagination = soup.find('div', class_='ulb-pagination')
            if pagination:
                next_link = pagination.find('a', class_='next-page')
                if not next_link:
                    print("No next page link found, stopping...")
                    break
            else:
                print("No pagination found, stopping...")
                break
            
            page += 1
            time.sleep(1)  # Be respectful to the server
        
        print(f"Total extracted {len(all_councils_data)} council records from {page-1} pages")
            
        # Convert to DataFrame
        if all_councils_data:
            columns = [
                'Sr_No', 'Division', 'District', 'Council_Name', 
                'Class', 'Foundation_Year', 'Population_2011', 'Area_sq_km'
            ]
            
            df = pd.DataFrame(all_councils_data, columns=columns)
            
            # Clean up the data
            df['Sr_No'] = pd.to_numeric(df['Sr_No'], errors='coerce')
            df['Foundation_Year'] = pd.to_numeric(df['Foundation_Year'], errors='coerce')
            df['Population_2011'] = df['Population_2011'].str.replace(',', '').astype('Int64', errors='ignore')
            df['Area_sq_km'] = pd.to_numeric(df['Area_sq_km'], errors='coerce')
            
            print(f"Successfully scraped {len(df)} municipal councils!")
            return df
        else:
            print("No data found. The page structure might have changed.")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return pd.DataFrame()
    
    except Exception as e:
        import traceback
        print(f"Error parsing the data: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return pd.DataFrame()

def save_data(df, format_type='csv', custom_filename=None):
    """
    Save the scraped data to different formats
    """
    if df.empty:
        print("No data to save!")
        return None
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if custom_filename:
        base_filename = custom_filename
    else:
        base_filename = f'maharashtra_councils_{timestamp}'
    
    saved_files = []
    
    if format_type.lower() == 'csv':
        filename = f'{base_filename}.csv'
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Data saved to {filename}")
        saved_files.append(filename)
    
    elif format_type.lower() == 'excel':
        filename = f'{base_filename}.xlsx'
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Data saved to {filename}")
        saved_files.append(filename)
    
    elif format_type.lower() == 'json':
        filename = f'{base_filename}.json'
        df.to_json(filename, orient='records', indent=2)
        print(f"Data saved to {filename}")
        saved_files.append(filename)
    
    elif format_type.lower() == 'all':
        # Save in all formats
        csv_file = f'{base_filename}.csv'
        excel_file = f'{base_filename}.xlsx'
        json_file = f'{base_filename}.json'
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        df.to_excel(excel_file, index=False, engine='openpyxl')
        df.to_json(json_file, orient='records', indent=2)
        
        print(f"Data saved to {csv_file}, {excel_file}, and {json_file}")
        saved_files.extend([csv_file, excel_file, json_file])
    
    return saved_files

def analyze_data(df):
    """
    Quick analysis of the scraped data
    """
    if df.empty:
        return
    
    print("\n=== DATA SUMMARY ===")
    print(f"Total councils: {len(df)}")
    print(f"Divisions: {df['Division'].nunique()}")
    print(f"Districts: {df['District'].nunique()}")
    
    print("\n=== COUNCIL CLASSES ===")
    print(df['Class'].value_counts())
    
    print("\n=== TOP 5 BY POPULATION ===")
    top_pop = df.nlargest(5, 'Population_2011')[['Council_Name', 'District', 'Population_2011']]
    print(top_pop.to_string(index=False))
    
    print("\n=== SAMPLE DATA ===")
    print(df.head().to_string(index=False))

def scrape_multiple_urls(urls_file='tabular_links.txt', max_pages_per_url=None, output_format='all'):
    """
    Scrape data from multiple URLs listed in a file
    Each URL gets its own set of output files
    """
    print("🌐 Multi-URL Maharashtra Data Scraper")
    print("=" * 50)
    
    # Read URLs from file
    urls = read_urls_from_file(urls_file)
    if not urls:
        print("❌ No URLs found to scrape!")
        return
    
    print(f"📋 Found {len(urls)} URLs to scrape")
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"scraped_data_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    
    successful_scrapes = 0
    failed_scrapes = 0
    all_results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"🔗 Processing URL {i}/{len(urls)}: {url}")
        print(f"{'='*60}")
        
        try:
            # Scrape data from this URL
            df = scrape_generic_table(url, max_pages_per_url)
            
            if not df.empty:
                # Generate filename from URL
                url_filename = get_filename_from_url(url)
                custom_filename = os.path.join(output_dir, url_filename)
                
                # Save data in specified format(s)
                saved_files = save_data(df, output_format, custom_filename)
                
                if saved_files:
                    successful_scrapes += 1
                    print(f"✅ Successfully scraped {len(df)} records from {url}")
                    
                    # Store results summary
                    result_summary = {
                        'url': url,
                        'records_count': len(df),
                        'columns_count': len(df.columns),
                        'saved_files': saved_files,
                        'status': 'success'
                    }
                    all_results.append(result_summary)
                    
                    # Quick analysis
                    print(f"📊 Data Summary: {len(df)} records, {len(df.columns)} columns")
                    if len(df) > 0:
                        print(f"📋 Sample columns: {list(df.columns[:5])}")
                else:
                    failed_scrapes += 1
                    print(f"❌ Failed to save data for {url}")
            else:
                failed_scrapes += 1
                print(f"❌ No data found for {url}")
                
        except Exception as e:
            failed_scrapes += 1
            print(f"❌ Error processing {url}: {e}")
        
        # Add delay between URLs to be respectful
        if i < len(urls):
            print("⏳ Waiting 2 seconds before next URL...")
            time.sleep(2)
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📊 SCRAPING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful scrapes: {successful_scrapes}")
    print(f"❌ Failed scrapes: {failed_scrapes}")
    print(f"📁 Output directory: {output_dir}")
    
    if all_results:
        total_records = sum(result['records_count'] for result in all_results)
        print(f"📈 Total records scraped: {total_records}")
        
        # Save summary report
        summary_file = os.path.join(output_dir, 'scraping_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Maharashtra Data Scraping Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Scraping completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total URLs processed: {len(urls)}\n")
            f.write(f"Successful scrapes: {successful_scrapes}\n")
            f.write(f"Failed scrapes: {failed_scrapes}\n")
            f.write(f"Total records scraped: {total_records}\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 20 + "\n")
            for i, result in enumerate(all_results, 1):
                f.write(f"{i}. {result['url']}\n")
                f.write(f"   Records: {result['records_count']}\n")
                f.write(f"   Files: {', '.join(result['saved_files'])}\n\n")
        
        print(f"📄 Summary report saved to: {summary_file}")
    
    return all_results

# Main execution
if __name__ == "__main__":
    print("🏛️ Maharashtra Data Scraper - Multi-URL Version")
    print("=" * 50)
    
    # Check if tabular_links.txt exists
    if os.path.exists('tabular_links.txt'):
        print("📋 Found tabular_links.txt - Running multi-URL scraper")
        
        # Scrape all URLs from the file with all improvements
        results = scrape_multiple_urls(
            urls_file='tabular_links.txt',
            max_pages_per_url=None,  # No limit - extract all available pages
            output_format='all'      # Save in CSV, Excel, and JSON formats
        )
        if results:
            print(f"\n🎉 Multi-URL scraping completed successfully!")
            print(f"📊 Processed {len(results)} URLs with data")
        else:
            print("\n❌ Multi-URL scraping failed or found no data")
    
    else:
        print("📋 tabular_links.txt not found - Running single URL scraper")
        print("🔗 Scraping municipal councils data from default URL...")
        
        # Fallback to original single URL scraper
    councils_df = scrape_maharashtra_councils()
    
    if not councils_df.empty:
        # Analyze the data
        analyze_data(councils_df)
        
        # Save in multiple formats
        save_data(councils_df, 'csv')
        save_data(councils_df, 'excel')
        
        print(f"\n✅ Success! Scraped {len(councils_df)} municipal councils from Maharashtra")
    else:
        print("❌ Failed to scrape data. Please check the website structure.")
    
    print("\n" + "="*50)
    print("🏁 Scraping session completed!")

# Additional utility functions
def get_councils_by_district(df, district_name):
    """Get all councils in a specific district"""
    return df[df['District'].str.contains(district_name, case=False, na=False)]

def get_councils_by_class(df, class_type):
    """Get all councils of a specific class"""
    return df[df['Class'] == class_type]

def search_councils(df, search_term):
    """Search councils by name"""
    return df[df['Council_Name'].str.contains(search_term, case=False, na=False)]
