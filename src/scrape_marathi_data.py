#!/usr/bin/env python3
"""
Scrape Marathi Normal Web Content using the existing scraper_v1.py
This script scrapes ONLY DMA_Links_MR.txt (normal web content, NOT tabular data)
Uses the robust scraper you already have for Marathi web pages
"""

import os
import sys
import time
from datetime import datetime

# Add src directory to path to import the existing scraper
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scraper_v1 import DMAWebscraper

def create_marathi_links_file():
    """Read ONLY DMA_Links_MR.txt for normal Marathi web content"""
    marathi_links_dir = os.path.join(os.path.dirname(__file__), "Marathi links")
    
    # Read ONLY DMA_Links_MR.txt (normal web content, NOT tabular)
    all_links = []
    
    # Read DMA_Links_MR.txt
    dma_links_file = os.path.join(marathi_links_dir, "DMA_Links_MR.txt")
    if os.path.exists(dma_links_file):
        with open(dma_links_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('http'):
                    all_links.append(line)
    else:
        # Fallback to DMA_Links_MR (without .txt)
        dma_links_file = os.path.join(marathi_links_dir, "DMA_Links_MR")
        if os.path.exists(dma_links_file):
            with open(dma_links_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line.startswith('http'):
                        all_links.append(line)
    
    # NOTE: We do NOT include tabular_links_MR here
    # That will be handled by scrape_marathi_tabular_data.py
    
    # Remove duplicates while preserving order
    unique_links = []
    seen = set()
    for link in all_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    
    # Write combined file
    combined_file = "marathi_combined_links.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for link in unique_links:
            f.write(link + '\n')
    
    print(f"✅ Created {combined_file} with {len(unique_links)} unique Marathi normal web URLs")
    return combined_file, len(unique_links)

def main():
    """Main function to scrape Marathi content"""
    print("🚀 Starting Marathi Content Scraping")
    print("=" * 60)
    
    # Create combined links file
    links_file, total_links = create_marathi_links_file()
    
    # Set up the scraper with Marathi normal data output directory
    scraper = DMAWebscraper(
        base_url="https://mahadma.maharashtra.gov.in",
        output_dir="final_data/marathi_normal_data"  # Matches naming convention
    )
    
    print(f"📁 Output directory: final_data/marathi_normal_data/")
    print(f"🔗 Total Marathi URLs to scrape: {total_links}")
    print(f"⏰ Estimated time: {total_links * 2:.0f} seconds (with 2s delay)")
    print()
    
    # Confirm before starting
    response = input("Do you want to proceed with scraping? (y/N): ")
    if response.lower() != 'y':
        print("Scraping cancelled.")
        return
    
    start_time = time.time()
    
    # Scrape all Marathi URLs (using 2-second delay to be respectful)
    print("🕷️ Starting scraping process...")
    chunks = scraper.scrape_from_file(links_file, delay=2.0)
    
    # Create master files
    print("\n📊 Creating master files...")
    total_chunks = scraper.create_master_files()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 MARATHI SCRAPING COMPLETE!")
    print("=" * 60)
    print(f"⏰ Total time: {elapsed_time/60:.1f} minutes")
    print(f"📁 Output directory: final_data/marathi_normal_data/")
    print(f"📄 Individual files: {len(scraper.scraped_data)} Marathi normal pages scraped")
    print(f"🔗 Total chunks: {total_chunks} chunks for vector embedding")
    print()
    print("📋 Files created:")
    print("   • Individual JSON files: [LINE_NUMBER]_[SLUG].json")
    print("   • scraping_summary.json - Scraping session overview")
    print("   • master_vector_chunks.json - All chunks for Pinecone")
    print("   • master_file_index.json - File directory")
    print()
    print("🚀 Next steps:")
    print("   1. Run: python3 src/advanced_preprocess.py (to process both English + Marathi)")
    print("   2. Run: python3 src/advanced_embed_upsert.py (to embed with multilingual model)")
    print("   3. Test with Marathi queries!")
    
    # Clean up the temporary links file
    try:
        os.remove(links_file)
        print(f"\n🧹 Cleaned up temporary file: {links_file}")
    except:
        pass

if __name__ == "__main__":
    main()
