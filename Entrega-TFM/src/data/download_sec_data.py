import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from tqdm import tqdm

SEC_PAGE_URL = "https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets"
BASE_URL = "https://www.sec.gov"
OUTPUT_DIR = Path("data/raw/sec_zips")
USER_AGENT = "MartinHerrera TFM-Research/1.0 (martinherreraroncero@gmail.com)"


def get_sec_zip_urls():
    """Scrapes the SEC web page and returns full URLs of all financial statement notes dataset ZIPs."""
    req = urllib.request.Request(SEC_PAGE_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8")
        
        rel_links = re.findall(r'href=[\x22\x27]([^\x22\x27]+\.zip)', html)
        urls = []
        for link in rel_links:
            if link.startswith("http"):
                urls.append(link)
            else:
                urls.append(BASE_URL + link)
        # Deduplicate preserving order
        unique_urls = list(dict.fromkeys(urls))
        print(f"[SEC Scraper] Found {len(unique_urls)} dataset ZIP files on SEC page.")
        return unique_urls
    except Exception as e:
        print(f"[SEC Scraper Error] Failed to fetch page: {e}")
        return []


def download_file(url: str, output_path: Path, max_retries: int = 3):
    """Downloads a single ZIP file with retry logic and User-Agent header."""
    if output_path.exists() and output_path.stat().st_size > 10000:
        try:
            import zipfile
            if zipfile.is_zipfile(output_path):
                return True, "skipped"
            else:
                print(f"[Warning] File {output_path.name} is corrupt or incomplete. Redownloading...")
                output_path.unlink()
        except Exception:
            output_path.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as response, open(output_path, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                block_size = 1024 * 64  # 64 KB blocks
                
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name, leave=False) as pbar:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        pbar.update(len(buffer))
            
            # Rate limiting compliance (< 10 requests per second)
            time.sleep(0.25)
            return True, "downloaded"
        except Exception as e:
            print(f"[Warning] Download attempt {attempt} failed for {output_path.name}: {e}")
            if output_path.exists():
                output_path.unlink()  # remove partial file
            time.sleep(2 * attempt)

    return False, "failed"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = get_sec_zip_urls()
    
    if not urls:
        print("[Error] No URLs found to download.")
        return

    success_count = 0
    skipped_count = 0
    failed_urls = []

    print(f"\n[SEC Downloader] Starting download of {len(urls)} ZIP files into '{OUTPUT_DIR}'...\n")

    for i, url in enumerate(urls, 1):
        filename = Path(url).name
        dest_path = OUTPUT_DIR / filename
        print(f"[{i}/{len(urls)}] Processing {filename}...")
        
        ok, status = download_file(url, dest_path)
        if ok:
            if status == "downloaded":
                success_count += 1
            else:
                skipped_count += 1
        else:
            failed_urls.append(url)

    print("\n" + "=" * 50)
    print(f"DOWNLOAD COMPLETE SUMMARY:")
    print(f"  - Total processed: {len(urls)}")
    print(f"  - Newly downloaded: {success_count}")
    print(f"  - Already existed (skipped): {skipped_count}")
    print(f"  - Failed: {len(failed_urls)}")
    if failed_urls:
        print(f"  - Failed URLs: {failed_urls}")
    print("=" * 50)


if __name__ == "__main__":
    main()
