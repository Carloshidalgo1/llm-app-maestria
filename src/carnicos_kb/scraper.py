from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from carnicos_kb.paths import DEFAULT_DATASET_DIR


DEFAULT_SITEMAP_URL = "https://alimentoscarnicos.com.co/wp-sitemap-posts-page-1.xml"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def get_urls(sitemap_url: str, user_agent: str, timeout: int, verify_tls: bool) -> list[str]:
    headers = {"User-Agent": user_agent}
    response = requests.get(sitemap_url, headers=headers, verify=verify_tls, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]


def slug_from_url(url: str, fallback: str) -> str:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    if not path_parts:
        return fallback
    return path_parts[-1].replace(".html", "").replace(".php", "") or fallback


def scrape_pages(
    output_dir: Path,
    sitemap_url: str,
    user_agent: str,
    timeout: int,
    delay: float,
    verify_tls: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = get_urls(sitemap_url, user_agent, timeout, verify_tls)

    print(f"Paginas encontradas: {len(urls)}")

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    saved_count = 0
    for index, url in enumerate(urls, start=1):
        slug = slug_from_url(url, fallback=f"pagina_{index}")
        filepath = output_dir / f"{slug}.md"

        try:
            response = session.get(url, verify=verify_tls, timeout=timeout)
            if response.status_code != 200:
                print(f"[{index}] HTTP {response.status_code}: {url}")
                continue

            content = trafilatura.extract(response.text, output_format="markdown")
            if not content:
                print(f"[{index}] Sin texto util: {url}")
                continue

            filepath.write_text(content, encoding="utf-8")
            saved_count += 1
            print(f"[{index}] Guardado: {filepath} ({len(content)} caracteres)")
        except Exception as error:
            print(f"[{index}] Error en {url}: {error}")

        time.sleep(delay)

    return saved_count


def parse_args() -> argparse.Namespace:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Scraping masivo desde sitemap XML.")
    parser.add_argument(
        "--sitemap-url",
        default=os.getenv("SITEMAP_URL", DEFAULT_SITEMAP_URL),
        help="URL del sitemap XML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("OUTPUT_DIR", str(DEFAULT_DATASET_DIR))),
        help="Directorio donde se guardan los Markdown extraidos.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("REQUEST_TIMEOUT", "10")),
        help="Timeout de requests en segundos.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("REQUEST_DELAY", "2")),
        help="Espera entre requests en segundos.",
    )
    parser.add_argument(
        "--user-agent",
        default=os.getenv("USER_AGENT", DEFAULT_USER_AGENT),
        help="User-Agent para las peticiones HTTP.",
    )
    parser.add_argument(
        "--verify-tls",
        action="store_true",
        help="Verifica certificados TLS. Por defecto se mantiene desactivado por compatibilidad.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_count = scrape_pages(
        output_dir=args.output_dir,
        sitemap_url=args.sitemap_url,
        user_agent=args.user_agent,
        timeout=args.timeout,
        delay=args.delay,
        verify_tls=args.verify_tls,
    )
    print(f"Total de paginas guardadas: {saved_count}")


if __name__ == "__main__":
    main()

