import requests
from bs4 import BeautifulSoup
import trafilatura
import os
import time

def get_urls():
    sitemap_url = "https://alimentoscarnicos.com.co/wp-sitemap-posts-page-1.xml"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        response = requests.get(sitemap_url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        return urls
    except Exception as e:
        print(f"❌ Error al obtener URLs: {e}")
        return []

def scrape_pages():
    urls = get_urls()
    folder = "dataset_carnicos"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    print(f"📂 Iniciando validación de {len(urls)} páginas...")

    # Usamos una sesión de requests para mantener la conexión
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    for i, url in enumerate(urls):
        slug = url.split("/")[-2] if url.endswith("/") else url.split("/")[-1]
        slug = slug or f"pagina_{i}"
        filepath = os.path.join(folder, f"{slug}.md")
        
        try:
            # 1. Descargamos el HTML manualmente para ver si el servidor responde
            resp = session.get(url, verify=False, timeout=10)
            
            if resp.status_code == 200:
                # 2. Intentamos extraer
                content = trafilatura.extract(resp.text, output_format='markdown')
                
                if content:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"✅ [{i+1}] Guardado: {slug}.md ({len(content)} caracteres)")
                else:
                    print(f"⚠️ [{i+1}] El servidor respondió, pero no se encontró texto útil en: {url}")
            else:
                print(f"❌ [{i+1}] Error HTTP {resp.status_code} en: {url}")

        except Exception as e:
            print(f"❌ [{i+1}] Error de conexión en {url}: {e}")
        
        # Pausa un poco más larga para no ser bloqueados
        time.sleep(2)

if __name__ == "__main__":
    scrape_pages()
