import trafilatura
import os

def run_scraper(url, output_file):
    print(f"🕵️ Iniciando extracción de: {url}")
    
    # 1. Descarga y extrae
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        # Usamos format='markdown' para que el LLM lo entienda perfecto después
        content = trafilatura.extract(downloaded, output_format='markdown', include_tables=True)
        
        if content:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ ¡Éxito! Contenido guardado en {output_file}")
        else:
            print("❌ No se pudo extraer contenido legible.")
    else:
        print("❌ Error al descargar la URL.")

if __name__ == "__main__":
    target = "https://alimentoscarnicos.com.co/"
    run_scraper(target, "alimentos_carnicos.md")
