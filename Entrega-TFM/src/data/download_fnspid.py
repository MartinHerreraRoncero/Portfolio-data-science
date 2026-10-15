"""
Módulo de Ingesta y Stream COMPLETO del Dataset Zihan1004/FNSPID desde HuggingFace Hub.
Descarga y procesa la totalidad de los titulares de noticias financieras sin ningún límite de artículos.

Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.
"""

import os
import sys
import urllib.request
import csv
import io
import time
import pandas as pd

OUTPUT_DIR = "data/raw/FNSPID"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fnspid_financial_news.csv")

SOURCES = [
    {
        "name": "All_external.csv",
        "url": "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv",
        "col_date": 0,
        "col_title": 1,
        "col_symbol": 2,
        "col_publisher": 4
    },
    {
        "name": "nasdaq_exteral_data.csv",
        "url": "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv",
        "col_date": 1,
        "col_title": 2,
        "col_symbol": 3,
        "col_publisher": 5
    }
]

def parse_year(date_str):
    """Extrae el año como entero de una cadena de fecha ISO/UTC."""
    if not date_str:
        return None
    # Intento rápido de substring de 4 dígitos si empieza por YYYY
    try:
        clean_str = date_str.strip()
        if len(clean_str) >= 4 and clean_str[:4].isdigit():
            y = int(clean_str[:4])
            if 1990 <= y <= 2030:
                return y
    except Exception:
        pass
    return None

def download_and_save_fnspid(max_articles=None):
    """
    Realiza streaming HTTP directo sobre los archivos CSV del dataset Zihan1004/FNSPID
    para extraer y guardar la totalidad de noticias financieras en data/raw/FNSPID/fnspid_financial_news.csv.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Iniciando descarga e ingesta COMPLETA de noticias Zihan1004/FNSPID (Sin límite)...")
    
    total_articles = 0
    start_time = time.time()
    
    # Abrir archivo CSV de salida
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['date', 'article_title', 'stock_symbol', 'publisher', 'year'])
        
        for source in SOURCES:
            print(f"\n[INFO] Conectando a {source['name']} ({source['url']})...")
            
            success = False
            for retry in range(5):
                try:
                    req = urllib.request.Request(
                        source['url'], 
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    )
                    resp = urllib.request.urlopen(req, timeout=30)
                    print(f"[INFO] Conexión establecida. Procesando stream de {source['name']}...")
                    
                    # Decodificador incremental para procesar líneas sobre el flujo de bytes
                    decoder = io.StringIO()
                    buffer = ""
                    source_count = 0
                    
                    # Leer en bloques de 4 MB
                    CHUNK_SIZE = 4 * 1024 * 1024
                    while True:
                        chunk_bytes = resp.read(CHUNK_SIZE)
                        if not chunk_bytes:
                            # Procesar cualquier contenido restante en buffer
                            if buffer:
                                reader = csv.reader(io.StringIO(buffer))
                                for row in reader:
                                    if len(row) > max(source['col_date'], source['col_title'], source['col_symbol'], source['col_publisher']):
                                        d_str = row[source['col_date']].strip()
                                        t_str = row[source['col_title']].strip()
                                        s_str = row[source['col_symbol']].strip()
                                        p_str = row[source['col_publisher']].strip()
                                        
                                        year = parse_year(d_str)
                                        if year and t_str:
                                            writer.writerow([d_str, t_str, s_str, p_str, year])
                                            source_count += 1
                                            total_articles += 1
                            break
                        
                        # Convertir bytes a string e ir acumulando líneas
                        text_chunk = chunk_bytes.decode('utf-8', errors='ignore')
                        buffer += text_chunk
                        
                        lines = buffer.split('\n')
                        # La última línea puede estar incompleta; mantenerla en buffer
                        buffer = lines[-1]
                        
                        # Procesar líneas completas
                        completed_text = '\n'.join(lines[:-1])
                        if completed_text:
                            reader = csv.reader(io.StringIO(completed_text))
                            rows_to_write = []
                            for row in reader:
                                # Omitir cabecera si coincide
                                if row and row[0] in ('Date', 'Unnamed: 0'):
                                    continue
                                if len(row) > max(source['col_date'], source['col_title'], source['col_symbol'], source['col_publisher']):
                                    d_str = row[source['col_date']].strip()
                                    t_str = row[source['col_title']].strip()
                                    s_str = row[source['col_symbol']].strip()
                                    p_str = row[source['col_publisher']].strip()
                                    
                                    year = parse_year(d_str)
                                    if year and t_str:
                                        rows_to_write.append([d_str, t_str, s_str, p_str, year])
                                        source_count += 1
                                        total_articles += 1
                                        
                                        if max_articles is not None and total_articles >= max_articles:
                                            break
                            
                            if rows_to_write:
                                writer.writerows(rows_to_write)
                                
                        if source_count % 100000 == 0 and source_count > 0:
                            elapsed = time.time() - start_time
                            print(f"[INFO] [{source['name']}] Ingestados {source_count:,} artículos (Total acumulado: {total_articles:,} | {elapsed:.1f}s)...")
                            
                        if max_articles is not None and total_articles >= max_articles:
                            break
                            
                    print(f"[INFO] Finalizado {source['name']}: {source_count:,} artículos extraídos con éxito.")
                    success = True
                    break
                    
                except Exception as e:
                    print(f"[WARN] Error en intento {retry+1}/5 para {source['name']}: {e}. Reintentando en 3s...")
                    time.sleep(3)
                    
            if max_articles is not None and total_articles >= max_articles:
                break
                
    elapsed_total = time.time() - start_time
    print(f"\n[ÉXITO] Ingesta completada. Total de artículos procesados: {total_articles:,} en {elapsed_total:.1f}s.")
    print(f"[INFO] Archivo guardado en: {OUTPUT_FILE}")
    return OUTPUT_FILE

if __name__ == "__main__":
    download_and_save_fnspid(max_articles=None)
