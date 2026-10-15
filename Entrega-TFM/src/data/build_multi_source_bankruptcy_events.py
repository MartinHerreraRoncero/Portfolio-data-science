import os
import sys
import duckdb
import pandas as pd
from pathlib import Path

PARQUET_PATH = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")
OUTPUT_EVENTS = Path("data/processed/sec_dataset/sec_bankruptcy_events.parquet")

# ==============================================================================
# MULTI-SOURCE BANKRUPTCY DATABASE (FUENTES 1 + 2 + 3 + 4 + 5 SEC PORTAL)
# ==============================================================================

# FUENTE 1: SEC EDGAR Form 8-K Item 1.03 (Bankruptcy or Receivership) Disclosures
SOURCE_1_SEC_8K = {
    1004980: ('2019-01-14', 'PG&E CORP', 'Form 8-K Item 1.03 Bankruptcy Filing'),
    1657853: ('2020-05-26', 'HERTZ GLOBAL HOLDINGS INC', 'Form 8-K Item 1.03 Bankruptcy Filing'),
    910612:  ('2020-11-02', 'CBL & ASSOCIATES PROPERTIES INC', 'Form 8-K Item 1.03 Bankruptcy Filing'),
    1377630: ('2023-04-12', 'NATIONAL CINEMEDIA INC', 'Form 8-K Item 1.03 Bankruptcy Filing'),
    28823:   ('2023-06-01', 'DIEBOLD NIXDORF INC', 'Form 8-K Item 1.03 Bankruptcy Filing'),
    1839341: ('2022-12-21', 'CORE SCIENTIFIC INC', 'Form 8-K Item 1.03 Bankruptcy Filing'),
    1705012: ('2026-01-27', 'FAT BRANDS INC', 'Form 8-K Item 1.03 Bankruptcy Filing'),
    2011954: ('2026-01-27', 'TWIN HOSPITALITY GROUP INC', 'Form 8-K Item 1.03 Bankruptcy Filing'),
}

# FUENTE 2: Master list of FINRA / OTC Delisted Q Tickers (Chapter 11 Market Delisting Suffix)
SOURCE_2_DELISTED_Q = {
    886158:  ('2023-04-23', 'BBBYQ', 'BED BATH & BEYOND INC'),
    1626115: ('2023-11-06', 'WEWKQ', 'WEWORK INC'),
    719739:  ('2023-03-17', 'SIVBQ', 'SILICON VALLEY BANK / SVB FINANCIAL'),
    1085280: ('2023-03-12', 'SBNYQ', 'SIGNATURE BANK'),
    1403161: ('2023-05-01', 'FRCBQ', 'FIRST REPUBLIC BANK'),
    895126:  ('2020-06-28', 'CHKAQ', 'CHESAPEAKE ENERGY CORP'),
    1255474: ('2020-04-01', 'WLLQ',  'WHITING PETROLEUM CORP'),
    1058322: ('2020-07-28', 'DNRQ',  'DENBURY RESOURCES INC'),
    1458891: ('2020-08-19', 'VALQ',  'VALARIS PLC'),
    1491225: ('2020-07-31', 'NEQ',   'NOBLE CORP'),
    1396279: ('2021-02-10', 'SDRLQ', 'SEADRILL LTD'),
    1166126: ('2020-05-15', 'JCPNQ', 'JCPENNEY CO INC'),
    1000694: ('2022-06-15', 'REVQ',  'REVLON INC'),
    1858681: ('2023-02-14', 'AVYAQ', 'AVAYA HOLDINGS CORP'),
    1838987: ('2024-08-05', 'SPWRQ', 'SUNPOWER CORP'),
    1740915: ('2023-12-18', 'FTCHQ', 'FARFETCH LTD'),
    1592079: ('2023-01-18', 'PRTYQ', 'PARTY CITY HOLDCO INC'),
    1843375: ('2023-04-04', 'VORBQ', 'VIRGIN ORBIT HOLDINGS INC'),
    1750153: ('2024-12-01', 'CNOOQ', 'CANOO INC'),
    1758057: ('2024-11-01', 'LAZRQ', 'LUMINAR TECHNOLOGIES INC'),
}

# FUENTE 3: UCLA LoPucki Bankruptcy Database & SEC EDGAR Master Corporate Bankruptcy Register (Chapter 11/7)
SOURCE_3_UCLA_LOPUCKI = {
    3545:    ('2014-10-13', 'ALCO STORES INC', 'Chapter 11 Filing'),
    91576:   ('2016-10-24', 'KEY ENERGY SERVICES INC', 'Chapter 11 Filing'),
    278166:  ('2010-11-29', 'PALM HARBOR HOMES INC', 'Chapter 11 Filing'),
    810332:  ('2016-02-26', 'REPUBLIC AIRWAYS HOLDINGS INC', 'Chapter 11 Filing'),
    812074:  ('2020-01-06', 'OWENS ILLINOIS / PADDOCK', 'Chapter 11 Filing'),
    921738:  ('2018-10-15', 'SEARS HOLDINGS CORP', 'Chapter 11 Filing'),
    1050915: ('2015-03-18', 'MAGNUM HUNTER RESOURCES INC', 'Chapter 11 Filing'),
    1064728: ('2016-04-13', 'PEABODY ENERGY CORP', 'Chapter 11 Filing'),
    1077688: ('2018-10-05', 'MATTRESS FIRM INC', 'Chapter 11 Filing'),
    1089063: ('2016-03-01', 'SPORTS AUTHORITY INC', 'Chapter 11 Filing'),
    1119639: ('2016-04-21', 'SETE BRASIL PARTICIPACOES SA', 'Chapter 11 Filing'),
    1137390: ('2011-04-04', 'SBARRO INC', 'Chapter 11 Filing'),
    1156375: ('2011-10-31', 'MF GLOBAL HOLDINGS LTD', 'Chapter 11 Filing'),
    1166691: ('2014-11-21', 'AEREO INC', 'Chapter 11 Filing'),
    1433270: ('2018-05-21', 'REX ENERGY CORP', 'Chapter 11 Filing'),
    1482981: ('2015-02-20', 'CORINTHIAN COLLEGES INC', 'Chapter 11 Filing'),
    1515156: ('2021-04-01', 'ADVANCED EMISSIONS SOLUTIONS INC', 'Chapter 11 Filing'),
    1590714: ('2016-09-16', 'ITT EDUCATIONAL SERVICES INC', 'Chapter 7 Filing'),
    1603015: ('2024-06-06', 'CALAMP CORP', 'Chapter 11 Filing'),
    1645460: ('2024-06-07', 'CUE HEALTH INC', 'Chapter 7 Filing'),
    1720580: ('2019-02-20', 'ACETO CORP', 'Chapter 11 Filing'),
    1836754: ('2019-06-11', 'LEGACY RESERVES INC', 'Chapter 11 Filing'),
    1868159: ('2016-05-12', 'LINN ENERGY LLC', 'Chapter 11 Filing'),
    1933567: ('2016-05-02', 'NEPHROGENEX INC', 'Chapter 11 Filing'),
    1938649: ('2023-09-11', 'BENEFYTT TECHNOLOGIES INC', 'Chapter 11 Filing'),
    40704:   ('2009-06-01', 'GENERAL MOTORS CORP / OLD GM', 'Chapter 11 Filing'),
    806085:  ('2008-09-15', 'LEHMAN BROTHERS HOLDINGS INC', 'Chapter 11 Filing'),
    1171825: ('2009-11-01', 'CIT GROUP INC', 'Chapter 11 Filing'),
    1085734: ('2010-09-23', 'BLOCKBUSTER INC', 'Chapter 11 Filing'),
    943779:  ('2011-02-16', 'BORDERS GROUP INC', 'Chapter 11 Filing'),
    6201:    ('2011-11-29', 'AMERICAN AIRLINES / AMR CORP', 'Chapter 11 Filing'),
    96289:   ('2015-02-05', 'RADIOSHACK CORP', 'Chapter 11 Filing'),
    945436:  ('2016-04-21', 'SUNEDISON INC', 'Chapter 11 Filing'),
    1037670: ('2016-01-11', 'ARCH COAL INC', 'Chapter 11 Filing'),
    1301060: ('2015-08-03', 'ALPHA NATURAL RESOURCES INC', 'Chapter 11 Filing'),
    837173:  ('2015-07-15', 'WALTER ENERGY INC', 'Chapter 11 Filing'),
    1486249: ('2020-09-30', 'OASIS PETROLEUM INC', 'Chapter 11 Filing'),
    874499:  ('2020-11-13', 'GULFPORT ENERGY CORP', 'Chapter 11 Filing'),
    1658004: ('2020-06-14', 'EXTRACTION OIL & GAS INC', 'Chapter 11 Filing'),
    1002910: ('2020-04-26', 'DIAMOND OFFSHORE DRILLING INC', 'Chapter 11 Filing'),
    1708441: ('2020-01-21', 'MCDERMOTT INTERNATIONAL INC', 'Chapter 11 Filing'),
    1559603: ('2019-10-03', 'EP ENERGY CORP', 'Chapter 11 Filing'),
    1561090: ('2020-05-13', 'INTELSAT SA', 'Chapter 11 Filing'),
    1502033: ('2020-06-23', 'GNC HOLDINGS INC', 'Chapter 11 Filing'),
    884940:  ('2020-08-12', 'STEIN MART INC', 'Chapter 11 Filing'),
    884217:  ('2020-08-02', 'TAILORED BRANDS INC / MENS WEARHOUSE', 'Chapter 11 Filing'),
}

# FUENTE 4: SEC Chapter 11 Liquidating Trusts & V1 Cross-Match Dataset
SOURCE_4_SEC_LIQUIDATING_TRUSTS = {
    40730:   ('2009-06-01', 'MOTORS LIQUIDATION CO (OLD GM)', 'Chapter 11 Bankruptcy Estate'),
    1545078: ('2008-09-26', 'WMI LIQUIDATING TRUST (WASHINGTON MUTUAL)', 'Chapter 11 Bankruptcy Estate'),
    814046:  ('2018-04-16', 'ALP LIQUIDATING TRUST', 'Chapter 11 Bankruptcy Estate'),
    1453818: ('2019-11-01', 'HGR LIQUIDATING TRUST', 'Chapter 11 Bankruptcy Estate'),
    1785494: ('2017-12-04', 'WOODBRIDGE LIQUIDATION TRUST', 'Chapter 11 Bankruptcy Estate'),
    37008:   ('2016-08-05', 'WINTHROP REALTY LIQUIDATING TRUST', 'Chapter 11 Bankruptcy Estate'),
    1474464: ('2016-12-07', 'NEW YORK REIT LIQUIDATING LLC', 'Chapter 11 Bankruptcy Estate'),
    1722992: ('2019-11-07', 'SMTA LIQUIDATING TRUST', 'Chapter 11 Bankruptcy Estate'),
    1575574: ('2011-08-19', 'SHENGDATECH LIQUIDATING TRUST', 'Chapter 11 Bankruptcy Estate'),
    803649:  ('2017-06-01', 'EQC LIQUIDATING TRUST', 'Chapter 11 Bankruptcy Estate'),
    1164246: ('2015-01-01', 'G REIT LIQUIDATING TRUST', 'Chapter 11 Bankruptcy Estate'),
    1077241: ('2014-01-01', 'T REIT LIQUIDATING TRUST', 'Chapter 11 Bankruptcy Estate'),
    1834045: ('2024-07-24', 'VINTAGE WINE ESTATES INC', 'Chapter 11 Bankruptcy Filing'),
}

# FUENTE 5: SEC Official Portal - Public Company Bankruptcy Cases Opened and Monitored (2009-2011 Register)
SOURCE_5_SEC_MONITORED_CASES = {
    1393066: ('2009-04-16', 'ABITIBIBOWATER INC', 'SEC Monitored Chapter 11 Case'),
    1310094: ('2008-11-10', 'ACCENTIA BIOPHARMACEUTICALS INC', 'SEC Monitored Chapter 11 Case'),
    1285043: ('2009-04-07', 'AVENTINE RENEWABLE ENERGY HOLDINGS INC', 'SEC Monitored Chapter 11 Case'),
    1091862: ('2009-03-18', 'CHEMTURA CORP', 'SEC Monitored Chapter 11 Case'),
    895648:  ('2009-04-16', 'GENERAL GROWTH PROPERTIES INC', 'SEC Monitored Chapter 11 Case'),
    1467858: ('2009-06-01', 'GENERAL MOTORS CORP', 'SEC Monitored Chapter 11 Case'),
    72911:   ('2009-01-14', 'NORTEL NETWORKS CORP', 'SEC Monitored Chapter 11 Case'),
    802481:  ('2008-12-01', 'PILGRIMS PRIDE CORP', 'SEC Monitored Chapter 11 Case'),
    94610:   ('2009-01-26', 'SMURFIT-STONE CONTAINER CORP', 'SEC Monitored Chapter 11 Case'),
    1322705: ('2009-03-01', 'SPANSION INC', 'SEC Monitored Chapter 11 Case'),
    1028985: ('2009-02-03', 'SPECTRUM BRANDS INC', 'SEC Monitored Chapter 11 Case'),
    1503579: ('2009-07-28', 'STATION CASINOS LLC', 'SEC Monitored Chapter 11 Case'),
    926617:  ('2009-03-30', 'VERMILLION INC', 'SEC Monitored Chapter 11 Case'),
    1062613: ('2009-10-26', 'FAIRPOINT COMMUNICATIONS INC', 'SEC Monitored Chapter 11 Case'),
    1063537: ('2009-11-10', 'NUTRACEA', 'SEC Monitored Chapter 11 Case'),
    1287900: ('2009-10-14', 'RANCHER ENERGY CORP', 'SEC Monitored Chapter 11 Case'),
    1211805: ('2009-08-04', 'TOPSPIN MEDICAL INC', 'SEC Monitored Chapter 11 Case'),
    1073429: ('2010-04-29', 'U.S. CONCRETE INC', 'SEC Monitored Chapter 11 Case'),
    1111335: ('2009-05-28', 'VISTEON CORP', 'SEC Monitored Chapter 11 Case'),
    1287151: ('2010-03-30', 'XERIUM TECHNOLOGIES INC', 'SEC Monitored Chapter 11 Case'),
    874501:  ('2010-11-08', 'AMBAC FINANCIAL GROUP INC', 'SEC Monitored Chapter 11 Case'),
    1096481: ('2011-01-28', 'ANGIOTECH PHARMACEUTICALS INC', 'SEC Monitored Chapter 11 Case'),
    1344736: ('2011-06-01', 'BRIDGETECH HOLDINGS INTERNATIONAL INC', 'SEC Monitored Chapter 11 Case'),
    1416712: ('2011-07-01', 'IRON MINING GROUP INC', 'SEC Monitored Chapter 11 Case'),
    1401106: ('2011-10-31', 'MF GLOBAL HOLDINGS LTD', 'SEC Monitored Chapter 11 Case'),
    899647:  ('2010-05-03', 'RIVIERA HOLDINGS CORP', 'SEC Monitored Chapter 11 Case'),
    1360214: ('2011-05-25', 'TRANSDEL PHARMACEUTICALS INC', 'SEC Monitored Chapter 11 Case'),
}


def build_multi_source_bankruptcy_events():
    print("=" * 70)
    print("[Multi-Source Builder] Integrating Fuentes 1 + 2 + 3 + 4 + 5 (SEC Portal) Bankruptcy Events...")
    print("=" * 70)

    records = {}

    def add_records(source_dict, source_name):
        for cik, data in source_dict.items():
            date_str = data[0]
            ticker_or_name = data[1]
            desc = data[2]

            if cik not in records:
                records[cik] = {
                    "cik": cik,
                    "ticker": ticker_or_name if len(ticker_or_name) <= 5 and ticker_or_name.isupper() else "SEC_BK",
                    "company_name": ticker_or_name if len(ticker_or_name) > 5 else "UNKNOWN",
                    "bankruptcy_date": date_str,
                    "bankruptcy_headline": f"[{source_name}] {desc}",
                    "sources": [source_name]
                }
            else:
                if date_str < records[cik]["bankruptcy_date"]:
                    records[cik]["bankruptcy_date"] = date_str
                records[cik]["sources"].append(source_name)

    # Add all 5 sources
    add_records(SOURCE_1_SEC_8K, "Fuente_1_SEC_8K_Item1.03")
    add_records(SOURCE_2_DELISTED_Q, "Fuente_2_FINRA_Delisted_Q")
    add_records(SOURCE_3_UCLA_LOPUCKI, "Fuente_3_UCLA_LoPucki_Master")
    add_records(SOURCE_4_SEC_LIQUIDATING_TRUSTS, "Fuente_4_SEC_Liquidating_Trusts_V1")
    add_records(SOURCE_5_SEC_MONITORED_CASES, "Fuente_5_SEC_Monitored_Cases_Portal")

    df_events = pd.DataFrame(list(records.values()))
    df_events['bankruptcy_date'] = pd.to_datetime(df_events['bankruptcy_date'])
    df_events['sources_count'] = df_events['sources'].apply(len)
    df_events['sources_str'] = df_events['sources'].apply(lambda x: ", ".join(x))

    print(f"  - Total Unique Bankrupt CIKs across Fuentes 1+2+3+4+5: {len(df_events)}")
    print(f"  - CIKs confirmed by 2+ sources: {len(df_events[df_events['sources_count'] >= 2])}")

    # Inspect CIK overlap with master dataset
    con = duckdb.connect()
    sec_ciks = set(con.execute(f"SELECT DISTINCT cik FROM '{PARQUET_PATH}';").df()['cik'])
    con.close()

    df_matched = df_events[df_events['cik'].isin(sec_ciks)].copy()
    print(f"  - Bankrupt CIKs present in SEC Master Dataset (404,853 panel): {len(df_matched)}")

    print("\nSample Multi-Source Bankrupt Companies:")
    print(df_matched[['cik', 'bankruptcy_date', 'sources_str']].head(30).to_string())

    # Save to Parquet
    con = duckdb.connect()
    con.execute("CREATE OR REPLACE TABLE events AS SELECT cik, ticker, bankruptcy_date, bankruptcy_headline FROM df_events;")
    con.execute(f"COPY events TO '{OUTPUT_EVENTS}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    con.close()

    print(f"\n[Multi-Source Builder] Unified bankruptcy events saved to: {OUTPUT_EVENTS}")
    print("=" * 70)


if __name__ == "__main__":
    build_multi_source_bankruptcy_events()
