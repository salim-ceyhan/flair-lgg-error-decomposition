# FAC Codes

Bu klasor, MATLAB'daki Beltrami ve Finsler tabanli goruntu filtrelerinin Python karsiliklarini icerir.

## Klasor Yapisi

- `src/`
  Paket kaynak kok dizini
- `src/fac_filters/`
  Asil Python paketi
- `inputs/`
  Ornek giris goruntuleri
- `docs/`
  Aciklama ve kullanim notlari

## Kaynak Dosyalar

- `src/fac_filters/run_metric.py`
  Birlesik CLI giris noktasi
- `src/fac_filters/metric_ui.py`
  Grafik arayuz giris noktasi
- `src/fac_filters/flow_utils.py`
  Ortak yardimcilar
- `src/fac_filters/analysis_utils.py`
  Histogram ve gorsel analiz uretimi
- `src/fac_filters/NewMetric.py`
  MATLAB karsiligi: `NewMetricFlow.m`
- `src/fac_filters/NMFlow.py`
  MATLAB karsiligi: `NMFlow.m`
- `src/fac_filters/INormalizedMironFlow.py`
  MATLAB karsiligi: `INormalizedMironFlow.m`
- `src/fac_filters/IngardenFlow.py`
  MATLAB karsiligi: `IngardenFlow.m`
- `src/fac_filters/RandersFlowV.py`
  MATLAB karsiligi: `RandersFlowV.m`
- `src/fac_filters/beltrami2D.py`
  MATLAB karsiligi: `beltrami2D.m`

## Kurulum

`vendor/FAC-codes` klasorunde:

```powershell
python -m pip install -e .
```

Bu kurulum sunlari saglar:

- `fac_filters` paketini editable olarak kullanirsin
- `fac-run` komutu olusur
- `fac-ui` komutu olusur
- `python -m fac_filters.run_metric` ve `python -m fac_filters.metric_ui` import hatasi olmadan calisir

Bagimliliklar `pyproject.toml` icinde tanimlidir:

- `numpy`
- `pillow`
- `matplotlib`

## CLI Kullanimi

### Kurulumdan sonra en rahat kullanim

```powershell
fac-run --metric newmetric --image "vendor/FAC-codes\inputs\Lena.jpg"
```

### Paket modulu olarak

```powershell
python -m fac_filters.run_metric --metric newmetric --image "vendor/FAC-codes\inputs\Lena.jpg"
```

### Parametreli ornek

```powershell
fac-run --metric randers --image "vendor/FAC-codes\inputs\Lena.jpg" --beta 0.2 --dt 0.2 --iterations 2
```

### Histogram analizi ile

```powershell
fac-run --metric beltrami --image "vendor/FAC-codes\inputs\Lena.jpg" --save-analysis --no-show
```

### Toplu klasor isleme

```powershell
fac-run --metric nm --input-dir "vendor/FAC-codes\inputs" --recursive --output-subdir "Filtered_nm" --no-show
```

### CSV raporlu batch

```powershell
fac-run --metric ingarden --input-dir "vendor/FAC-codes\inputs" --report-csv "vendor/FAC-codes\outputs\ingarden_report.csv" --no-show
```

## UI Kullanimi

Kurulumdan sonra:

```powershell
fac-ui
```

Alternatif olarak:

```powershell
python -m fac_filters.metric_ui
```

Arayuz uzerinden:

- yontem secebilirsin
- goruntu secebilirsin
- `beta`, `dt`, `iterations` girebilirsin
- cikti klasorunu ayarlayabilirsin
- histogram analizini acabilirsin

## Cikti Davranisi

Varsayilan olarak cikti, giris goruntusunun bulundugu klasorde ayri alt klasore yazilir:

- varsayilan alt klasor: `Filtered`
- cikti adi: `<girdi_adi>_<metric>.png`

Ornek:

```text
inputs\Filtered\Lena_newmetric.png
```

Analiz acik ise ayrica:

```text
inputs\Filtered\Analysis\Lena_newmetric_analysis.png
```

## Desteklenen Metrikler

- `newmetric`
- `nm`
- `inm`
- `ingarden`
- `randers`
- `beltrami`

## Notlar

- `beltrami` klasik Riemannian Beltrami akisidir.
- diger akislar Finsler/Randers/Miron/Ingarden genisletmeleridir.
- hesap yuku yuksektir; buyuk goruntulerde `iterations` dusuk baslayarak denenmelidir.

