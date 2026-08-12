# FAC Codes

Bu klasor, Beltrami ve Finsler tabanli goruntu filtrelerinin Python paketini icerir.

## Dizinler

- `src/`: Python kaynak kodlari
- `inputs/`: ornek giris goruntuleri
- `docs/`: ayrintili kullanim notlari

## Hizli Kurulum

`vendor/FAC-codes` klasorunde:

```powershell
python -m pip install -e .
```

Bu kurulumdan sonra komutlari klasore girmeden de calistirabilirsin:

```powershell
fac-run --metric newmetric --image "vendor/FAC-codes\inputs\Lena.jpg"
fac-ui
```

Detayli kullanim icin:

- [docs/README.md](vendor/FAC-codes\docs\README.md)

