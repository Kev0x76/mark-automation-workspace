# V5 OCR Service

This folder is intentionally separate from the V5 verification pipeline.

The CLI contract is:

```powershell
python ocr_service/ocr_cli.py --image path/to/crop.png
```

It prints JSON:

```json
{"spans":[{"text":"3210808","bbox":[0,0,10,10],"confidence":0.9,"source":"ocr_service"}],"events":[]}
```

Supported engines are tried in this order when available:

1. RapidOCR
2. Tesseract
3. PaddleOCR

OCR errors are returned as `events`; the V5 main process should keep running
and must not treat missing OCR as successful field extraction.
