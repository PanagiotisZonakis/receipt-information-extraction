# Receipt Information Extraction

Extracts the **total amount** from a photo of a receipt.

**Pipeline:** photo → YOLOv8 detects the receipt → crop → EasyOCR reads the text → spaCy Matcher finds the Total.

![demo](demo.png)

## Results

| Stage | Metric | Value |
|---|---|---|
| Receipt detection (YOLOv8n) | mAP50 on held-out test split | 0.954 |


The extraction rules were tuned on a separate dev set; the test set was evaluated once at the end.

## How it works

1. **Detection:** YOLOv8n fine-tuned on a Roboflow receipts dataset (50 epochs).
2. **Cropping:** the highest-confidence box is cropped (with small padding) to remove background.
3. **OCR:** EasyOCR (English + German) reads the crop; text boxes are grouped into lines.
4. **Extraction:** a spaCy `Matcher` looks for keywords such as *Total / Summe / Gesamt*, an optional currency code (CHF, EUR) and an amount. Subtotal lines are skipped. A regex fallback handles OCR errors.

## Tools

Python, YOLOv8 (Ultralytics), EasyOCR, spaCy, OpenCV, pandas

## Run it

1. Open `receipt_extraction.ipynb` in Google Colab (GPU runtime).
2. Add your Roboflow API key as a Colab secret named `api_key`.
3. Run all cells.

## Limitations

- Small, single-class training set.
- OCR errors on blurry or crumpled receipts.
- Rules are tuned for receipts that contain a "Total"-like keyword.
- Evaluated on N receipts, so the accuracy estimate is approximate.

## Possible improvements

- Extract date and merchant name.
- Train on more varied receipts.
- Try a layout-aware model (e.g. LayoutLM) instead of rules.
