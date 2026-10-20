# নখের রোগ শনাক্তকরণ -- Streamlit Demo

STEP37-এ lock করা দুইটা final model (dataset_1: ConvNeXt-Tiny+CBAM, dataset_2: ConvNeXt-Tiny) দিয়ে একটা upload-করো-ও-predict-পাও demo app।

## লোকালি চালানো

```bash
pip install -r requirements.txt
streamlit run app.py
```

প্রথমবার চালালে `checkpoints/` ফোল্ডারে Google Drive থেকে দুইটা `.pt` ফাইল (~110MB করে) auto-download হয়ে যাবে (`gdown` দিয়ে)। পরের বার আর ডাউনলোড লাগবে না।

## Streamlit Community Cloud-এ deploy করা

1. এই ফোল্ডারটা একটা নতুন public GitHub repo-তে push করো (checkpoints/ ফোল্ডার push করো না -- `.gitignore` এ বাদ দেওয়া আছে, app নিজে থেকেই ডাউনলোড করে নেবে)।
2. [share.streamlit.io](https://share.streamlit.io) -এ গিয়ে GitHub দিয়ে sign in করো।
3. "New app" → তোমার repo, branch (`main`), আর main file path (`app.py`) সিলেক্ট করো।
4. Deploy চাপো -- প্রথম build-এ ৩-৫ মিনিট লাগতে পারে (torch install + প্রথম checkpoint ডাউনলোড)।

## গুরুত্বপূর্ণ নোট

- Preprocessing (224×224 resize, [0,1] normalize) এই পাইপলাইনের STEP41-এর ঠিক same কোড থেকে নেওয়া -- এটা না মিললে accuracy mismatch হবে, তাই `model_defs.py` / `app.py`-এর preprocessing অংশ হাত দিয়ে বদলিও না।
- App চালু হওয়ার সময় checkpoint-এর SHA256 STEP37 lock-record-এর সাথে মিলিয়ে দেখে -- মিসম্যাচ হলে warning দেখাবে।
- এই demo একটা research prototype, ক্লিনিক্যাল ব্যবহারের জন্য না (STEP52, STEP59-এর disclosed limitation অনুযায়ী)।
