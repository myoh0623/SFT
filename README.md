```bash
git clone https://github.com/myoh0623/SFT
cd SFT
git checkout network
cd model_inference
python save_model.py
wget https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
ls
mv ImageNetLabels.txt model
cd ..
```