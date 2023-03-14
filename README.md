```bash
git clone https://github.com/myoh0623/SFT
cd SFT
git checkout dockercompose
cd model_inference
python save_model.py
wget https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
ls
mv ImageNetLabels.txt model
cd ..

docker build -t myoh0623/flask_model_inference:0.3  -f Dockerfile.flask .
docker build -t myoh0623/streamlit_front:0.3  -f Dockerfile.streamlit .
docker rm -f $(docker ps -qa)   
# using docker
sudo docker run --name streamlit_app -p 8501:8501 --network net2 myoh0623/streamlit_front:0.3
sudo docker run -d --name flask_app --network net2 myoh0623/flask_model_inference:0.3

# using docker compose
sudo docker compose up 
```